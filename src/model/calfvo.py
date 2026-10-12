""" This module mixes the frozen encoder's tokens across space and time. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from src.model.timesformer import Block, trunc_normal_


class CalfVO(nn.Module):
    """ Frozen encoder, then divided space-time attention, then relative poses.
        The frames of a window are encoded independently by the frozen backbone, given
        a camera token and positional embeddings, mixed across space and time by the
        block stack, and read out by the pose head.
    """

    def __init__(self, patch_size, embed_dim, depth, num_heads, pose_embed_dim,
                 num_frames, regression_head, backbone, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # The token-grid width, used to unflatten the patch axis inside a block.
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Zero-initialised, unlike cls_token: the positional embedding is learned
        # from zero. Sized to the tokens the frozen encoder emits, plus the camera
        # token.
        self.pos_embed = nn.Parameter(torch.zeros(1, pose_embed_dim, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialise before attaching the backbone and the head, never after.
        # self.apply walks the whole tree, so doing this with them attached would
        # re-initialise the frozen CroCo encoder's pretrained weights and override
        # the head's own default init. Both are left as they arrive.
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.backbone = backbone
        self.head = regression_head

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """ Encodes every frame and mixes the tokens across space and time.
        Args:
            x: A window of images of shape (B, channels, T, height, width).
        Returns:
            Tokens of shape (B, 1 + N * T, embed_dim), with the camera token first
            and the rest patch major.
        """
        B, ch, T, height, width = x.shape
        # Every frame through the frozen encoder, batched as (B*T).
        x = self.backbone(x.permute(0, 2, 1, 3, 4).reshape(B * T, ch, height, width))
        W = self.patch_size

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Time embeddings. num_frames is 2 while a window is 8 frames, so this
        # interpolates the two learned entries up to T on every forward. It is the
        # normal path, not a fallback.
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            x = x + new_time_embed.transpose(1, 2)
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        # Patch-major from here on: token 1 + n*T + t is patch n of frame t, which
        # is what the pose head's read-out depends on.
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.training:
            for blk in self.blocks:
                x = checkpoint(blk, x,
                               torch.tensor(B, device=x.device),
                               torch.tensor(T, device=x.device),
                               torch.tensor(W, device=x.device),
                               use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x, B, T, W)

        return self.norm(x)

    def forward(self, x):
        """ Predicts the relative poses between consecutive frames of a window.
        Args:
            x: A window of images of shape (B, channels, T, height, width).
        Returns:
            A dict with relative_poses of shape (B, T - 1, 4, 4) and confidences of
            shape (B, T - 1, 2).
        """
        x = self.forward_features(x)
        # The head runs in fp32: the rotation goes through an SVD, and the
        # log-variances are exponentiated in the loss.
        with autocast("cuda", enabled=False):
            rel_poses, confidences = self.head(x.float())
        if self.training:
            rel_poses = rel_poses.half()
            confidences = confidences.half()
        return {
            "relative_poses": rel_poses,
            "confidences": confidences
        }
