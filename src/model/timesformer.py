""" This module defines divided space-time attention blocks, after TimeSformer
(Bertasius et al., ICML 2021). """
import math
import warnings

import torch
import torch.nn as nn
from einops import rearrange


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """ Fills a tensor in place from a truncated normal distribution.
    Args:
        tensor: The tensor to fill.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        a: Lower truncation bound.
        b: Upper truncation bound.
    Returns:
        The same tensor, filled.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """ Drops whole samples from the main path of a residual block.
    Args:
        x: The residual branch's output.
        drop_prob: Probability of dropping a sample.
        training: Whether to drop at all. Inference is always a no-op.
    Returns:
        x with some samples zeroed and the rest rescaled.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Broadcast over any number of trailing dims, not just 2D ConvNets.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """ Module wrapper around drop_path, so it can sit in a block's forward. """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ The standard two-layer transformer MLP with an activation and dropout. """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """ Multi-head self-attention with an explicit softmax.
        The frozen encoder swaps this for EfficientAttention, which uses a fused
        kernel. These blocks are trained from scratch and keep the plain version.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        """ Attends every token to every other token.
        Args:
            x: Tokens of shape (B, N, C).
        Returns:
            The attended tokens, of the same shape as x.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ One divided space-time block, meaning temporal attention, then spatial,
        then the MLP. The camera token is prepended to each frame's spatial sequence,
        so it attends over space only, and its per-frame responses are averaged back
        into a single token after the spatial attention.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0.1, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=drop)

        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        """ Attends over time, then over space, then applies the MLP.
        Args:
            x: Tokens of shape (B, 1 + N * T, dim), with the camera token first.
            B: Batch size.
            T: Frames per window.
            W: Token grid width, used to unflatten the patch axis.
        Returns:
            The updated tokens, of the same shape as x.
        """
        # Gradient checkpointing passes B, T and W as tensors.
        if isinstance(B, torch.Tensor):
            B, T, W = B.item(), T.item(), W.item()
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        # Temporal
        xt = x[:, 1:, :]
        xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
        res_temporal = self.drop_path(
            self.temporal_attn(self.temporal_norm1(xt)))
        res_temporal = rearrange(
            res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x[:, 1:, :] + res_temporal

        # Spatial
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(
            cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        # The camera token saw each frame separately, so average its responses.
        cls_token = res_spatial[:, 0, :]
        cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
        cls_token = torch.mean(cls_token, 1, True)
        res_spatial = res_spatial[:, 1:, :]
        res_spatial = rearrange(
            res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
        res = res_spatial
        x = xt

        # Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
