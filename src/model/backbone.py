""" This module builds the frozen encoder the model runs on. """
import torch
import torch.nn.functional as F
from croco.models.blocks import Attention
from dust3r.model import AsymmetricCroCo3DStereo


CROCO_DUST3R_WEIGHTS = "naver/DUSt3R_ViTLarge_BaseDecoder_224_linear"
DINOV2_WEIGHTS = "facebook/dinov2-large"


class EfficientAttention(Attention):
    """ CroCo's attention with the manual softmax replaced by a fused kernel.
        scaled_dot_product_attention picks a flash or memory efficient backend, which
        matters here because the encoder runs once per frame of every window.
    """

    def forward(self, x, xpos):
        """ Applies rotary position embeddings and fused attention.
        Args:
            x: Tokens of shape (B, N, C).
            xpos: Token positions for the rotary embedding, or None.
        Returns:
            The attended tokens, of the same shape as x.
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CroCoBackbone(torch.nn.Module):
    """ The encoder half of a DUSt3R CroCo model, without its decoder.
        Only the patch embedding, the positional embedding, the encoder blocks and the
        final norm are kept, since the model reads patch tokens and never decodes.
    """

    def __init__(self, croco_model, use_flash_attention=True):
        super().__init__()
        self.enc_embed_dim = croco_model.enc_embed_dim
        self.patch_embed = croco_model.patch_embed
        self.enc_pos_embed = croco_model.enc_pos_embed
        if use_flash_attention:
            for blk in croco_model.enc_blocks:
                old_attn = blk.attn

                new_attn = EfficientAttention(
                    dim=old_attn.qkv.in_features,
                    num_heads=old_attn.num_heads,
                    qkv_bias=True,
                    attn_drop=old_attn.attn_drop.p,
                    proj_drop=old_attn.proj_drop.p if hasattr(old_attn, "proj_drop") else 0.,
                    rope=getattr(old_attn, "rope", None))
                new_attn.load_state_dict(old_attn.state_dict(), strict=True)
                blk.attn = new_attn
        self.enc_blocks = croco_model.enc_blocks
        self.enc_norm = croco_model.enc_norm

    def freeze_blocks(self, block_ids: list):
        """ Stops the listed encoder blocks from receiving gradients.
        Args:
            block_ids: Indices into enc_blocks. The released config lists all 24.
        """
        for block_id in block_ids:
            for param in self.enc_blocks[block_id].parameters():
                param.requires_grad = False

    def forward(self, img):
        """ Encodes one batch of images into patch tokens.
        Args:
            img: Images of shape (B, 3, height, width).
        Returns:
            Patch tokens of shape (B, N, enc_embed_dim).
        """
        x, pos = self.patch_embed(img)
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]

        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x


class DinoV2Backbone(torch.nn.Module):
    """ A frozen DINOv2 encoder, emitting one token per patch.
        Every parameter is frozen in the constructor, so blocks_to_freeze does not
        apply to this backbone. DINOv2 expects ImageNet normalisation rather than the
        training mixture's, so a config using it has to set norm_mean and norm_std.
    """

    def __init__(self, dino_model):
        super().__init__()
        self.dino_model = dino_model
        for param in self.dino_model.parameters():
            param.requires_grad = False

    # Compiling this hands Dinov2Model.forward a None last_hidden_state through
    # transformers' encoder output wrapper. Running the frozen encoder eagerly costs
    # nothing, since it has no gradients to fuse.
    @torch.compiler.disable
    def forward(self, img):
        """ Encodes one batch of images into patch tokens.
        Args:
            img: Images of shape (B, 3, height, width).
        Returns:
            Patch tokens of shape (B, N, embed_dim), with the CLS token dropped.
        """
        features = self.dino_model(img)[0]
        return features[:, 1:]


def get_backbone(name="crocov2_dust3r", blocks_to_freeze=(), img_size=(224, 224)):
    """ Builds the frozen encoder the model runs on.
        Weights are fetched from the Hub on first use and cached.
    Args:
        name: Which encoder to build, either crocov2_dust3r or dinov2_large.
        blocks_to_freeze: Encoder blocks to freeze. CroCo only, since the DINOv2
            wrapper freezes everything.
        img_size: Input resolution the encoder is built for. CroCo only.
    Returns:
        The frozen backbone.
    """
    if name == "crocov2_dust3r":
        model = AsymmetricCroCo3DStereo.from_pretrained(
            CROCO_DUST3R_WEIGHTS, img_size=list(img_size))
        backbone = CroCoBackbone(model)
        backbone.freeze_blocks(blocks_to_freeze)
        return backbone

    if name == "dinov2_large":
        from transformers import AutoModel
        return DinoV2Backbone(AutoModel.from_pretrained(DINOV2_WEIGHTS))

    raise ValueError(
        f"Unknown backbone {name!r}. Known names are 'crocov2_dust3r' and "
        "'dinov2_large'.")
