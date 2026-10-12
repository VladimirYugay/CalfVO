""" This module builds the model and exposes its parts. """
from functools import partial

import torch

from src.model.backbone import get_backbone
from src.model.calfvo import CalfVO
from src.model.head import PoseHead

__all__ = ["CalfVO", "PoseHead", "get_backbone", "get_model"]


def get_model(config: dict):
    """ Builds the frozen CroCo encoder, the divided space-time CalfVO and the PoseHead.
        Only one model is released, so there is nothing to dispatch on. The config
        supplies its shape, meaning embed_dim, depth and the token grid, along with
        which encoder blocks to freeze. It does not select components.
    Args:
        config: The model section of the run config, with num_views added by the
            caller.
    Returns:
        The assembled CalfVO.
    """
    # The head reads out one descriptor per frame plus the camera token, and emits
    # one relative pose per consecutive pair, hence num_views - 1.
    regression_head = PoseHead(config["embed_dim"], config["num_views"] - 1)

    backbone = get_backbone(config.get("backbone", "crocov2_dust3r"),
                            config.get("blocks_to_freeze", ()),
                            config["img_size"])

    # One position per token the frozen encoder emits, plus the camera token.
    # patch_size in these configs is the token grid width, so CroCo is 14 and
    # DINOv2 is 16 at the same resolution.
    pose_embed_dim = config["patch_size"] ** 2 + 1

    # num_frames is 2 rather than 8 on purpose: the time embedding's two learned
    # entries are interpolated up to the window length on every forward.
    return CalfVO(
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=4,
        pose_embed_dim=pose_embed_dim,
        num_frames=2,
        regression_head=regression_head,
        backbone=backbone,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
