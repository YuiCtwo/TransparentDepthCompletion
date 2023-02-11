# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch

from .swin_t import SwinTransformer


def build_model(cfg, img_size):
    in_channels = cfg.in_channels
    embed_dim = cfg.embed_dim
    window_size = cfg.window_size
    model = SwinTransformer(img_size=img_size,
                            patch_size=4,
                            in_chans=in_channels,
                            embed_dim=embed_dim,
                            depths=[2, 2, 6, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=window_size,
                            drop_path_rate=0.2)
    if cfg.pretrained_model:
        snapshot = torch.load(cfg.pretrained_model, map_location=torch.device("cpu"))
        if in_channels != 3:
            # remove pretrained layer: patch_embed
            for k in list(snapshot["model"].keys()):
                if "patch_embed" in k:
                    snapshot["model"].pop(k)
        model.load_state_dict(snapshot["model"])
    if cfg.frozen_param:
        for p in model.parameters():
            p.requires_grad = False
    return model
