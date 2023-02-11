# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_t import SwinTransformer


def build_model(cfg, img_size):
    in_channels = cfg.in_channels
    embed_dim = cfg.embed_dim
    model = SwinTransformer(img_size=img_size,
                            patch_size=4,
                            in_chans=in_channels,
                            embed_dim=embed_dim,
                            depths=[2, 2, 6, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=8,
                            drop_path_rate=0.2)

    return model
