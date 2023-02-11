from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from cfg.model_zoo import model_efficientnet

from .efficientnet import EfficientNet
from .blocks import AdaptiveBlock, FusionBlock, CRPBlock, SharedEncoder
from .spade import *


def build_efficientnet(model_str="efficientnet-b4", use_pretrained=True):
    efficientnet = EfficientNet(model_str)
    if use_pretrained:
        state_dict = load_state_dict_from_url(model_efficientnet[model_str], progress=True)
        efficientnet.load_state_dict(state_dict, strict=False)
    return efficientnet


# Use DM_LRN with efficientnet-b4 backbone
class DM_LRN(nn.Module):
    def __init__(self,
                 extract_feature=True,
                 predict_depth=False,
                 depth_max_min=False):
        super(DM_LRN, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=7, padding=3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.extract_feature = extract_feature
        self.predict_depth = predict_depth
        self.backbone = build_efficientnet(use_pretrained=False)
        self.feature_channels = self.backbone.feature_channels
        self.modulation = SPADE
        self.channels = 256
        self.upsample = "bilinear"
        self.use_crp = True
        self.mask_encoder_ksize = 3
        self.depth_max_min = depth_max_min

        self.modulation32 = AdaptiveBlock(
            self.channels, self.channels, self.channels,
            modulation=self.modulation,
            upsample=self.upsample
        )
        self.modulation16 = AdaptiveBlock(
            self.channels // 2, self.channels // 2, self.channels // 2,
            modulation=self.modulation,
            upsample=self.upsample
        )
        self.modulation8 = AdaptiveBlock(
            self.channels // 4, self.channels // 4, self.channels // 4,
            modulation=self.modulation,
            upsample=self.upsample
        )
        self.modulation4 = AdaptiveBlock(
            self.channels // 8, self.channels // 8, self.channels // 8,
            modulation=self.modulation,
            upsample=self.upsample
        )

        self.modulation4_1 = AdaptiveBlock(
            self.channels // 8, self.channels // 16, self.channels // 8,
            modulation=self.modulation,
            upsample=self.upsample
        )
        self.modulation4_2 = AdaptiveBlock(
            self.channels // 16, self.channels // 16, self.channels // 16,
            modulation=self.modulation,
            upsample=self.upsample
        )

        self.mask_encoder = SharedEncoder(
            out_channels=(
                self.channels, self.channels // 2, self.channels // 4,
                self.channels // 8, self.channels // 8, self.channels // 16
            ),
            scales=(32, 16, 8, 4, 2, 1),
            upsample=self.upsample,
            kernel_size=self.mask_encoder_ksize
        )

        self.fusion_32x16 = FusionBlock(self.channels // 2, self.channels, upsample=self.upsample)
        self.fusion_16x8 = FusionBlock(self.channels // 4, self.channels // 2, upsample=self.upsample)
        self.fusion_8x4 = FusionBlock(self.channels // 8, self.channels // 4, upsample=self.upsample)

        self.adapt1 = nn.Conv2d(self.feature_channels[-1], self.channels, 1, bias=False)
        self.adapt2 = nn.Conv2d(self.feature_channels[-2], self.channels // 2, 1, bias=False)
        self.adapt3 = nn.Conv2d(self.feature_channels[-3], self.channels // 4, 1, bias=False)
        self.adapt4 = nn.Conv2d(self.feature_channels[-4], self.channels // 8, 1, bias=False)

        if self.use_crp:
            self.crp1 = CRPBlock(self.channels, self.channels)
            self.crp2 = CRPBlock(self.channels // 2, self.channels // 2)
            self.crp3 = CRPBlock(self.channels // 4, self.channels // 4)
            self.crp4 = CRPBlock(self.channels // 8, self.channels // 8)

        self.predictor = nn.Sequential(*[
            nn.Conv2d(self.channels // 16, self.channels // 16, 1, padding=0, groups=self.channels // 16),
            nn.Conv2d(self.channels // 16, 1, 3, padding=1)
        ])

    def forward(self, batch):

        color, raw_depth, mask = batch["color"], batch["raw_depth"], batch["mask"]

        x = torch.cat([color, raw_depth], dim=1)
        mask = mask + 1.0
        x = self.stem(x)
        features = self.backbone(x)[::-1]
        if self.use_crp:
            f1 = self.crp1(self.adapt1(features[0]))
        else:
            f1 = self.adapt1(features[0])
        f2 = self.adapt2(features[1])
        f3 = self.adapt3(features[2])
        f4 = self.adapt4(features[3])

        mask_features = self.mask_encoder(mask)

        x = self.modulation32(f1, mask_features[0])
        x = self.fusion_32x16(f2, x)
        x = self.crp2(x) if self.use_crp else x

        x = self.modulation16(x, mask_features[1])
        x = self.fusion_16x8(f3, x)
        x = self.crp3(x) if self.use_crp else x

        x = self.modulation8(x, mask_features[2])
        x = self.fusion_8x4(f4, x)
        x = self.crp4(x) if self.use_crp else x

        x = self.modulation4(x, mask_features[3])

        x = F.interpolate(x, scale_factor=2, mode=self.upsample, align_corners=True)
        x = self.modulation4_1(x, mask_features[4])
        x = F.interpolate(x, scale_factor=2, mode=self.upsample, align_corners=True)
        x = self.modulation4_2(x, mask_features[5])
        if not self.extract_feature:
            x = self.predictor(x)
            if self.depth_max_min:
                x = x * batch["depth_scale"] + batch["depth_min"]
            if not self.predict_depth:
                x = torch.cat([color, x], dim=1)
        return x
