import re
from collections import OrderedDict

import torch
import torch.nn as nn

from torch.autograd import Variable
from einops import rearrange

from model.DM_LRN.blocks import SharedEncoder
from model.pointnet import PointNet
from model.swin_transformer_net import SwinTransformer
from model.DM_LRN.spade import SPADE
from model.rgb_net import PyramidPool
from model.grnet import PointCloudNet
from utils.rgbd2pcd import get_xyz, random_xyz_sampling


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class AdaptiveConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, mask_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        # solve problem: grad and param does not obey, use inplace=False
        self.relu = nn.LeakyReLU(inplace=True)
        self.spade_normalization = SPADE(out_channels, mask_channels, upsample="bilinear")

    def forward(self, x, y):
        x = self.conv(x)
        x = self.spade_normalization(x, y)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adding_channels, mask_channels=None):
        super(DecoderBlock, self).__init__()
        if not mask_channels:
            mask_channels = in_channels // 2
        self.adaptive_block_1 = AdaptiveConvBlock(
            in_channels + adding_channels, out_channels, mask_channels
        )
        self.adaptive_block_2 = AdaptiveConvBlock(
            out_channels, out_channels, mask_channels
        )
        # TODO: adjust pt feature channel
        # self.pt_adjust = nn.Conv2d(pt_channels, adding_channels, kernel_size=3, padding=1, bias=True)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, y, mask_feature, adding_feat):
        # print("q1")
        # print(x)
        x = torch.concat([x, y], dim=1)
        # print("q2")
        # print(x)
        x = self.up(x)
        x = self.adaptive_block_1(x, mask_feature)
        x = x + adding_feat
        x = self.adaptive_block_2(x, mask_feature)
        return x


class SwinVDC(nn.Module):

    def __init__(self, cfg):
        super(SwinVDC, self).__init__()
        self.cfg = cfg
        self.frame_h = cfg.general.frame_h
        self.frame_w = cfg.general.frame_w
        self.predictor_channel = cfg.model.decoder.predictor_channel
        self.mask_encoder_channels = cfg.model.decoder.mask_encoder_channels
        if self.mask_encoder_channels is None:
            self.mask_encoder_channels = 384
        self.depth_norm = cfg.dataset.data_aug.depth_norm
        self.refine_times = cfg.model.decoder.refine_times
        if self.refine_times < 1:
            raise ValueError("refine time must >= 1!")
        self.mask_encode = cfg.model.decoder.mask_encode
        self.running_type = cfg.general.running_type
        self.loss_weight = cfg.loss.weight
        if self.cfg.model.color.type == "Swin-Transformer":
            model_cfg = self.cfg.model.color
            self.transformer_model = SwinTransformer(
                hidden_dim=model_cfg.embed_dim,
                layers=(2, 2, 6, 2),
                heads=(3, 6, 12, 24),
                channels=model_cfg.in_channels,
                head_dim=32,
                window_size=model_cfg.window_size,
                downscaling_factors=(2, 2, 2, 2),
                relative_pos_embedding=True
            )
        else:
            raise NotImplementedError("Unknown color model type: %s" % self.cfg.model.color.type)
        
        if self.cfg.model.pnet.type == "GRNet":
            model_cfg = self.cfg.model.pnet
            self.pt_model = PointCloudNet(
                model_cfg.gridding_scale,
                (cfg.general.frame_h, cfg.general.frame_w),
                model_cfg.num_filters
            )
        elif self.cfg.model.pnet.type == "PointNet":
            model_cfg = self.cfg.model.pnet
            self.pt_model = PointNet(
                (cfg.general.frame_h, cfg.general.frame_w),
                model_cfg.out_channels
            )
        else:
            raise NotImplementedError("Unknown point cloud model type: %s" % self.cfg.model.pnet.type)
        self.decoder_modules = nn.ModuleList()
        self.up_sampling_layer = cfg.model.decoder.up_sampling
        self.up_sampling_layer_num = len(self.up_sampling_layer)
        # mask encoder block
        self.mask_encoder = SharedEncoder(
            out_channels=(
                self.mask_encoder_channels, self.mask_encoder_channels // 2,
                self.mask_encoder_channels // 4, self.mask_encoder_channels // 8
            ),
            scales=(16, 8, 4, 2),
            upsample="bilinear",
            kernel_size=3
        )
        # for i in range(self.up_sampling_layer_num):
        #     self.up_sampling_layer[i].append(self.mask_encoder_channels // (2 ** i))
        if self.mask_encode:
            for i in range(self.up_sampling_layer_num):
                self.decoder_modules.append(DecoderBlock(*(self.up_sampling_layer[i])))
        else:
            raise NotImplementedError("Must have a mask encoder")
            # for i in range(self.up_sampling_layer_num):
            #     self.decoder_modules.append(DecoderBlockWithoutMask(*(self.up_sampling_layer[i])))

        self.predictor = nn.Sequential(
            nn.Conv2d(self.predictor_channel, self.predictor_channel // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.predictor_channel // 4, self.predictor_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.predictor_channel // 4, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, batch):
        pt = batch["pt"]
        xs = self.transformer_model(batch["color"])
        mask = batch["mask"]
        batch_size = mask.size()[0]
        pt_size = pt.size()
        if self.mask_encode:
            mask = mask + 1.0
            mask_features = self.mask_encoder(mask)
        else:
            mask_features = [None for _ in range(self.up_sampling_layer_num)]
        res = batch["raw_depth"]
        for t in range(self.refine_times, 0, -1):
            x = xs[0]
            ps = self.pt_model(pt)
            for i in range(self.up_sampling_layer_num):
                if i != self.up_sampling_layer_num - 1:
                    x = self.decoder_modules[i](x, ps[i], mask_features[i], xs[i + 1])
                else:
                    x = self.decoder_modules[i](x, ps[i], mask_features[i], 0)
            res = self.predictor(x)
            if t > 1:
                with torch.no_grad():
                    new_depth = batch["raw_depth"].clone()
                    new_depth[batch["mask"] > 0] = res[batch["mask"] > 0]
                    pt = get_xyz(new_depth, batch["fx"], batch["fy"], batch["cx"], batch["cy"])
                    pt_tmp = torch.zeros(pt_size).to(mask.device)
                    for b in range(batch_size):
                        pt_tmp[b, :, :] = random_xyz_sampling(pt[b, :, :, :])
                pt = pt_tmp.clone()
        if self.depth_norm:
            res = res * batch["depth_scale"]
        return res


