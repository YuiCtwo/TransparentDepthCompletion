import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling
from einops import rearrange


def grnet_conv(in_channels, out_channels, kernel_size=4, padding=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2),
        nn.MaxPool3d(kernel_size=2)
    )


def grnet_dconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=stride, bias=False, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU()
    )


def random_pt_sampling(n_points, pred_cloud, partial_cloud=None):
    if partial_cloud is not None:
        pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

    _ptcloud = torch.split(pred_cloud, 1, dim=0)
    ptclouds = None
    for p in _ptcloud:
        p = p.to(pred_cloud.device)
        non_zeros = torch.sum(p, dim=2).ne(0)
        p = p[non_zeros].unsqueeze(dim=0)
        n_pts = p.size(1)
        if n_pts < n_points:
            rnd_idx = torch.cat([torch.randint(0, n_pts, (n_points,))])
        else:
            rnd_idx = torch.randperm(p.size(1))[:n_points]
        rnd_idx = rnd_idx.to(pred_cloud.device)
        if ptclouds is not None:
            ptclouds = torch.cat([ptclouds, p[:, rnd_idx, :]], dim=0).contiguous()
        else:
            ptclouds = p[:, rnd_idx, :]

    return ptclouds


class PointCloudNet(nn.Module):

    def __init__(self, gridding_scale=64, img_size=(256, 256), num_filters=48):
        super(PointCloudNet, self).__init__()
        self.hws = [
            (img_size[0] // 16, img_size[1] // 16),
            (img_size[0] // 8, img_size[1] // 8),
            (img_size[0] // 4, img_size[1] // 4),
            (img_size[0] // 2, img_size[1] // 2)
        ]
        self.num_samples = [hw[0]*hw[1] for hw in self.hws]
        self.g_scale = gridding_scale
        self.gridding_layer = Gridding(scale=gridding_scale)
        self.gridding_reverse_layer = GriddingReverse(scale=gridding_scale)

        # conv
        self.conv_1 = grnet_conv(1, num_filters)
        self.conv_2 = grnet_conv(num_filters, num_filters * 2)
        self.conv_3 = grnet_conv(num_filters * 2, num_filters * 4)
        self.conv_4 = grnet_conv(num_filters * 4, num_filters * 8)
        # d_conv
        self.dconv_4 = grnet_dconv(num_filters * 8, num_filters * 4)
        self.dconv_3 = grnet_dconv(num_filters * 4, num_filters * 2)
        self.dconv_2 = grnet_dconv(num_filters * 2, num_filters * 1)
        self.dconv_1 = grnet_dconv(num_filters, 1)

        # TODO: try another sampling strategy
        # self.sampling_layer_1 = RandomPointSampling(n_points=num_samples[0])
        # self.sampling_layer_2 = RandomPointSampling(n_points=num_samples[1])
        # self.sampling_layer_3 = RandomPointSampling(n_points=num_samples[2])
        # self.sampling_layer_4 = RandomPointSampling(n_points=num_samples[3])
        self.feature_sampling_layer = CubicFeatureSampling()

    def forward(self, partial_cloud):
        partial_cloud = rearrange(partial_cloud, 'b s n -> b n s')
        pt_features_x = self.gridding_layer(partial_cloud)
        pt_features_x = pt_features_x.view(-1, 1, self.g_scale, self.g_scale, self.g_scale)
        # print(pt_features_x.size())  # [b, 1, 64, 64, 64]
        # Unet-like structure
        pt_features_2x = self.conv_1(pt_features_x)  # [b, 48, 32, 32, 32]
        pt_features_4x = self.conv_2(pt_features_2x)  # [b, 96, 16, 16, 16]
        pt_features_8x = self.conv_3(pt_features_4x)  # [b, 192, 8, 8, 8]
        pt_features_16x = self.conv_4(pt_features_8x)  # [b, 384, 4, 4, 4]
        # print(pt_features_16x.size())
        # TODO: is bridge layer necessary?
        pt_features_8x = self.dconv_4(pt_features_16x) + pt_features_8x  # [b, 192, 8, 8, 8]
        pt_features_4x = self.dconv_3(pt_features_8x) + pt_features_4x  # [b, 96, 16, 16, 16]
        pt_features_2x = self.dconv_2(pt_features_4x) + pt_features_2x  # [b, 48, 32, 32, 32]
        pt_features_x = self.dconv_1(pt_features_2x) + pt_features_x  # [b, 1, 64, 64, 64]
        pt_features_x = self.gridding_reverse_layer(pt_features_x.squeeze(dim=1))
        # sampling point
        x1 = random_pt_sampling(self.num_samples[3], pt_features_x)
        x2 = random_pt_sampling(self.num_samples[2], x1)
        x3 = random_pt_sampling(self.num_samples[1], x2)
        x4 = random_pt_sampling(self.num_samples[0], x3)

        # sampling feature
        x4 = self.feature_sampling_layer(x4, pt_features_16x)
        x3 = self.feature_sampling_layer(x3, pt_features_8x)
        x2 = self.feature_sampling_layer(x2, pt_features_4x)
        x1 = self.feature_sampling_layer(x1, pt_features_2x)
        x4 = rearrange(x4, "b (h w) e c -> b (e c) h w", h=self.hws[0][0], w=self.hws[0][1])
        x3 = rearrange(x3, "b (h w) e c -> b (e c) h w", h=self.hws[1][0], w=self.hws[1][1])
        x2 = rearrange(x2, "b (h w) e c -> b (e c) h w", h=self.hws[2][0], w=self.hws[2][1])
        x1 = rearrange(x1, "b (h w) e c -> b (e c) h w", h=self.hws[3][0], w=self.hws[3][1])
        return x4, x3, x2, x1
