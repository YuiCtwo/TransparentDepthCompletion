from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CRPBlock(nn.Module):
    def conv1x1(self, in_planes, out_planes, stride=1, bias=False):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                         padding=0, bias=bias)

    def __init__(
            self, in_planes, out_planes, n_stages=4
    ):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self, '{}_{}'.format(i + 1, 'crp'),
                self.conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes, stride=1, bias=False
                )
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'crp'))(top)
            x = top + x
        return x


class FusionBlock(nn.Module):
    def __init__(
            self, hidden_dim, small_planes, upsample="bilinear",
    ):
        super(FusionBlock, self).__init__()
        self.act = nn.LeakyReLU(0.2, True)
        self.upsample = upsample
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True)
        self.conv2 = nn.Conv2d(small_planes, hidden_dim, 1, bias=True)

    def forward(self, input1, input2):
        x1 = self.conv1(input1)
        x2 = F.interpolate(
            self.conv2(input2), size=x1.size()[-2:], mode=self.upsample, align_corners=True
        )
        return self.act(x1 + x2)

    # def forward(self, mask):
    #
    #     mask = F.interpolate(
    #         mask, scale_factor=1. / self.scale, mode=self.upsample
    #     )
    #     if self.round:
    #         mask = torch.round(mask).float()
    #
    #     x = mask
    #     for conv, act in zip(self.convs, self.acts):
    #         x = conv(x)
    #         x = act(x)
    #     return x


class SharedEncoder(nn.Module):
    def __init__(
            self, out_channels, scales, in_channels=1, kernel_size=3, upsample="bilinear"):
        super(SharedEncoder, self).__init__()
        self.scales = scales
        self.upsample = upsample
        self.feature_extractor = nn.Sequential(*[
            nn.Conv2d(in_channels, 32, kernel_size, padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size, padding=(kernel_size - 1) // 2),
            nn.LeakyReLU(0.2, True)
        ])

        self.predictors = []
        for oup in out_channels:
            self.predictors.append(
                nn.Sequential(*[
                    nn.Conv2d(64, oup, kernel_size=3, padding=0),
                    nn.LeakyReLU(0.2, True)
                ])
            )
        self.predictors = nn.ModuleList(self.predictors)

    def forward(self, x):
        features = self.feature_extractor(x)
        res = []
        for it, scale in enumerate(self.scales):
            features_scaled = F.interpolate(features, scale_factor=1. / scale, mode=self.upsample, align_corners=True, recompute_scale_factor=True)
            res.append(
                self.predictors[it](features_scaled)
            )
        return tuple(res)


class AdaptiveBlock(nn.Module):
    def __init__(
            self, x_in_ch, x_out_ch, y_ch, modulation, upsample='bilinear'
    ):
        super(AdaptiveBlock, self).__init__()

        x_hidden_ch = min(x_in_ch, x_out_ch)
        self.learned_res = x_in_ch != x_out_ch

        if self.learned_res:
            self.residual = nn.Conv2d(x_in_ch, x_out_ch, kernel_size=1, bias=False)

        self.modulation1 = modulation(x_ch=x_in_ch, y_ch=y_ch, upsample=upsample)
        self.act1 = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Conv2d(x_in_ch, x_hidden_ch, kernel_size=3, padding=1, bias=True)
        self.modulation2 = modulation(x_ch=x_hidden_ch, y_ch=y_ch, upsample=upsample)
        self.act2 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(x_hidden_ch, x_out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x, skip):
        if self.learned_res:
            res = self.residual(x)
        else:
            res = x

        x = self.modulation1(x, skip)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.modulation2(x, skip)
        x = self.act2(x)
        x = self.conv2(x)

        return x + res
