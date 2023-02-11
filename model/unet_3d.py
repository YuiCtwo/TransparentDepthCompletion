import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ELU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels)
    )


def conv_trans_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ELU(inplace=True)
    )


def max_pooling():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


class Unet3D(nn.Module):

    def __init__(self, in_channels, out_channels, num_filters, model_depth=5):
        super(Unet3D, self).__init__()
        self.model_depth = model_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.module_dict = nn.ModuleDict()
        # Encoder: down sampling
        down_rate = 1
        for i in range(self.model_depth):
            if i == 0:
                self.module_dict["down_{}".format(i + 1)] = conv_block(self.in_channels, self.num_filters)
            else:
                self.module_dict["down_{}".format(i + 1)] = conv_block(self.num_filters * down_rate,
                                                                       self.num_filters * down_rate * 2)
                down_rate *= 2
            self.module_dict["pool_{}".format(i + 1)] = max_pooling()

        self.bridge = conv_block(self.num_filters * down_rate, self.num_filters * down_rate * 2)
        self.module_dict["bridge"] = self.bridge
        # Decoder: up sampling
        down_rate *= 2
        up_rate = 2 ** self.model_depth
        for i in range(self.model_depth):
            self.module_dict["up_{}".format(i + 1)] = conv_block(self.num_filters * up_rate,
                                                                 self.num_filters * up_rate // 3)
            self.module_dict["trans_{}".format(i + 1)] = conv_trans_block(self.num_filters * down_rate,
                                                                          self.num_filters * down_rate)
            down_rate //= 2
            up_rate //= 2

        self.out = nn.Sequential(
            nn.Conv3d(self.num_filters, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=self.out_channels),
            nn.ELU(inplace=True)
        )
        self.module_dict["out"] = self.out

    def forward(self, x):
        # Down Sampling
        down_sampling_data = []
        for i in range(self.model_depth):
            x = self.module_dict["down_{}".format(i + 1)](x)
            down_sampling_data.append(x)
            x = self.module_dict["pool_{}".format(i + 1)](x)
        # Bridge
        x = self.bridge(x)
        # Up Sampling
        for i in range(self.model_depth):
            x = self.module_dict["trans_{}".format(i + 1)](x)
            x = torch.cat([x, down_sampling_data[self.model_depth - i - 1]], dim=1)
            x = self.module_dict["up_{}".format(i + 1)](x)
        # Output
        x = self.out(x)
        return x

