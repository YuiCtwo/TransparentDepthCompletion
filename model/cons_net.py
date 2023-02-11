import torch
import torch.nn as nn
import torch.nn.functional as F
from model.unet_2d import UNet
import numpy as np


class ConsNet(nn.Module):
    def __init__(self, cfg):
        super(ConsNet, self).__init__()
        self.cfg = cfg
        self.cons_net = UNet(4, 4)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(
                    m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, depth, xyz_map):

        # b,_,h,w = depth.size()
        # if nmap.size(3) == 3:
        #     nmap = nmap.permute(0, 3, 1, 2)
        input_features = torch.cat((depth, xyz_map), dim=1)

        output = self.cons_net(input_features) + input_features
        depth = output[:, 0].unsqueeze(1)
        xyz_map = F.normalize(output[:, 1:], dim=1)
        return depth, xyz_map
