import re

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

from model.cons_net import ConsNet
from model.pointnet import VPointNet
from model.DM_LRN.spade import SPADE
from model.rgb_net import PyramidPool
from model.unet_2d import UNet
from utils.inverse_warp import inverse_warp_pytorch, inverse_warp_cpp
from utils.normal2depth import normal_to_depth
from utils.rgbd2pcd import get_surface_normal_from_depth, get_xyz, get_surface_normal_from_xyz, random_xyz_sampling


def convtext(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, use_bn=True):
    if use_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )


def adaptive_convtext():
    pass


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes, eps=1e-5)
    )



class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.disp = Variable(
            torch.Tensor(
                np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])
            ), requires_grad=False
        )

    def forward(self, x):
        disp = self.disp.to(x.device)
        disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        return torch.sum(x * disp, 1)


class DFNet(nn.Module):

    def __init__(self, cfg):
        super(DFNet, self).__init__()
        self.cfg = cfg
        self.frame_h = cfg.general.frame_h
        self.frame_w = cfg.general.frame_w
        self.cost_volume_layer_num = 4
        self.depth_norm = cfg.dataset.data_aug.depth_norm
        self.L = cfg.model.depth_plane_num
        self.add_aggregation_residual = False
        if self.cfg.model.color.type == "PyramidPool":
            model_cfg = self.cfg.model.color
            self.rgb_model = PyramidPool(model_cfg.out_channels)
            self.rgb_out_channel = model_cfg.out_channels
        else:
            raise NotImplementedError("Unknown color model type: %s" % self.cfg.model.color.type)
        self.add_pt_feature = True
        if self.cfg.model.pnet.type == "VPointNet":
            model_cfg = self.cfg.model.pnet
            self.pt_model = VPointNet(
                (cfg.general.frame_h, cfg.general.frame_w),
                model_cfg.hidden_dim,
                model_cfg.out_channels
            )
            self.pt_out_channels = model_cfg.out_channels
        else:
            print("do not use pt feature!")
            self.add_pt_feature = False
            self.pt_out_channels = 0

        self.cost_volume_header = nn.Sequential(
            convbn_3d(self.rgb_out_channel * 2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.cost_volume_tail = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.residual_cost_volume = nn.ModuleList()
        for i in range(self.cost_volume_layer_num):
            self.residual_cost_volume.append(
                nn.Sequential(
                    convbn_3d(32, 32, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    convbn_3d(32, 32, 3, 1, 1),
                )
            )

        self.context_network_in_channel = self.rgb_out_channel + 1 + self.pt_out_channels
        self.context_network = nn.Sequential(
            convtext(self.context_network_in_channel, 96, 3, 1, 1),
            convtext(96, 96, 3, 1, 2),
            convtext(96, 96, 3, 1, 4),
            convtext(96, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.depth_regression = DisparityRegression(self.L)
        self.normal_refine = cfg.model.normal.refine
        self.normal_input_dim = cfg.model.normal.input_dim
        if cfg.model.normal.input_dim:
            self.normal_predictor = nn.Sequential(
                convtext(self.normal_input_dim, 8, 3, 1, 1),
                convtext(8, 16, 3, 1, 2),
                convtext(16, 32, 3, 1, 8),
                convtext(32, 16, 3, 1, 16),
                convtext(16, 8, 3, 1, 1),
                convtext(8, 3, 3, 1, 1)
            )
        if self.normal_refine:
            self.refine_weight = cfg.model.normal.refine_weight

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

    def forward(self, batch):
        b, _, h, w = batch["cur_color"].size()
        cur_img_feat = self.rgb_model(batch["cur_color"])
        ref_img_feat = self.rgb_model(batch["forward_color"])

        b, c, hh, ww = ref_img_feat.size()
        device = cur_img_feat.device
        cur_mask = F.interpolate(batch["mask"], size=(hh, ww), mode="bilinear", align_corners=True)
        min_depth = batch["min_depth"][:, :, :hh, :ww]
        raw_depth = F.interpolate(batch["raw_depth"], size=(hh, ww), mode="bilinear", align_corners=True)

        depth_gap = (1 - min_depth) / self.L  # bx1
        depth = Variable(torch.ones(b, 1, hh, ww)).to(device) * min_depth
        # wc = Variable(torch.zeros(b, 3, self.L, hh, ww)).to(device)

        cost = Variable(torch.zeros(b, c * 2, self.L, hh, ww)).to(device)
        for l in range(self.L):
            depth = depth + depth_gap * l
            # predict locally
            depth[~(cur_mask > 0)] = raw_depth[~(cur_mask > 0)]
            cost[:, :c, l, :, :] = cur_img_feat
            cost[:, c:, l, :, :] = inverse_warp_pytorch(depth, ref_img_feat, batch["R_mat"], batch["t_vec"],
                                                        batch["fx"], batch["fy"], batch["cx"], batch["cy"])
            # with torch.no_grad():
            #     wc[:, :, l, :, :] = get_xyz(depth, batch["fx"], batch["fy"], batch["cx"], batch["cy"])

        cost = cost.contiguous()
        cost = self.cost_volume_header(cost)
        for i in range(self.cost_volume_layer_num):
            cost = cost + self.residual_cost_volume[i](cost)
        cost = self.cost_volume_tail(cost)

        refined_cost = Variable(torch.zeros(b, 1, self.L, hh, ww)).to(device)
        # cost aggregation
        if self.add_pt_feature:
            cur_pt_feature = self.pt_model(batch["pt"])
            for i in range(self.L):
                refined_cost[:, :, i, :, :] = self.context_network(
                    torch.cat([cur_img_feat, cost[:, :, i, :, :], cur_pt_feature], 1)
                ) + cost[:, :, i, :, :]
        else:
            for i in range(self.L):
                refined_cost[:, :, i, :, :] = self.context_network(
                    torch.cat([cur_img_feat, cost[:, :, i, :, :]], 1)
                ) + cost[:, :, i, :, :]

        refined_cost_squeezed = torch.squeeze(refined_cost, 1)
        refined_cost_squeezed = F.softmax(
            F.interpolate(refined_cost_squeezed, [h, w], mode='bilinear', align_corners=True), dim=1
        )
        pred = self.depth_regression(refined_cost_squeezed)
        depth = batch["min_depth"] + ((1 - batch["min_depth"]) / self.L) * pred.unsqueeze(1)

        if not self.training:
            depth = depth * batch["depth_scale"]
            return depth

        if self.normal_input_dim:
            nmap = Variable(torch.zeros(b, 3, hh, ww)).to(device)
            for l in range(self.L):
                nmap = nmap + self.normal_predictor(refined_cost[:, :, l, :, :])
            nmap = F.interpolate(nmap, [h, w], mode='bilinear', align_corners=True)
            nmap = F.normalize(nmap, dim=1)
            if self.normal_refine:
                depth[~(batch["mask"] > 0)] = batch["raw_depth"][~(batch["mask"] > 0)]
                pt_map = get_xyz(depth, batch["fx"], batch["fy"], batch["cx"], batch["cy"]).float()
                refined_depth = normal_to_depth(pt_map, nmap, batch["fx"], batch["fy"], batch["cx"], batch["cy"])
                refined_depth = self.refine_weight * refined_depth + (1 - self.refine_weight) * depth
                refined_depth = refined_depth * batch["depth_scale"]
                return refined_depth, nmap
            else:
                depth = depth * batch["depth_scale"]
                return depth, nmap
        else:
            depth = depth * batch["depth_scale"]
            return depth
        # else:
        #     return depth, nmap


class CDFNet(nn.Module):

    def __init__(self, cfg):
        super(CDFNet, self).__init__()
        self.cfg = cfg
        self.frame_h = cfg.general.frame_h
        self.frame_w = cfg.general.frame_w
        self.cost_volume_layer_num = 4
        self.depth_norm = cfg.dataset.data_aug.depth_norm
        self.L = cfg.model.depth_plane_num
        self.add_aggregation_residual = False
        if self.cfg.model.color.type == "PyramidPool":
            model_cfg = self.cfg.model.color
            self.rgb_model = PyramidPool(model_cfg.out_channels)
            self.rgb_out_channel = model_cfg.out_channels
        else:
            raise NotImplementedError("Unknown color model type: %s" % self.cfg.model.color.type)

        self.add_pt_feature = False
        self.pt_out_channels = 0

        self.cost_volume_header = nn.Sequential(
            convbn_3d(self.rgb_out_channel * 2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.cost_volume_tail = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.residual_cost_volume = nn.ModuleList()
        for i in range(self.cost_volume_layer_num):
            self.residual_cost_volume.append(
                nn.Sequential(
                    convbn_3d(32, 32, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    convbn_3d(32, 32, 3, 1, 1),
                )
            )

        self.context_network_in_channel = self.rgb_out_channel + 1 + self.pt_out_channels
        self.context_network = nn.Sequential(
            convtext(self.context_network_in_channel, 96, 3, 1, 1),
            convtext(96, 96, 3, 1, 2),
            convtext(96, 96, 3, 1, 4),
            convtext(96, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.depth_regression = DisparityRegression(self.L)

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

    def forward(self, cur_color, forward_color, mask, raw_depth, min_depth, R_mat, t_vec, depth_scale,
                fx: float, fy: float, cx: float, cy: float):
        b, _, h, w = cur_color.size()
        cur_img_feat = self.rgb_model(cur_color)
        ref_img_feat = self.rgb_model(forward_color)

        b, c, hh, ww = ref_img_feat.size()
        device = cur_img_feat.device
        cur_mask = F.interpolate(mask, size=(hh, ww), mode="bilinear", align_corners=True)
        min_depth_resized = min_depth[:, :, :hh, :ww]
        raw_depth = F.interpolate(raw_depth, size=(hh, ww), mode="bilinear", align_corners=True)

        depth_gap = (1 - min_depth_resized) / self.L  # bx1
        depth = torch.ones(b, 1, hh, ww).to(device).float() * min_depth_resized
        cost = torch.zeros(b, c * 2, self.L, hh, ww).to(device).float()
        for l in range(self.L):
            depth = depth + depth_gap * l
            # predict locally
            depth[~(cur_mask > 0)] = raw_depth[~(cur_mask > 0)]
            cost[:, :c, l, :, :] = cur_img_feat
            cost[:, c:, l, :, :] = inverse_warp_cpp(depth, ref_img_feat, R_mat,t_vec, fx, fy, cx, cy)

        cost = cost.contiguous()
        cost = self.cost_volume_header(cost)
        for i, layer in enumerate(self.residual_cost_volume):
            cost = cost + layer(cost)
        cost = self.cost_volume_tail(cost)

        refined_cost = torch.zeros(b, 1, self.L, hh, ww).to(device).float()
        # cost aggregation
        for i in range(self.L):
            refined_cost[:, :, i, :, :] = self.context_network(torch.cat([cur_img_feat, cost[:, :, i, :, :]], 1)) \
                                          + cost[:, :, i, :, :]

        refined_cost_squeezed = torch.squeeze(refined_cost, 1)
        refined_cost_squeezed = F.softmax(
            F.interpolate(refined_cost_squeezed, [h, w], mode='bilinear', align_corners=True), dim=1
        )
        pred = self.depth_regression(refined_cost_squeezed)
        depth = min_depth + ((1 - min_depth) / self.L) * pred.unsqueeze(1)

        depth = depth * depth_scale
        return depth
