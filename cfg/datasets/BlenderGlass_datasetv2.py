import json
import os

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from cfg.datasets.BlenderGlass_dataset import BlenderGlass
from cfg.datasets.cleargrasp_dataset import mask_loader, safe_resize
from utils import rgbd2pcd
from utils.exr_handler import png_depth_loader
from utils.rgbd2pcd import get_surface_normal_from_xyz, random_xyz_sampling


class BlenderGlassV2(BlenderGlass):

    def __init__(self, root, scene="scene1", img_h=480, img_w=640, rgb_aug=False, max_norm=True, depth_factor=1000,
                 forward_windows_size=1):
        self.forward_windows_size = forward_windows_size
        super().__init__(root, scene, img_h, img_w, rgb_aug, max_norm, depth_factor)

    def __len__(self):
        # TODO: may not ignore?
        return len(self.img_list) - self.forward_windows_size  # ignore last data

    def __getitem__(self, index):
        assert index < (len(self.img_list) - self.forward_windows_size)
        cur_rgb, cur_mask, cur_render_depth, cur_raw_depth = self._get_item(index)

        # depth norm
        if self.max_norm:
            depth_ma = np.amax(cur_raw_depth)
            cur_raw_depth = cur_raw_depth / depth_ma
        else:
            depth_ma = 1.0
        depth_mi = np.amin(cur_raw_depth[~(cur_mask > 0)])

        cur_pose = self.pose_list[index]
        color = self.transform_seq(cur_rgb)
        xyz_img = rgbd2pcd.compute_xyz(cur_raw_depth, self.camera)
        xyz_gt = rgbd2pcd.compute_xyz(cur_render_depth, self.camera)
        xyz_img = torch.from_numpy(xyz_img).permute(2, 0, 1).float()
        xyz_gt = torch.from_numpy(xyz_gt).permute(2, 0, 1).float()  # 3xHxW

        sn_mask = np.where(cur_mask > 0, 255, 0).astype(np.uint8)
        sn_mask = cv2.erode(sn_mask, kernel=self.dilation_kernel)
        sn_mask[sn_mask != 0] = 1
        # sn_mask = np.logical_not(sn_mask)
        sn_mask = np.logical_and(sn_mask, cur_mask)

        sn_mask = torch.from_numpy(sn_mask).unsqueeze(0).float()
        cur_mask = torch.from_numpy(cur_mask).unsqueeze(0).float()

        cur_raw_depth = torch.from_numpy(cur_raw_depth).unsqueeze(0).float()
        cur_render_depth = torch.from_numpy(cur_render_depth).unsqueeze(0).float()
        cur_rgb = safe_resize(np.array(cur_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255
        cur_rgb = torch.from_numpy(cur_rgb).float()
        depth_gt_sn = get_surface_normal_from_xyz(xyz_gt.unsqueeze(0)).squeeze(0)
        pt = random_xyz_sampling(xyz_img, self.sampled_points_num)

        data_dict = {
            "cur_color": color,
            "cur_rgb": cur_rgb,
            "raw_depth": cur_raw_depth,
            "depth_scale": torch.tensor(depth_ma).repeat(self.img_h, self.img_w).float().unsqueeze(0),
            "min_depth": torch.tensor(depth_mi).repeat(self.img_h, self.img_w).float().unsqueeze(0),
            "mask": cur_mask,
            "sn_mask": sn_mask,
            "pt": pt,
            # "xyz_gt": xyz_gt,
            "cur_gt_depth": cur_render_depth,
            "depth_gt_sn": depth_gt_sn,
            "fx": torch.from_numpy(np.array(self.camera["fx"])).float(),
            "fy": torch.from_numpy(np.array(self.camera["fy"])).float(),
            "cx": torch.from_numpy(np.array(self.camera["cx"])).float(),
            "cy": torch.from_numpy(np.array(self.camera["cy"])).float(),
        }

        for i in range(1, self.forward_windows_size+1):
            forward_rgb, forward_mask, forward_depth, _ = self._get_item(index + i)
            forward_color = self.transform_seq(forward_rgb)

            forward_pose = self.pose_list[index + i]
            trans_mat = np.matmul(np.linalg.inv(forward_pose), cur_pose)

            forward_depth = torch.from_numpy(forward_depth).unsqueeze(0).float()
            forward_rgb = safe_resize(np.array(forward_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255
            forward_rgb = torch.from_numpy(forward_rgb).float()

            data_dict["forward_{}_color".format(i)] = forward_color
            data_dict["forward_{}_R_mat".format(i)] = torch.tensor(trans_mat[:3, :3]).float()
            data_dict["forward_{}_t_vec".format(i)] = torch.tensor(trans_mat[:3, 3]).float()
            data_dict["forward_{}_rgb".format(i)] = forward_rgb
            data_dict["forward_{}_depth".format(i)] = forward_depth

        return data_dict
