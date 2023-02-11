import json
import os
from PIL import Image
import imageio
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from cfg.datasets.BlenderGlass_dataset import load_blender_data
from utils.rgbd2pcd import get_xyz
from utils.exr_handler import png_depth_loader


def photometric_loss(current_depth, current_rgb, forward_rgb, mask, camera, R, t):
    """
    将当前的深度图投影到后一帧
    :param current_depth: 当前帧深度图
    :param current_rgb: 当前帧 RGB图
    :param forward_rgb: 下一帧的 RGB图
    :param mask: 重建物体在图像中的位置
    :param camera: 相机内参
    :param R: 旋转矩阵
    :param t: 平移向量
    """
    batch_size = current_depth.size(0)
    h = current_depth.size(2)
    w = current_depth.size(3)
    cu = camera["cx"]
    cv = camera["cy"]
    fu = camera["fx"]
    fv = camera["fy"]
    pt_current = get_xyz(current_depth, fu, fv, cu, cv).float()
    # pt_current = pt_current.permute(0, 2, 3, 1)  # [bs, 3, h, w] -> [bs, h, w, 3]
    XYZ_ = torch.bmm(R, pt_current.view(batch_size, 3, -1))
    X = (XYZ_[:, 0, :] + t[:, 0].unsqueeze(1)).view(-1, 1, h, w)
    Y = (XYZ_[:, 1, :] + t[:, 1].unsqueeze(1)).view(-1, 1, h, w)
    Z = (XYZ_[:, 2, :] + t[:, 2].unsqueeze(1)).view(-1, 1, h, w)

    # compute pixel coordinates
    U_proj = fu[0] * X / Z + cu[0]  # horizontal pixel coordinate
    V_proj = fv[0] * Y / Z + cv[0]  # vertical pixel coordinate

    # normalization to [-1, 1], required by torch.nn.functional.grid_sample
    U_proj_normalized = (2 * U_proj / (w - 1) - 1).view(batch_size, -1)
    V_proj_normalized = (2 * V_proj / (h - 1) - 1).view(batch_size, -1)

    U_proj_mask = ((U_proj_normalized > 1) + (U_proj_normalized < -1)).detach()
    U_proj_normalized[U_proj_mask] = 2
    V_proj_mask = ((V_proj_normalized > 1) + (V_proj_normalized < -1)).detach()
    V_proj_normalized[V_proj_mask] = 2

    pixel_coords = torch.stack([U_proj_normalized, V_proj_normalized], dim=2).view(batch_size, h, w, 2)  # [B, H, W, 2]
    recon_rgb = F.grid_sample(forward_rgb, pixel_coords)
    diff = (current_rgb - recon_rgb).abs()
    diff = torch.sum(diff, 1)  # sum along the color channel

    # compare only pixels that are not black
    valid_mask = (torch.sum(current_rgb, 1) > 0).float() * (torch.sum(recon_rgb, 1) > 0).float()
    if mask is not None:
        valid_mask = valid_mask * torch.squeeze(mask).float()
    valid_mask = valid_mask.byte().detach()
    if valid_mask.numel() > 0:
        diff = diff[valid_mask > 0]
        if diff.nelement() > 0:
            return diff.mean()
        else:
            # this is expected during early stage of training, try larger batch size.
            print("warning: diff.nelement() == 0 in PhotometricLoss")
    else:
        print("warning: 0 valid pixel in PhotometricLoss")
    return 1.0


if __name__ == '__main__':
    base_dir = "/home/ctwo/glass/scene1"
    depth_base_dir = base_dir + "/depth"
    imgs, poses, [H, W, focal] = load_blender_data(os.path.join(base_dir, "rgb"))
    pos_1 = poses[1]
    pos_2 = poses[2]
    camera_intrinsics = {
        "fx": torch.FloatTensor(np.array([focal])),
        "fy": torch.FloatTensor(np.array([focal])),
        "cx": torch.FloatTensor(np.array([H/2])),
        "cy": torch.FloatTensor(np.array([W/2]))
    }

    trans_mat = np.matmul(np.linalg.inv(pos_2), pos_1)
    print(trans_mat)
    r_mat = torch.tensor(trans_mat[:3, :3], dtype=torch.float32).unsqueeze(0)
    t_vec = torch.tensor(trans_mat[:3, 3], dtype=torch.float32).unsqueeze(0)
    print(r_mat)
    print(t_vec)
    trans_mat = torch.tensor(trans_mat, dtype=torch.float32).unsqueeze(0)

    depth_file_path = depth_base_dir + "/Image0001.png"
    depth_file_path_forward = depth_base_dir + "/Image0003.png"
    depth_cur = png_depth_loader(depth_file_path)
    depth_forward = png_depth_loader(depth_file_path_forward)
    depth_cur = torch.from_numpy(depth_cur).unsqueeze(0).unsqueeze(0).float()
    depth_forward = torch.from_numpy(depth_forward).unsqueeze(0).unsqueeze(0).float()

    rgb_1 = np.array(Image.open(imgs[1]).convert("RGB")).transpose(2, 0, 1)
    rgb_1 = torch.tensor(rgb_1, dtype=torch.float32).unsqueeze(0)
    rgb_2 = np.array(Image.open(imgs[3]).convert("RGB")).transpose(2, 0, 1)
    rgb_2 = torch.tensor(rgb_2, dtype=torch.float32).unsqueeze(0)
    print(rgb_1.size())
    out = photometric_loss(depth_cur, rgb_2, camera_intrinsics, r_mat, t_vec)
    out = out.squeeze().cpu().numpy()
    print(out)
    # p_loss = PhotometricLoss()
    # print(p_loss(rgb_2, r_mat))
    # plt.imshow(out)
    # plt.show()
