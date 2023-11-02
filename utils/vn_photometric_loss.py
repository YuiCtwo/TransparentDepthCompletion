import os

import cv2
import numpy as np
import torch
from PIL import Image
import scipy
import torch.nn.functional as F
import sophus as sp

from cfg.datasets.BlenderGlass_dataset import load_blender_data
from utils.exr_handler import png_depth_loader
from utils.keyframe_utils import KeyFrame
from utils.rgbd2pcd import get_xyz, compute_xyz
from utils.visualize import vis_mask


def huber_norm(x, huber_delta: float):
    return np.where(x <= huber_delta, x * x / (2 * huber_delta), x - huber_delta / 2)


def get_interpolated_grad(grad, x, y):
    x0 = np.floor(x).long()
    x1 = x0 + 1
    y0 = np.floor(y).long()
    y1 = y0 + 1

    # 将 x0,x1,y0,y1 约束到合法的范围内
    x0 = np.clip(x0, 0, grad.shape[1] - 1)
    x1 = np.clip(x1, 0, grad.shape[1] - 1)
    y0 = np.clip(y0, 0, grad.shape[0] - 1)
    y1 = np.clip(y1, 0, grad.shape[0] - 1)
    Ia = grad[y0, x0]
    Ib = grad[y1, x0]
    Ic = grad[y0, x1]
    Id = grad[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


class VARNormalizedPhotometricLoss:

    def __init__(self, refer_frame: KeyFrame, current_grey, camera, huber_delta=3):
        """
        根据当前的图片(current) 和参考图/关键帧(refer) 优化关于运动 R,t 的函数
        将参考帧变化到当前帧
        :param current_grey: (1 x h x w)

        :param camera: dict{cx, cy, fx, fy, yres, xres}
        :param huber_delta: float
        """
        self.huber_delta = huber_delta
        self.refer_depth = refer_frame.D
        self.d = 1 / (self.refer_depth.squeeze() + 1e-9)
        self.cov_d = refer_frame.inverse_depth_variance

        self.refer_grey = refer_frame.grey
        self.cx = camera["cx"]
        self.cy = camera["cy"]
        self.fx = camera["fx"]
        self.fy = camera["fy"]
        self.h = camera["yres"]
        self.w = camera["xres"]
        self.current_grey = current_grey.squeeze()
        self.grad_x, self.grad_y = np.gradient(self.current_grey)
        self.gaussian_image_intensity_noise = 0.1
        self.high_grad_mask = refer_frame.high_grad_mask > 0
        self.pt_current = compute_xyz(self.refer_depth, camera).squeeze().transpose(2, 0, 1).reshape(3, -1)  # 3xn
        self.cameraPixelNoise2 = 4
        self.var_weight = 1.0

    def loss_fn(self, R, t):
        """
        :param R: 3x3
        :param t: 3x1
        :return: (R, t)
        """
        XYZ = np.matmul(R, self.pt_current)
        X = (XYZ[0, :] + t[0]).reshape(self.h, self.w)
        Y = (XYZ[1, :] + t[1]).reshape(self.h, self.w)
        Z = (XYZ[2, :] + t[2]).reshape(self.h, self.w)

        U_proj = self.fx * X / Z + self.cx
        V_proj = self.fy * Y / Z + self.cy
        # opencv do not support np.float64
        U_proj = U_proj.astype(np.float32)
        V_proj = V_proj.astype(np.float32)
        g0 = (X * Z - t[2] * X) / (Z * Z * self.d)
        g1 = (Y * Z - t[2] * Y) / (Z * Z * self.d)
        gx = cv2.remap(self.grad_x, U_proj, V_proj, cv2.INTER_CUBIC)
        gy = cv2.remap(self.grad_y, U_proj, V_proj, cv2.INTER_CUBIC)
        gx = gx * self.fx
        gy = gy * self.fy

        coord_np = np.where(self.high_grad_mask > 0)
        coord = list(zip(coord_np[0], coord_np[1]))
        J = self.compute_J(X, Y, Z, gx, gy, coord) + 1e-9
        recon = cv2.remap(self.current_grey, U_proj, V_proj, cv2.INTER_CUBIC)
        r = np.abs(self.refer_grey[self.high_grad_mask] - recon[self.high_grad_mask])
        r2 = r * r
        # 计算 weight
        # loss = loss * loss / (norm_weight + 1e-5)
        dr_dD = (gx * g0 + gy * g1)[self.high_grad_mask > 0]
        wp = 1 / (self.gaussian_image_intensity_noise + dr_dD * dr_dD * self.cov_d * self.var_weight)
        weighted_rp = np.abs(r * np.sqrt(wp)) + 1e-9
        # wh = np.abs(np.where(weighted_rp < (self.huber_delta / 2), 1, self.huber_delta / weighted_rp))
        wh = np.abs(np.where(weighted_rp <= self.huber_delta, self.huber_delta / weighted_rp, 1))
        wh_ = wh.reshape(-1, 1).repeat(6, axis=1)
        J = J * wh_
        # loss = r2 * wh * wp
        loss = huber_norm(r2 * wp, self.huber_delta)
        # norm_weight = np.abs(np.where(J <= self.huber_delta, 1 / self.huber_delta, 1 / J))
        # loss = huber_norm(r2, self.huber_delta)
        # w_J = np.abs(np.where(r <= self.huber_delta, r / self.huber_delta, 1))
        # w_J = w_J.reshape(-1, 1).repeat(6, axis=1)
        # J = w_J * J
        # J = np.sum(J*w_J, axis=0, keepdims=True)
        # loss = huber_norm(r2, self.huber_delta)
        loss = np.mean(loss, keepdims=True)
        J = np.mean(J, axis=0, keepdims=True)
        return loss, (2 * J, loss)
        # return np.mean(r2), (2 * J, r2)

    def compute_J(self, px, py, pz, dx, dy, coord):
        J = []
        for (y_idx, x_idx) in coord:
            x = px[y_idx, x_idx]
            y = py[y_idx, x_idx]
            z = pz[y_idx, x_idx]
            gx = dx[y_idx, x_idx]
            gy = dy[y_idx, x_idx]
            _z = 1 / z
            _z2 = 1 / (z * z)
            res = np.array([0] * 6, dtype=np.float32)
            res[0] = _z * gx
            res[1] = _z * gy
            res[2] = (-x * _z2 * gx) + (-y * _z2 * gy)
            res[3] = (-x * y * _z2) * gx + (1.0 + y * y * _z2) * gy
            res[4] = (1.0 + x * x * _z2) * gx + (x * y * _z2) * gy
            res[5] = (-y * _z) * gx + (x * _z) * gy
            # !important remember to add `-`
            J.append(-res)
        J = np.array(J)
        return J


def tracker(refer_frame: KeyFrame, current_grey, camera, max_iteration=50):
    H = sp.SE3(refer_frame.R, refer_frame.t)
    huber_delta = 3
    obj_function = VARNormalizedPhotometricLoss(refer_frame, current_grey, camera, huber_delta)
    loss_old = 1e8
    for it in range(max_iteration):
        loss, (J, weighted_r) = obj_function.loss_fn(H.rotationMatrix(), H.translation())
        # compute update value delta_H at `lie` format
        A = np.matmul(J.T, J)
        b = -np.matmul(J.T, weighted_r)
        delta_H = np.linalg.lstsq(A, b, rcond=None)[0]
        print("iter: {}, loss={}".format(it, loss))
        # if np.abs(loss_old - loss) < 1e-3:
        #     break
        # else:
        #     loss_old = loss
        # if loss_old < loss:
        #     break
        # else:
        #     loss_old = loss
        H = sp.SE3.exp(delta_H) * H

    return H.rotationMatrix(), H.translation()


def cal_distance(p1, p2):
    return np.sum((p2 - p1) ** 2)


if __name__ == '__main__':
    # scale_factor = 1000
    base_dir = "/home/ctwo/glass/scene1"
    depth_base_dir = base_dir + "/depth"
    imgs, poses, [H, W, focal] = load_blender_data(os.path.join(base_dir, "rgb"))
    pos_1 = poses[1]
    pos_2 = poses[5]

    trans_mat = np.matmul(np.linalg.inv(pos_2), pos_1)
    r_mat = trans_mat[:3, :3]
    t_vec = trans_mat[:3, 3]
    print(trans_mat)
    print(r_mat)
    print(t_vec)

    depth_file_path_1 = depth_base_dir + "/Image0001.png"
    depth_file_path_2 = depth_base_dir + "/Image0005.png"
    depth_1 = png_depth_loader(depth_file_path_1)
    depth_2 = png_depth_loader(depth_file_path_2)
    depth_1 = np.expand_dims(depth_1, axis=0)
    depth_2 = np.expand_dims(depth_2, axis=0)

    rgb_1 = np.array(Image.open(imgs[1]).convert("RGB"))
    grey_1 = np.array(Image.open(imgs[1]).convert("L"))
    grey_1 = np.expand_dims(grey_1, axis=0)
    rgb_2 = np.array(Image.open(imgs[5]).convert("RGB"))
    grey_2 = np.array(Image.open(imgs[5]).convert("L"))
    grey_2 = np.expand_dims(grey_2, axis=0)
    kf = KeyFrame(None, depth_1, rgb_1, grey_1, None)
    # vis_mask(kf.high_grad_mask)

    camera_intrinsics = {
        "fx": np.array([focal]),
        "fy": np.array([focal]),
        "cx": np.array([H / 2]),
        "cy": np.array([W / 2]),
        "yres": depth_1.shape[1],
        "xres": depth_1.shape[2]
    }
    # kf.t = t_vec
    # kf.R = r_mat
    # print(kf.inverse_depth_variance)
    r, t = tracker(kf, grey_2, camera_intrinsics)
    print(r)
    print(t)
    test_point = np.array([1, 1, 1])  # 1x3
    p1 = np.matmul(test_point, r_mat) + t_vec
    p2 = np.matmul(test_point, r) + t
    print(cal_distance(p1, p2))

# [[ 9.9999911e-01 -6.2315352e-04  1.2142658e-03]
#  [ 6.2342733e-04  9.9999988e-01 -2.2599101e-04]
#  [-1.2141280e-03  2.2670627e-04  9.9999928e-01]]
