"""
revised from https://github.com/ZhengXinyue/png2klg
"""

import os
import numpy as np
import cv2
import zlib
from tqdm import tqdm

from cfg.datasets.cleargrasp_dataset import mask_loader
from utils import rgbd2pcd
from utils.exr_handler import png_depth_loader
from utils.icp import icp_o3d


def write_klg(base_dir, rgb_imgs, depth_imgs, mask_imgs, timestamps, camera, target_klg_path, drop_num=0, depth_scale=4000):
    n_frames = len(rgb_imgs)
    print('total frames: %d' % n_frames)
    klg = open(target_klg_path, 'wb')
    # <number of frames encoded as 32 bit int>
    klg.write(np.uint32(n_frames))
    curr_frame = 0
    for rgb, depth, mask, t in zip(rgb_imgs, depth_imgs, mask_imgs, timestamps):
        if curr_frame >= (n_frames - drop_num):
            continue
        # <timestamp encoded as 64 bit int>
        timestamp = np.uint64(t)

        cur_depth_path = base_dir + '/' + depth

        depth_image = cv2.imread(cur_depth_path, cv2.IMREAD_UNCHANGED)

        mask_path = base_dir + '/' + mask
        mask_image = cv2.imread(mask_path)
        mask_image = mask_image.astype(np.uint8)
        b_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

        depth_image[b_mask > 0] = 0

        mask_image = cv2.imencode('.jpeg', mask_image)[1].tostring()
        # mask.astype(np.uint32)
        mask_size = np.uint32(len(mask_image))
        # why need this `.byteswap()`?
        depth_image = depth_image.astype(np.uint16)
        # depth_image = depth_image.byteswap()
        # <depth buffer zlib compressed in the following depth_size number of bytes>
        # level = 9, best compression
        depth_image = zlib.compress(depth_image, 9)
        # <depth_size encoded as 32 bit int>
        depth_size = np.uint32(len(depth_image))

        cur_rgb_path = base_dir + '/' + rgb
        cur_rgb_image = cv2.imread(cur_rgb_path)
        cur_rgb_image = cur_rgb_image.astype(np.uint8)
        # <rgb buffer jpeg compressed in the following image_size number of bytes>
        cur_rgb_image = cv2.imencode('.jpeg', cur_rgb_image)[1].tostring()
        # <image_size (rgb) encoded as 32 bit int>
        cur_rgb_size = np.uint32(len(cur_rgb_image))
        # print(cur_rgb_size)
        next_rgb_path = base_dir + '/' + rgb_imgs[curr_frame+1]
        next_rgb_image = cv2.imread(next_rgb_path)
        next_rgb_image = next_rgb_image.astype(np.uint8)
        next_rgb_image = cv2.imencode('.jpeg', next_rgb_image)[1].tostring()
        next_img_size = np.uint32(len(next_rgb_image))

        cur_depth = png_depth_loader(cur_depth_path)
        next_depth_path = base_dir + '/' + depth_imgs[curr_frame+1]
        next_depth = png_depth_loader(next_depth_path)
        cur_pt = rgbd2pcd.compute_xyz(cur_depth, camera).reshape(-1, 3)
        next_pt = rgbd2pcd.compute_xyz(next_depth, camera).reshape(-1, 3)
        H = icp_o3d(cur_pt, next_pt, max_iterations=50, tolerance=0.01)
        # R_mat = np.ascontiguousarray(H[:3, :3], dtype=np.float32).tobytes()
        # t_vec = np.ascontiguousarray(H[:3, 3], dtype=np.float32).tobytes()
        H = np.ascontiguousarray(H, dtype=np.float32).tobytes()
        # H = zlib.compress(H, 9)
        # H = np.ascontiguousarray(H, dtype=np.float32).tobytes()
        H_size = np.uint32(len(H))
        # print(H_size)
        # t_size = np.uint32(len(t_vec))
        # pt =

        klg.write(timestamp)
        klg.write(depth_size)
        klg.write(cur_rgb_size)
        klg.write(next_img_size)
        klg.write(mask_size)
        klg.write(H_size)
        # klg.write(t_size)
        klg.write(depth_image)
        klg.write(cur_rgb_image)
        klg.write(next_rgb_image)
        klg.write(mask_image)
        # klg.write(R_mat)
        klg.write(H)

        # klg.write(bytes(camera["fx"]))
        # klg.write(bytes(camera["fy"]))
        # klg.write(bytes(camera["cx"]))
        # klg.write(bytes(camera["cy"]))

        curr_frame += 1
        # break

    klg.close()
