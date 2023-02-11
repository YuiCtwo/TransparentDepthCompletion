import json
import os

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from cfg.datasets.cleargrasp_dataset import mask_loader
from utils import rgbd2pcd
from utils.exr_handler import png_depth_loader
from utils.rgbd2pcd import get_surface_normal_from_xyz, random_xyz_sampling, get_xyz
import torch.nn.functional as F

def load_blender_data(base_dir):
    with open(os.path.join(base_dir, 'transforms.json'), 'r') as fp:
        metas = json.load(fp)
    imgs = []
    poses = []
    num = 0
    for frame in metas['frames']:
        img_path = os.path.join(base_dir, frame['file_path'])
        imgs.append(img_path)
        poses.append(np.array(frame['transform_matrix']).astype(np.float32))
        # poses = np.array(poses).astype(np.float32)
        num += 1

    print("Total number of img: {}".format(num))
    img0 = imageio.imread(imgs[0])
    H, W = img0.shape[:2]
    camera_angle_x = float(metas['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    return imgs, poses, [H, W, focal]


def safe_resize(img, img_w, img_h):
    res = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    res[np.isnan(res)] = 0.0
    res[np.isinf(res)] = 0.0
    return res


class BlenderGlass(Dataset):

    def __init__(self, root,
                 scene="scene1", img_h=480, img_w=640,
                 rgb_aug=False,
                 max_norm=True,
                 depth_factor=1000):
        self.scene = scene
        self.img_list = []
        self.camera_parameters = None
        self.mask_list = []
        self.depth_list = []
        self.pose_list = []
        self.img_name_list = []
        self.root_dir = root
        self.depth_factor = depth_factor
        self.img_h = img_h
        self.img_w = img_w
        self.sampled_points_num = (self.img_h // 4) * (self.img_w // 4)
        self.transform_seq = transforms.Compose(
            [
                transforms.Resize((self.img_h, self.img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._load_data(self.scene)
        # compute camera parameters
        f = self.camera_parameters[0]
        cx, cy = self.camera_parameters[1], self.camera_parameters[2]
        img_size = (cx*2, cy*2)  # wxh
        self.camera_origin = {
            "fx": float(f),
            "fy": float(f),
            "cx": float(cx),
            "cy": float(cy),
            "xres": int(img_size[0]),
            "yres": int(img_size[1])
        }

        scale = (self.img_w / img_size[0], self.img_h / img_size[1])
        fx = f * scale[0]
        fy = fx
        cx = cx * scale[0]
        cy = cy * scale[1]
        self.camera = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "xres": self.img_w,
            "yres": self.img_h
        }
        self.max_norm = max_norm
        self.dilation_kernel = np.ones((3, 3)).astype(np.uint8)

    def _load_data(self, scene_name):
        base_dir = os.path.join(self.root_dir, scene_name)
        depth_base_dir = os.path.join(base_dir, "depth")
        mask_base_dir = os.path.join(base_dir, "mask")
        rgb_dir = os.path.join(base_dir, "rgb")
        with open(os.path.join(base_dir, 'transforms.json'), 'r') as fp:
            metas = json.load(fp)
        frames = sorted(metas["frames"], key=lambda x: x["file_path"])
        for idx, frame in enumerate(frames):
            t = frame['file_path'].find("rgb") + 4
            img_name = frame['file_path'][t:]  # ignore './rgb/', '/rgb/
            img_path = os.path.join(rgb_dir, img_name)
            obj_pose = np.array(frame['transform_matrix']).astype(np.float32)
            mask_name = "Image" + str(idx).zfill(4) + ".png"
            mask_path = os.path.join(mask_base_dir, mask_name)
            depth_path = os.path.join(depth_base_dir, img_name)
            if not os.path.exists(depth_path):
                depth_path = os.path.join(depth_base_dir, mask_name)

            self.depth_list.append(depth_path)
            self.img_list.append(img_path)
            self.pose_list.append(obj_pose)
            self.mask_list.append(mask_path)
            self.img_name_list.append(img_name)

        img0 = imageio.imread(self.img_list[0])
        H, W = img0.shape[:2]
        camera_angle_x = float(metas['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        self.camera_parameters = (focal, W / 2, H / 2)

    def __len__(self):
        return len(self.img_list) - 1  # ignore last data

    def _get_item(self, index):
        rgb = Image.open(self.img_list[index]).convert("RGB")

        mask = mask_loader(self.mask_list[index])

        render_depth = png_depth_loader(self.depth_list[index])
        raw_depth = render_depth.copy()
        raw_depth[np.where(mask > 0)] = 0
        render_depth = safe_resize(render_depth, self.img_w, self.img_h) / self.depth_factor
        raw_depth = safe_resize(raw_depth, self.img_w, self.img_h) / self.depth_factor

        mask = safe_resize(mask, self.img_w, self.img_h)
        return rgb, mask, render_depth, raw_depth

    def __getitem__(self, index):
        assert index != (len(self.img_list) - 1)
        cur_rgb, cur_mask, cur_render_depth, cur_raw_depth = self._get_item(index)
        forward_rgb, forward_mask, forward_depth, _ = self._get_item(index+1)
        # depth norm
        if self.max_norm:
            depth_ma = np.amax(cur_raw_depth)
            cur_raw_depth = cur_raw_depth / depth_ma
        else:
            depth_ma = 1.0
        depth_mi = np.amin(cur_raw_depth[~(cur_mask > 0)])
        
        # compute transform matrix
        cur_pose = self.pose_list[index]
        forward_pose = self.pose_list[index+1]
        trans_mat = np.matmul(np.linalg.inv(forward_pose), cur_pose)
        
        color = self.transform_seq(cur_rgb)
        forward_color = self.transform_seq(forward_rgb)

        xyz_img = rgbd2pcd.compute_xyz(cur_raw_depth, self.camera)
        resized_xyz = safe_resize(xyz_img, img_w=self.img_w // 4, img_h=self.img_h // 4)

        resized_xyz = torch.from_numpy(resized_xyz).float()
        pt = resized_xyz.view(-1, 3).transpose(1, 0)
        xyz_gt = rgbd2pcd.compute_xyz(cur_render_depth, self.camera)
        # xyz_img = torch.from_numpy(xyz_img).permute(2, 0, 1).float()
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
        forward_depth = torch.from_numpy(forward_depth).unsqueeze(0).float()
        cur_rgb = safe_resize(np.array(cur_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255
        forward_rgb = safe_resize(np.array(forward_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255
        cur_rgb = torch.from_numpy(cur_rgb).float()
        forward_rgb = torch.from_numpy(forward_rgb).float()
        depth_gt_sn = get_surface_normal_from_xyz(xyz_gt.unsqueeze(0)).squeeze(0)

        # pt = random_xyz_sampling(xyz_img, self.sampled_points_num)
        return {
            "cur_color": color,
            "cur_rgb": cur_rgb,
            "raw_depth": cur_raw_depth,
            "depth_scale": torch.tensor(depth_ma).repeat(self.img_h, self.img_w).float().unsqueeze(0),
            # TODO: min_depth set to 0
            "min_depth": torch.tensor(depth_mi).repeat(self.img_h, self.img_w).float().unsqueeze(0),
            # "min_depth": torch.tensor(0.01).repeat(self.img_h, self.img_w).float().unsqueeze(0),
            "mask": cur_mask,
            "sn_mask": sn_mask,
            "pt": pt,
            # "xyz_gt": xyz_gt,
            "cur_gt_depth": cur_render_depth,
            "depth_gt_sn": depth_gt_sn,
            "R_mat": torch.tensor(trans_mat[:3, :3]).float(),
            "t_vec": torch.tensor(trans_mat[:3, 3]).float(),
            "forward_rgb": forward_rgb,
            "forward_color": forward_color,
            "forward_depth": forward_depth,
            "fx": torch.from_numpy(np.array(self.camera["fx"])).float(),
            "fy": torch.from_numpy(np.array(self.camera["fy"])).float(),
            "cx": torch.from_numpy(np.array(self.camera["cx"])).float(),
            "cy": torch.from_numpy(np.array(self.camera["cy"])).float(),
        }

