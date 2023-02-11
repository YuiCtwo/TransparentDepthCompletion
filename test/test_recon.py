import random
from time import time

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model.df_net import DFNet
from utils.icp import icp, icp_o3d
from utils.rgbd2pcd import random_xyz_sampling, get_xyz, compute_xyz, uniform_xyz_sampling
from utils.visualize import vis_depth, vis_surface_normal, vis_surface_normal_3d, vis_pt_3d, make_dir
from skimage.transform import resize
from torchvision import transforms

from cfg.datasets.BlenderGlass_dataset import png_depth_loader, safe_resize
from cfg.datasets.cleargrasp_dataset import mask_loader
from cfg.datasets.RealTime_dataset import RealTimeCaptureDataset
from model.vdcnet import SwinVDC
from utils import config_loader, rgbd2pcd
from utils import exr_handler

import os

#
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# random.seed(0)

ds_path = "/home/ctwo/recon_data"
out_dir = "depth_df"
make_dir("/home/ctwo/recon_data", out_dir)
vis_path = "/home/ctwo/recon_data/vis"
vis_surface_normal_path = "/home/ctwo/recon_data/surface_normal"
out_img_h = 480
out_img_w = 640
depth_factor = 500
# cfg_file_path = "../experiments/swin_pt_spade_norm_refine3.yml"
cfg_file_path = "../experiments/dfnet_L50.yml"
cfg = config_loader.CFG.read_from_yaml(cfg_file_path)
model = DFNet(cfg)
snapshot = torch.load(cfg.general.pretrained_weight)
model.load_state_dict(snapshot.pop("model"), strict=False)
device = torch.device("cuda:0")
model = model.to(device)

dataset = RealTimeCaptureDataset(ds_path, img_w=320, img_h=240, depth_factor=depth_factor)
dataset_lens = len(dataset)
print("Total frame: {}".format(dataset_lens))

input_dataset = []

for idx in range(dataset_lens):
    data = dataset[idx]
    png_name = dataset.img_name_list[idx]
    depth_name = dataset.depth_list[idx]
    mask_name = dataset.mask_list[idx]
    original_mask = mask_loader(mask_name)
    original_depth = png_depth_loader(depth_name)

    H = np.eye(4)
    # solve icp
    if idx != dataset_lens - 1:
        cur_pt = data["cur_xyz"].reshape(-1, 3)
        forward_pt = data["forward_xyz"].reshape(-1, 3)  # nx3
        # compute icp
        H = icp_o3d(cur_pt, forward_pt, max_iterations=50, tolerance=0.01)
        # H = icp(cur_pt, forward_pt)

    model_input = {
        "cur_color": data["cur_color"].to(device).unsqueeze(0),
        # "cur_rgb": data["cur_rgb"].to(device).unsqueeze(0),
        "raw_depth": data["raw_depth"].to(device).unsqueeze(0),
        "depth_scale": data["depth_scale"].to(device).unsqueeze(0),
        "min_depth": data["min_depth"].to(device).unsqueeze(0),
        "mask": data["mask"].to(device).unsqueeze(0),
        "pt": data["pt"].to(device).unsqueeze(0),
        # "forward_rgb": data["forward_rgb"].to(device).unsqueeze(0),
        "forward_color": data["forward_color"].to(device).unsqueeze(0),
        "fx": data["fx"].to(device).unsqueeze(0),
        "fy": data["fy"].to(device).unsqueeze(0),
        "cx": data["cx"].to(device).unsqueeze(0),
        "cy": data["cy"].to(device).unsqueeze(0),
        "R_mat": torch.tensor(H[:3, :3]).float().to(device).unsqueeze(0),
        "t_vec": torch.tensor(H[:3, 3]).float().to(device).unsqueeze(0),
    }

    model.eval()
    with torch.no_grad():
        output_depth = model(model_input)
        # output_depth[~(model_input["mask"] > 0)] = model_input["raw_depth"][~(model_input["mask"] > 0)]
        # xyz = get_xyz(output_depth, model_input["fx"], model_input["fy"], model_input["cx"], model_input["cy"])
        # xyz = xyz.squeeze(0)
        # pt = random_xyz_sampling(xyz, 4800)
        # model_input["pt"] = pt
        # output_depth = model(model_input)
        output_depth = output_depth.squeeze().cpu().numpy()

    output_depth = safe_resize(output_depth, out_img_w, out_img_h) * depth_factor
    output_depth[~(original_mask > 0)] = original_depth[~(original_mask > 0)]

    png_filename = "{}.png".format(png_name)
    vis_save_path = os.path.join(vis_path, png_filename)
    vis_sn_save_path = os.path.join(vis_surface_normal_path, png_filename)
    vis_depth(output_depth / np.amax(output_depth), 0, 1, visualize=False,
              save_path=vis_save_path, color_map='Greys')
    vis_surface_normal(output_depth, dataset.camera_origin, visualize=False,
                       save_path=vis_sn_save_path, color_map="RdBu_r")
    # _original_depth = original_depth / np.amax(original_depth)
    # xyz = compute_xyz(_original_depth, dataset.camera_origin)
    # xyz = xyz[original_mask > 0]
    # pt = xyz.transpose(1, 0)
    # vis_pt_3d(pt, v_size=0.01)
    # vis_surface_normal_3d(pt, v_size=0.02)
    out_file_path = os.path.join(os.path.join(ds_path, out_dir), png_filename)
    exr_handler.png_writer(output_depth, out_file_path)
    print("{} predict complete".format(png_name))
