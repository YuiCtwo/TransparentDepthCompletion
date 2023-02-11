import sys
from time import time

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("/home/ctwo/VDCNet")
from model.df_net import DFNet
from utils.rgbd2pcd import random_xyz_sampling, get_xyz, compute_xyz
from utils.visualize import vis_depth, vis_surface_normal, vis_surface_normal_3d, vis_pt_3d, make_dir, vis_normal
from skimage.transform import resize
from torchvision import transforms

from cfg.datasets.BlenderGlass_dataset import png_depth_loader, safe_resize, BlenderGlass
from cfg.datasets.cleargrasp_dataset import mask_loader
from cfg.datasets.RealTime_dataset import RealTimeCaptureDataset
from utils import config_loader, rgbd2pcd
from utils import exr_handler

import os

test_scene = "val_seq4"
output_filename = "depth_sn"
cfg_file_path = "../experiments/dfnet_pt_sn.yml"

parser = argparse.ArgumentParser(description=None)
parser.add_argument("--cfg", type=str, default=cfg_file_path)
parser.add_argument("--test_scene", type=str, default=test_scene)
parser.add_argument("--output_filename", type=str, default=output_filename)
args = parser.parse_args()

test_scene = args.test_scene
output_filename = args.output_filename
cfg_file_path = args.cfg

ds_root = "/home/ctwo/blender_glass_data"
ds_path = os.path.join(ds_root, test_scene)
make_dir(ds_path, "vis_depth")
make_dir(ds_path, "vis_sn")
make_dir(ds_path, output_filename)
out_path = os.path.join(ds_path, output_filename)
vis_path = os.path.join(ds_path, "vis_depth")
vis_surface_normal_path = os.path.join(ds_path, "vis_sn")

outputImgHeight = 480
outputImgWidth = 640
cfg = config_loader.CFG.read_from_yaml(cfg_file_path)
model = DFNet(cfg)
snapshot = torch.load(cfg.general.pretrained_weight)
model.load_state_dict(snapshot.pop("model"), strict=False)
device = torch.device("cuda:0")
model = model.to(device)
depth_factor = 4000
dataset = BlenderGlass(root=ds_root,
                       scene=test_scene,
                       img_h=cfg.general.frame_h,
                       img_w=cfg.general.frame_w,
                       rgb_aug=False,
                       max_norm=True,
                       depth_factor=depth_factor
                       )
dataloader = DataLoader(
    dataset,
    shuffle=False,
    num_workers=8,
    batch_size=1,
)
print("Total frame: {}".format(len(dataset)))
start_time = time()
for idx, batch in enumerate(dataloader):
    # if idx < 30:
    #     continue
    png_name = dataset.img_name_list[idx]
    depth_name = dataset.depth_list[idx]
    mask_name = dataset.mask_list[idx]
    original_mask = mask_loader(mask_name)
    original_depth = png_depth_loader(depth_name)
    for k in batch:
        batch[k] = batch[k].to(device)
    model.eval()
    eval_time_start = time()
    with torch.no_grad():
        output_depth = model(batch)
        output_depth[~(batch["mask"] > 0)] = batch["raw_depth"][~(batch["mask"] > 0)]
        xyz = get_xyz(output_depth, batch["fx"], batch["fy"], batch["cx"], batch["cy"])
        xyz = xyz.squeeze(0)
        pt = random_xyz_sampling(xyz, 4800)
        batch["pt"] = pt
        output_depth = model(batch)
        output_depth = output_depth.squeeze().cpu().numpy()
    # print(time()-eval_time_start)
    output_depth = safe_resize(output_depth, outputImgWidth, outputImgHeight) * depth_factor
    # vis_surface_normal(original_depth, dataset.camera_origin, visualize=True,
    #                    save_path=None, color_map="RdBu_r")
    output_depth[~(original_mask > 0)] = original_depth[~(original_mask > 0)]
    # print("eval time used: {}".format(time()-eval_time_start))
    # png_filename = "{}.png".format(png_name)
    vis_save_path = os.path.join(vis_path, png_name)
    vis_sn_save_path = os.path.join(vis_surface_normal_path, png_name)
    vis_depth(output_depth / np.amax(output_depth), 0, 1, visualize=False,
              save_path=vis_save_path, color_map='Greys')
    vis_surface_normal(output_depth, dataset.camera_origin, visualize=False,
                       save_path=vis_sn_save_path, color_map="RdBu_r")
    # vis_normal(nmap)
    # _original_depth = original_depth / np.amax(original_depth)
    # xyz = compute_xyz(_original_depth, dataset.camera_origin)
    # xyz = xyz[original_mask > 0]
    # pt = xyz.transpose(1, 0)
    # vis_pt_3d(pt, v_size=0.01)
    # vis_surface_normal_3d(pt, v_size=0.01)
    out_file_path = os.path.join(out_path, png_name)
    # exr_handler.png_writer(output_depth, out_file_path)
    exr_handler.png_writer(output_depth, out_file_path)
    print("{} predict complete".format(png_name))
    # break

print("test end, time used: {:.2f}s".format(time() - start_time))
