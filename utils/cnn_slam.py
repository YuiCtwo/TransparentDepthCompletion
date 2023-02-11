import json

import cv2
import imageio
import numpy as np
import os

import torch
from PIL import Image
from skimage.transform import resize
from cfg.datasets.BlenderGlass_dataset import load_blender_data, safe_resize
from cfg.datasets.cleargrasp_dataset import mask_loader
from cfg.datasets.RealTime_dataset import RealTimeCaptureDataset
from model.vdcnet import SwinVDC
from utils import rgbd2pcd
from utils.config_loader import CFG
from utils.exr_handler import png_depth_loader
from utils.keyframe_utils import KeyFrame, depth_fusion, uncertainty_propagate
from utils.rgbd2pcd import random_xyz_sampling
from utils.vn_photometric_loss import tracker
from utils.visualize import vis_mask


def cnn_slam_dataset(base_dir, ds_name="real_time"):
    if ds_name == "real_time":
        return RealTimeCaptureDataset(base_dir)

def cnn_predict_depth(model, data: dict, device, out_w=640, out_h=480):
    mask = data["mask"]
    original_mask = data["original_mask"]
    raw_depth = data["raw_depth"]
    original_depth = data["original_depth"]
    inp = {
        "color": data["color"].to(device).unsqueeze(0),
        "mask": data["input_mask"].to(device).unsqueeze(0),
        "depth_scale": data["depth_scale"].to(device).unsqueeze(0),
        "pt": data["pt"].to(device).unsqueeze(0),
    }
    model.eval()
    with torch.no_grad():
        output_depth = model(inp)
        output_depth = output_depth.squeeze().cpu().numpy()
    raw_depth[mask > 0] = output_depth[mask > 0]
    raw_depth = resize(raw_depth, (out_h, out_w))
    original_depth[original_mask > 0] = raw_depth[original_mask > 0]

    original_depth = np.expand_dims(original_depth, axis=0)
    return original_depth  # (1 x h x w)


def cnn_dataloader(dataset, idx):

    data, png_name = dataset[idx]
    mask = data["original_mask"]
    png_name = png_name + ".png"
    png_path = os.path.join(dataset.base_dir, "rgb")
    png_path = os.path.join(png_path, png_name)
    rgb = np.array(Image.open(png_path).convert("RGB"))
    grey = np.expand_dims(np.array(Image.open(png_path).convert("L")), axis=0)
    
    return data, rgb, grey, mask, png_name

def wait_key():
    """
    Used just for debug
    """
    input(">Press Enter to continue...")


def cnn_slam_main():
    base_dir = "/home/ctwo/recon_data"
    sigma_p = 0.1
    # Step 0: initialize first KeyFrame list and model
    cfg_file_path = "../experiments/swin_norm.yml"
    cfg = CFG.read_from_yaml(cfg_file_path)
    print("Load configuration from {}".format(cfg_file_path))
    if cfg.general.model == "SwinVDC":
        model = SwinVDC(cfg)
    else:
        print("Unsupported model")
        return
    device = torch.device("cuda:0")
    model = model.to(device)
    

    dataset = cnn_slam_dataset(base_dir=base_dir)
    start = 0
    end = len(dataset)
    camera = dataset.camera_origin

    data, rgb, grey, mask, png_name = cnn_dataloader(dataset, start)
    
    predict_depth = cnn_predict_depth(model, data=data, device=device)
    # print(predict_depth.shape)
    keyframe_list = [KeyFrame(None, predict_depth, rgb, grey, mask)]

    refer_frame = keyframe_list[0]
    refer_frame.U = np.ones_like(predict_depth) * sigma_p

    start += 1
    for i in range(start, end):
        # vis_mask(refer_frame.high_grad_mask)
        # current img i
        # if KeyFrame.check_keyframe():

        # Step 1: predict depth from rgb and incomplete depth
        data, rgb, grey, mask, png_name = cnn_dataloader(dataset, i)

        predict_depth = cnn_predict_depth(model, data=data, device=device)

        # Step 2: track the motion, find R, t
        R, t = tracker(refer_frame, grey, camera)
        H = np.array([
            [R[0, 0], R[0, 1], R[0, 2], t[0]],
            [R[1, 0], R[1, 1], R[1, 2], t[1]],
            [R[2, 0], R[2, 1], R[2, 2], t[2]],
            [0, 0, 0, 1],
        ])
        print(R)
        print(t)
        # Step 3: check keyframe
        current_frame = KeyFrame(H, predict_depth, rgb, grey, mask)
        # Step 4: compute uncertainty map
        uncertainty_map, renewed_depth = uncertainty_propagate(current_frame, refer_frame, camera)
        current_frame.U = uncertainty_map
        current_frame.D = renewed_depth
        # Step5: depth map and uncertainty map fuse
        # depth_fusion(current_frame, refer_frame, camera)
        keyframe_list.append(current_frame)
        # Step 6: global pose optimization
        print()
        # Step 7: gen point cloud
        wait_key()
        refer_frame = current_frame


if __name__ == '__main__':
    cnn_slam_main()
