import argparse
import random

import torch
import os

from model.df_net import DFNet, CDFNet
from train.df_trainer import DFTrainer
from train.trainer import Trainer
from model.vdcnet import SwinVDC
from model.DM_LRN.dm_lrn import DM_LRN
from utils.config_loader import CFG

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# random.seed(0)


def get_param_num(m):
    total = sum(p.numel() for p in m.parameters())
    print("{:.2f}".format(total / (1024 * 1024)))


def build_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("cfg_file", type=str, help="配置文件位置")
    args = parser.parse_args()


def main():

    cfg_file_path = "experiments/df_L50.yml"
    # cfg_file_path = "experiments/dfnet_pt_sn.yml"
    # cfg_file_path = "experiments/dfnet_pt.yml"
    cfg = CFG.read_from_yaml(cfg_file_path)
    print("Load configuration from {}".format(cfg_file_path))
    if cfg.general.model == "DFNet":
        model = CDFNet(cfg)
        # model = DFNet(cfg)
        model_trainer = DFTrainer(cfg, model)
    elif cfg.general.model == "SwinVDC":
        model = SwinVDC(cfg)
        model_trainer = Trainer(cfg, model)
    elif cfg.general.model == "DM_LRN":
        model = DM_LRN(extract_feature=False, predict_depth=True, depth_max_min=cfg.dataset.data_aug.depth_max_min)
        model_trainer = Trainer(cfg, model)
    else:
        raise Exception("Unknown model")

    if cfg.general.running_type == "train":
        model_trainer.start_training()
    elif cfg.general.running_type == "val":
        print(model_trainer.start_validating())
    # get_param_num(model)


if __name__ == '__main__':
    main()
