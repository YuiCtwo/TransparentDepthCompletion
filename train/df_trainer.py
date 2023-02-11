import os
import numpy as np
import torch
import re

from time import time

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from cfg.datasets.BlenderGlass_dataset import BlenderGlass
from cfg.datasets.BlenderGlass_datasetv2 import BlenderGlassV2
from cfg.datasets.cleargrasp_dataset import ClearGrasp
from train.sobel_grad_loss import sobel_normal_loss, Sobel
from utils.meter import AggregationMeter, Statistics
from train.metrics import Metrics, SSIMLoss, depth_grad_loss_unsupervised, depth_loss, surface_normal_cos_loss, \
    ConsLoss, depth_grad_loss, surface_normal_l1_loss, surface_normal_loss
from train.consistency_loss import photometric_geometry_loss, photometric_geometry_loss_v2
from utils.rgbd2pcd import get_surface_normal_from_depth, get_xyz, get_surface_normal_from_xyz


def dict2device(data, device):
    for k in data:
        data[k] = data[k].to(device)


class DFTrainer:

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.print_freq = cfg.general.print_freq
        self.camera = None
        self.depth_factor = 4000
        self.running_type = cfg.general.running_type
        if self.cfg.general.pretrained_weight:
            self.pretrained_model_path = self.cfg.general.pretrained_weight
            print("load weight from:{}".format(self.pretrained_model_path))
            snapshot = torch.load(self.pretrained_model_path)
            self.model.load_state_dict(snapshot.pop("model"), strict=False)
        else:
            print("do not use pretrained weight")
            self.model.init_weights()

        if not self.cfg.training.use_multi_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == torch.device("cpu"):
                raise EnvironmentError("No GPUs")
        self.model = self.model.to(self.device)
        if torch.cuda.is_available() and self.cfg.training.use_multi_gpu:
            # Move all model parameters and buffers to the GPU
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Create tensorboard writers
        self.train_writer = SummaryWriter(os.path.join(cfg.general.log_dir, 'writer'))



        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.training.lr,
            # weight_decay=self.cfg.training.weight_decay,
            # betas=self.cfg.training.betas
        )

        self.lr_schedular = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.cfg.training.decay_epochs,
            gamma=self.cfg.training.decay_gamma
        )

        self._init_criterion()
        self._init_metrics()
        self._init_dataset()

    def _init_metrics(self):
        self.metrics_class = Metrics()
        self.metrics_dict = {}
        for m in self.cfg.metrics.type:
            self.metrics_dict[m] = self._choose_metrics(m)

    def _choose_metrics(self, metric_str):
        try:
            func = getattr(self.metrics_class, metric_str)
            return func
        except Exception:
            raise AttributeError("metric '%s' not found" % metric_str)

    def _init_criterion(self):
        self.loss_weight = self.cfg.loss.weight
        self.loss_type = self.cfg.loss.type
        self.ssim_loss = SSIMLoss()
        # self.sp_cons_loss = ConsLoss(reduction="mean")
        self.sobel_fn = Sobel().to(self.device)

    def criterion(self, output, batch):
        camera = {
            "fx": batch["fx"],
            "fy": batch["fy"],
            "cx": batch["cx"],
            "cy": batch["cy"]
        }
        if self.running_type == "train" and self.cfg.model.normal.input_dim:
            pred = output[0]
            pred_sn = output[1]
            out = 0.8 * depth_loss(pred, batch["cur_gt_depth"], batch["mask"], reduction="mean")
            out = out + 0.2 * surface_normal_l1_loss(pred_sn, batch["depth_gt_sn"], batch["sn_mask"])
            # if self.cfg.model.normal.refine:
            #     out += 0.5 * depth_loss(output[2], batch["cur_gt_depth"], batch["mask"], reduction="mean")
        else:
            pred = output
            out = depth_loss(pred, batch["cur_gt_depth"], batch["mask"], reduction="mean")
        # pred = output
        # out = 0.5 * depth_loss(pred, batch["cur_gt_depth"], batch["mask"], reduction="mean")

        for i, loss_str in enumerate(self.loss_type):
            if loss_str == "depth_grad_loss_unsupervised":
                out = out + depth_grad_loss_unsupervised(pred, batch["cur_rgb"], batch["mask"]) * self.loss_weight[i]

            elif loss_str == "consistency_loss":
                out = out + photometric_geometry_loss(
                    batch["cur_rgb"], batch["forward_rgb"], pred, batch["forward_depth"],
                    batch["R_mat"], batch["t_vec"], camera, batch["mask"], ssim_fn=None
                ) * self.loss_weight[i]

            elif loss_str == "consistency_loss_v2":
                out = out + photometric_geometry_loss_v2(pred, self.cfg.model.forward_windows, batch, camera) * self.loss_weight[i]
            elif loss_str == "sobel_normal_loss":
                out = out + sobel_normal_loss(pred, batch["cur_gt_depth"], batch["mask"], self.sobel_fn) * self.loss_weight[i]
            elif loss_str == "cos_normal_loss":
                out = out + surface_normal_cos_loss(pred, batch["depth_gt_sn"], batch["mask"], camera, reduction="mean")
        return out

    def _init_dataset(self):
        if self.cfg.general.running_type == "train":
            self._init_train_dataset(["seq1", "seq2", "seq3", "seq4", "two_glass", "two_glass2"])
            self._init_val_dataset(["val_seq1", "val_seq2", "val_seq4"])
        elif self.cfg.general.running_type == "test":
            if self.cfg.dataset.name == "cleargrasp":
                val_dataset = ClearGrasp(self.cfg.dataset.dataset_dir, "val",
                                         self.cfg.general.frame_h, self.cfg.general.frame_w,
                                         specific_ds=None,
                                         max_norm=self.cfg.dataset.data_aug.depth_norm)
                self.val_data_loader = DataLoader(
                    val_dataset,
                    batch_size=self.cfg.training.valid_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=self.cfg.training.num_workers,
                    drop_last=True
                )

    def _init_val_dataset(self, scene_name):
        dataset_name = list(scene_name)
        val_dataset = []
        for scene in dataset_name:
            if self.cfg.dataset.name == "blenderglass_v2":
                ds = BlenderGlassV2(
                    root=self.cfg.dataset.dataset_dir,
                    scene=scene,
                    img_h=self.cfg.general.frame_h,
                    img_w=self.cfg.general.frame_w,
                    rgb_aug=self.cfg.dataset.data_aug.rgb_aug,
                    max_norm=self.cfg.dataset.data_aug.depth_norm,
                    depth_factor=self.depth_factor,
                    forward_windows_size=self.cfg.model.forward_windows
                )
            else:
                ds = BlenderGlass(
                    root=self.cfg.dataset.dataset_dir,
                    scene=scene,
                    img_h=self.cfg.general.frame_h,
                    img_w=self.cfg.general.frame_w,
                    rgb_aug=self.cfg.dataset.data_aug.rgb_aug,
                    max_norm=self.cfg.dataset.data_aug.depth_norm,
                    depth_factor=self.depth_factor
                )
            val_dataset.append(ds)
        dataset = ConcatDataset(val_dataset)
        self.val_data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.training.train_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg.training.num_workers,
            drop_last=True
        )

    def _init_train_dataset(self, scene_name):
        if type(scene_name) == str:
            dataset_name = [scene_name]
        else:
            dataset_name = scene_name
        training_dataset = []
        for scene in dataset_name:
            if self.cfg.dataset.name == "blenderglass_v2":
                ds = BlenderGlassV2(
                    root=self.cfg.dataset.dataset_dir,
                    scene=scene,
                    img_h=self.cfg.general.frame_h,
                    img_w=self.cfg.general.frame_w,
                    rgb_aug=self.cfg.dataset.data_aug.rgb_aug,
                    max_norm=self.cfg.dataset.data_aug.depth_norm,
                    depth_factor=self.depth_factor,
                    forward_windows_size=self.cfg.model.forward_windows
                )
            else:
                ds = BlenderGlass(
                    root=self.cfg.dataset.dataset_dir,
                    scene=scene,
                    img_h=self.cfg.general.frame_h,
                    img_w=self.cfg.general.frame_w,
                    rgb_aug=self.cfg.dataset.data_aug.rgb_aug,
                    max_norm=self.cfg.dataset.data_aug.depth_norm,
                    depth_factor=self.depth_factor
                )
            training_dataset.append(ds)
        training_dataset = ConcatDataset(training_dataset)
        self.train_data_loader = DataLoader(
            training_dataset,
            batch_size=self.cfg.training.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.training.num_workers,
            drop_last=True
        )

    def start_training(self):
        print("Total number of epochs: {}".format(self.cfg.training.epochs))
        loss_meter = Statistics()
        best_epoch_loss = 1e8
        training_start_time = time()
        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            self.running_type = "train"
            loss_meter.reset()
            batch_nums = len(self.train_data_loader)
            torch.cuda.empty_cache()
            for batch_idx, batch in enumerate(self.train_data_loader):
                batch_start_time = time()
                # clear gradient
                self.optimizer.zero_grad()
                dict2device(batch, self.device)
                pred = self.model(batch)
                batch_size = batch["cur_color"].size()[0]
                loss = self.criterion(pred, batch)
                loss.backward()
                self.optimizer.step()
                # update meter to collect loss per batch
                loss_meter.update(loss.cpu().item(), batch_size)
                # clear cache, preventing crash for small batch_size

                if batch_idx % self.print_freq == 0 or batch_idx < 10:
                    print(
                        "ep: {}, it {}/{} -- time: {:.2f}  loss: {:.4f}".format(
                            epoch + 1, batch_idx + 1, batch_nums,
                            time() - batch_start_time,
                            loss.cpu().item()
                        )
                    )
                if batch_idx % 5 == 0:
                    self.train_writer.add_scalar("loss", loss, epoch * batch_nums + batch_idx)

                del batch, loss, pred
            self.lr_schedular.step()
            validating_start_time = time()
            val_loss, metric_loss = self.start_validating()
            print("val use time: {:.4f}s".format(time() - validating_start_time))
            torch.cuda.empty_cache()
            metric_writer = {}
            for one in metric_loss.keys():
                metric_writer[one] = metric_loss[one]
            self.train_writer.add_scalars("metric", metric_writer, epoch)
            print(metric_loss)
            filename = None
            if epoch % self.cfg.training.save_epochs == 0:
                filename = "ckpt-epoch-{}.pth".format(epoch)
            if val_loss < best_epoch_loss:
                best_epoch_loss = val_loss
                filename = "ckpt-best.pth"
            if filename:
                output_path = os.path.join(self.cfg.general.checkpoint_dir, filename)
                save_dict = {
                    "epoch_idx": epoch,
                    # "lr": self.lr_schedular.state_dict()["param_groups"][0]["lr"],
                    "metrics_loss": metric_loss,
                    "training_loss": loss_meter.global_avg
                }
                if self.cfg.training.use_multi_gpu:
                    save_dict["model"] = self.model.module.state_dict()
                else:
                    save_dict["model"] = self.model.state_dict()
                torch.save(save_dict, output_path)
                print("Save model to {}".format(output_path))

        print("Training finished, total time used: {:.2f}h".format((time() - training_start_time) / 3600))

    def start_validating(self):
        print("start validating....")
        self.running_type = "val"
        metrics_num = len(self.metrics_dict.keys())
        metric_meter = AggregationMeter(metrics_num)
        loss_meter = Statistics()
        metrics_vec = []
        self.model.eval()
        loss_meter.reset()
        metric_meter.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_data_loader):
                dict2device(batch, self.device)
                pred = self.model(batch)
                batch_size = batch["cur_color"].size()[0]
                loss = self.criterion(pred, batch)
                for k, v in self.metrics_dict.items():
                    metrics_vec.append(
                        (self.metrics_dict[k](pred, batch["cur_gt_depth"], batch["mask"]).cpu().item(),
                         batch_size))
                metric_meter.update(metrics_vec)
                loss_meter.update(loss.cpu().item(), batch_size)
                metrics_vec.clear()
            print("validating finished, loss in val dataset: {:.4f}".format(loss_meter.global_avg))
        res = {}
        for idx, k in enumerate(self.metrics_dict.keys()):
            res[k] = metric_meter.global_avg[idx]
        return loss_meter.global_avg, res
