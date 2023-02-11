import os

import numpy as np
import torch
import re

from time import time

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from cfg.datasets.BlenderGlass_dataset import BlenderGlass
from cfg.datasets.cleargrasp_dataset import ClearGrasp
from utils.meter import AggregationMeter, Statistics
from train.metrics import ConsLoss, Metrics, \
    surface_normal_cos_loss, depth_loss, pairwise_L1_depth_loss, \
    surface_normal_l1_loss, surface_normal_loss

from utils.photometric_loss import photometric_loss


class Trainer:

    def __init__(self, cfg, model, pretrained_model_path=None):
        self.cfg = cfg
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        if self.cfg.general.pretrained_weight:
            self.pretrained_model_path = self.cfg.general.pretrained_weight
        if self.pretrained_model_path:
            print("load weight from:{}".format(self.pretrained_model_path))
            snapshot = torch.load(self.pretrained_model_path)
            self.model.load_state_dict(snapshot.pop("model"))
        else:
            print("do not use pretrained weight")
        if not self.cfg.training.use_multi_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == torch.device("cpu"):
                raise EnvironmentError("No GPUs")
        self.model = self.model.to(self.device)
        if torch.cuda.is_available() and self.cfg.training.use_multi_gpu:
            # Move all model parameters and buffers to the GPU
            self.model = torch.nn.DataParallel(self.model)

        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Create tensorboard writers
        self.train_writer = SummaryWriter(os.path.join(cfg.general.log_dir, 'writer'))

        # DataLoader init
        self.val_data_loader = None
        self.train_data_loader = None
        self._init_optimizer(
            lr=self.cfg.training.lr,
            step_size=self.cfg.training.decay_epochs,
            gamma=self.cfg.training.decay_gamma
        )

        self._init_criterion()
        self._init_metrics()

    def _init_optimizer(self, lr=2e-4, weight_decay=0, betas=(0.9, 0.999), step_size=10, gamma=0.5):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            # weight_decay=weight_decay,
            # betas=betas
        )

        self.lr_schedular = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )

    def _choose_metrics(self, metric_str):
        try:
            func = getattr(self.metrics_class, metric_str)
            return func
        except Exception:
            raise AttributeError("metric '%s' not found" % metric_str)

    def _init_criterion(self):
        self.loss_weight = self.cfg.loss.weight
        self.loss_type = self.cfg.loss.type
        self.cons_loss = ConsLoss(reduction="mean").to(self.device)

    def criterion(self, pred, batch, camera):
        out = 0
        for i, loss_str in enumerate(self.loss_type):
            if loss_str == "depth_loss":
                out += depth_loss(pred, batch["gt_depth"], batch["loss_mask"], reduction="mean", beta=0.02) * \
                       self.loss_weight[i]
            elif loss_str == "surface_normal_cos":
                out += surface_normal_cos_loss(pred, batch["depth_gt_sn"], batch["loss_mask"], camera,
                                               reduction="sum") * \
                       self.loss_weight[i]
            elif loss_str == "surface_normal_l1":
                out += surface_normal_l1_loss(pred, batch["depth_gt_sn"], batch["loss_mask"], camera, reduction="sum",
                                              beta=0.5) * self.loss_weight[i]
            elif loss_str == "pairwise_l1":
                out += pairwise_L1_depth_loss(pred, batch["gt_depth"], batch["loss_mask"]) * self.loss_weight[i]

            elif loss_str == "depth_normal_refine":
                out += self.cons_loss(pred, batch["depth_gt_sn"], batch["loss_mask"], batch["gt_depth"], camera)
        return out

    def _init_metrics(self):
        self.metrics_class = Metrics()
        self.metrics_dict = {}
        for m in self.cfg.metrics.type:
            self.metrics_dict[m] = self._choose_metrics(m)

    def _init_pic_dataset(self):
        if self.cfg.dataset.name == "cleargrasp":
            train_dataset = ClearGrasp(self.cfg.dataset.dataset_dir, "train",
                                       self.cfg.general.frame_h, self.cfg.general.frame_w,
                                       specific_ds=None,
                                       max_norm=self.cfg.dataset.data_aug.depth_norm,
                                       rgb_aug=self.cfg.dataset.data_aug.rgb_aug)
            val_dataset = ClearGrasp(self.cfg.dataset.dataset_dir, "val",
                                     self.cfg.general.frame_h, self.cfg.general.frame_w,
                                     specific_ds=None,
                                     max_norm=self.cfg.dataset.data_aug.depth_norm)
        else:
            raise NotImplementedError("Dose not support dataset: {}".format(self.cfg.dataset.name))
        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.training.num_workers,
            drop_last=True
        )
        self.val_data_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.valid_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg.training.num_workers,
            drop_last=True
        )
        print("Prepare dataset: {} ---- Done".format(self.cfg.dataset.name))

    def start_training(self):
        self._init_pic_dataset()
        print("Total number of epochs: {}".format(self.cfg.training.epochs))
        loss_meter = Statistics()
        best_epoch_loss = 1e8
        training_start_time = time()
        # self.model.train()
        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            loss_meter.reset()
            batch_nums = len(self.train_data_loader)
            for batch_idx, batch in enumerate(self.train_data_loader):
                batch_start_time = time()
                # clear gradient
                self.optimizer.zero_grad()
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                camera = {
                    "cx": batch["cx"],
                    "cy": batch["cy"],
                    "fx": batch["fx"],
                    "fy": batch["fy"]
                }
                pred = self.model(batch)
                batch_size = batch["color"].size()[0]
                loss = self.criterion(pred, batch, camera)
                loss.backward()
                self.optimizer.step()
                # update meter to collect loss per batch
                loss_meter.update(loss.cpu().item(), batch_size)
                # clear cache, preventing crash for small batch_size

                if batch_idx % 100 == 0:
                    print(
                        "ep: {}, it {}/{} -- time: {:.2f}  loss: {:.4f}".format(
                            epoch + 1, batch_idx + 1, batch_nums,
                            time() - batch_start_time,
                            loss_meter.global_avg
                        )
                    )
                if batch_idx % 1000 == 0:
                    self.train_writer.add_scalar("loss", loss, epoch * batch_nums + batch_idx)
                # del batch, loss, pred, camera
                del batch, loss, pred, camera

            self.lr_schedular.step()
            val_loss, metric_loss = self.pic_validating()
            # clear cache, otherwise CUDA may out of memory
            torch.cuda.empty_cache()
            metric_writer = {}
            threshold_writer = {}
            for one in metric_loss.keys():
                if "depth_failing" in one:
                    threshold_writer[one] = metric_loss[one]
                else:
                    metric_writer[one] = metric_loss[one]
            self.train_writer.add_scalars("metric", metric_writer, epoch)
            self.train_writer.add_scalars("threshold", threshold_writer, epoch)
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
                    # "lr": self.optimizer.state_dict()["param_groups"][0]["lr"],
                    "best_epoch_loss": val_loss,
                    "metrics_loss": metric_loss,
                }
                if self.cfg.training.use_multi_gpu:
                    save_dict["model"] = self.model.module.state_dict()
                else:
                    save_dict["model"] = self.model.state_dict()
                torch.save(save_dict, output_path)
                print("Save model to {}".format(output_path))

        print("Training finished, total time used: {:.2f}h".format((time() - training_start_time) / 3600))

    def start_validating(self):
        self._init_pic_dataset()
        self.pic_validating()

    def pic_validating(self):
        print("validating....")
        metrics_num = len(self.metrics_dict.keys())
        metric_meter = AggregationMeter(metrics_num)
        loss_meter = Statistics()
        metrics_vec = []
        self.model.eval()
        loss_meter.reset()
        metric_meter.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_data_loader):
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                camera = {
                    "cx": batch["cx"],
                    "cy": batch["cy"],
                    "fx": batch["fx"],
                    "fy": batch["fy"]
                }
                pred = self.model(batch)
                batch_size = pred.size()[0]
                loss = self.criterion(pred, batch, camera)
                for k, v in self.metrics_dict.items():
                    metrics_vec.append(
                        (self.metrics_dict[k](pred, batch["gt_depth"], batch["loss_mask"]).cpu().item(),
                         batch_size))
                metric_meter.update(metrics_vec)
                loss_meter.update(loss.cpu().item(), batch_size)
                metrics_vec.clear()
            print("validating finished, loss in val dataset: {:.4f}".format(loss_meter.global_avg))
        res = {}
        for idx, k in enumerate(self.metrics_dict.keys()):
            res[k] = metric_meter.global_avg[idx]
        return loss_meter.global_avg, res
