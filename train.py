import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from configs.load_config import load_config
from data.data_module import ThreeDDataModule
from models.pipeline_main import Fusion3DDetector
from utils.metrics import get_3d_iou, center_distance, size_l1_error, yaw_error_rad
from utils.geometry import params_10_to_params_7


class FocalHeatmapLoss(nn.Module):
    """
    CenterNet-style focal loss for dense heatmaps.
    Expects:
      pred_logits: raw logits [B, 1, H, W]
      gt: target heatmap in [0, 1] [B, 1, H, W]
    """
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits, gt):
        pred = torch.sigmoid(pred_logits).clamp(min=1e-6, max=1 - 1e-6)

        pos_inds = (gt.eq(1)).float()
        neg_inds = (gt.lt(1)).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos > 0:
            loss = (pos_loss + neg_loss) / num_pos
        else:
            loss = neg_loss

        return loss


class BBoxTask(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.model = Fusion3DDetector(
            freeze_rgb_layers=bool(cfg["model"].get("freeze_rgb_layers", True)),
            head_dropout=float(cfg["model"].get("dropout", 0.2)),
            output_grid_size=tuple(cfg["data"]["grid_size"]),
        )

        self.cls_loss = FocalHeatmapLoss(
            alpha=float(cfg["train"].get("focal_alpha", 2.0)),
            beta=float(cfg["train"].get("focal_beta", 4.0)),
        )
        self.reg_loss = nn.HuberLoss(delta=float(cfg["train"].get("huber_delta", 1.0)))

        self.seg_loss = nn.BCEWithLogitsLoss()

    @property
    def grid_size(self):
        return tuple(self.cfg["data"]["grid_size"])

    @property
    def range_x(self):
        return tuple(self.cfg["data"]["range_x"])

    @property
    def range_z(self):
        return tuple(self.cfg["data"]["range_z"])

    @property
    def z_shift(self):
        return float(self.cfg["data"].get("z_shift", 1.5))

    def forward(self, rgb, pc):
        return self.model(rgb, pc)

    def _decode_p10_to_world(self, p10, grid_pos):
        return params_10_to_params_7(
            p10,
            grid_pos=grid_pos,
            grid_size=self.grid_size,
            range_x=self.range_x,
            range_z=self.range_z,
        )

    def training_step(self, batch, batch_idx):
        rgb = batch["rgb"]
        pc = batch["pc"]
        target_map = batch["target_map"]
        reg_mask = batch["reg_mask"] > 0  # [B, H, W]
        gt_mask = batch["mask"].float()   # [B, 1, H_img, W_img] from dataset

        pred_logits, pred_boxes, pred_seg = self.model(rgb, pc)

        gt_logits = target_map[:, 0:1, :, :]
        gt_boxes = target_map[:, 1:, :, :]

        # Resize mask target to segmentation head resolution
        gt_mask = F.interpolate(
            gt_mask,
            size=pred_seg.shape[-2:],
            mode="nearest",
        )

        # 1. Heatmap classification loss
        loss_cls = self.cls_loss(pred_logits, gt_logits)

        # 2. Regression loss only at exact object centers
        if reg_mask.any():
            pred_boxes_hw = pred_boxes.permute(0, 2, 3, 1)  # [B, H, W, 10]
            gt_boxes_hw = gt_boxes.permute(0, 2, 3, 1)      # [B, H, W, 10]

            masked_pred = pred_boxes_hw[reg_mask]
            masked_gt = gt_boxes_hw[reg_mask]

            loss_reg = self.reg_loss(masked_pred, masked_gt)
        else:
            loss_reg = pred_boxes.sum() * 0.0

        # 3. Segmentation loss
        loss_seg = self.seg_loss(pred_seg, gt_mask)

        reg_w = float(self.cfg["train"].get("reg_weight", 50.0))
        seg_w = float(self.cfg["train"].get("seg_weight", 0.5))
        cls_w = float(self.cfg["train"].get("cls_weight", 2.0))

        total_loss = cls_w * loss_cls + reg_w * loss_reg + seg_w * loss_seg

        self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=rgb.size(0))
        self.log("train/loss_cls", loss_cls, on_step=True, on_epoch=True, batch_size=rgb.size(0))
        self.log("train/loss_reg", loss_reg, on_step=True, on_epoch=True, batch_size=rgb.size(0))
        self.log("train/loss_seg", loss_seg, on_step=True, on_epoch=True, batch_size=rgb.size(0))

        with torch.no_grad():
            probs = torch.sigmoid(pred_logits)
            pos_mask = gt_logits > 0
            pos_mean = probs[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=self.device)

            self.log("train/heatmap_pos_mean", pos_mean, on_step=True, on_epoch=False, batch_size=rgb.size(0))
            self.log("train/heatmap_all_mean", probs.mean(), on_step=True, on_epoch=False, batch_size=rgb.size(0))
            self.log("train/num_reg", reg_mask.sum().float(), on_step=True, on_epoch=False, batch_size=rgb.size(0))

            seg_probs = torch.sigmoid(pred_seg)
            self.log("train/seg_mean", seg_probs.mean(), on_step=True, on_epoch=False, batch_size=rgb.size(0))

        return total_loss

    def validation_step(self, batch, batch_idx):
        rgb = batch["rgb"]
        pc = batch["pc"]
        target_map = batch["target_map"]
        reg_mask = batch["reg_mask"] > 0
        gt_mask = batch["mask"].float()

        pred_logits, pred_boxes, pred_seg = self.model(rgb, pc)

        gt_logits = target_map[:, 0:1, :, :]
        gt_boxes = target_map[:, 1:, :, :]

        gt_mask = F.interpolate(
            gt_mask,
            size=pred_seg.shape[-2:],
            mode="nearest",
        )

        loss_cls = self.cls_loss(pred_logits, gt_logits)

        if reg_mask.any():
            pred_boxes_hw = pred_boxes.permute(0, 2, 3, 1)
            gt_boxes_hw = gt_boxes.permute(0, 2, 3, 1)

            masked_pred = pred_boxes_hw[reg_mask]
            masked_gt = gt_boxes_hw[reg_mask]

            loss_reg = self.reg_loss(masked_pred, masked_gt)
        else:
            loss_reg = pred_boxes.sum() * 0.0

        loss_seg = self.seg_loss(pred_seg, gt_mask)

        reg_w = float(self.cfg["train"].get("reg_weight", 50.0))
        seg_w = float(self.cfg["train"].get("seg_weight", 0.5))
        cls_w = float(self.cfg["train"].get("cls_weight", 2.0))

        total_loss = cls_w * loss_cls + reg_w * loss_reg + seg_w * loss_seg

        self.log("val/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=rgb.size(0))
        self.log("val/loss_cls", loss_cls, on_step=False, on_epoch=True, batch_size=rgb.size(0))
        self.log("val/loss_reg", loss_reg, on_step=False, on_epoch=True, batch_size=rgb.size(0))
        self.log("val/loss_seg", loss_seg, on_step=False, on_epoch=True, batch_size=rgb.size(0))
        self.log("val/num_reg", reg_mask.sum().float(), on_step=False, on_epoch=True, batch_size=rgb.size(0))

        indices = torch.nonzero(reg_mask, as_tuple=False)  # [N, 3] => (b, y, x)

        if indices.numel() == 0:
            zero = torch.tensor(0.0, device=self.device)
            self.log("val/iou", zero, prog_bar=True, on_epoch=True, batch_size=rgb.size(0))
            self.log("val/center_err", zero, on_epoch=True, batch_size=rgb.size(0))
            self.log("val/size_l1", zero, on_epoch=True, batch_size=rgb.size(0))
            self.log("val/yaw_err_rad", zero, on_epoch=True, batch_size=rgb.size(0))
            return total_loss

        ious = []
        center_errors = []
        size_errors = []
        yaw_errors = []

        pred_boxes_cpu = pred_boxes.detach().cpu()
        target_map_cpu = target_map.detach().cpu()

        for idx_triplet in indices:
            b, y, x = idx_triplet.tolist()

            p10 = pred_boxes_cpu[b, :, y, x].numpy()
            g10 = target_map_cpu[b, 1:, y, x].numpy()

            p7 = self._decode_p10_to_world(p10, grid_pos=(y, x))
            g7 = self._decode_p10_to_world(g10, grid_pos=(y, x))

            # Undo z-shift to report in original world coordinates
            p7[2] += self.z_shift
            g7[2] += self.z_shift

            p7_t = torch.from_numpy(p7).to(self.device)
            g7_t = torch.from_numpy(g7).to(self.device)

            ious.append(get_3d_iou(p7_t, g7_t))
            center_errors.append(center_distance(p7_t, g7_t))
            size_errors.append(size_l1_error(p7_t, g7_t))
            yaw_errors.append(yaw_error_rad(p7_t, g7_t))

        mean_iou = torch.stack(ious).mean()
        mean_center = torch.stack(center_errors).mean()
        mean_size = torch.stack(size_errors).mean()
        mean_yaw = torch.stack(yaw_errors).mean()

        self.log("val/iou", mean_iou, prog_bar=True, on_step=False, on_epoch=True, batch_size=rgb.size(0))
        self.log("val/iou_step", mean_iou, on_step=True, on_epoch=False, batch_size=rgb.size(0))
        self.log("val/center_err", mean_center, on_step=False, on_epoch=True, batch_size=rgb.size(0))
        self.log("val/size_l1", mean_size, on_step=False, on_epoch=True, batch_size=rgb.size(0))
        self.log("val/yaw_err_rad", mean_yaw, on_step=False, on_epoch=True, batch_size=rgb.size(0))

        return total_loss

    def configure_optimizers(self):
        learning_rate = float(self.cfg["train"]["learning_rate"])
        weight_decay = float(self.cfg["train"].get("weight_decay", 0.01))

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=float(self.cfg["train"].get("onecycle_pct_start", 0.1)),
            div_factor=float(self.cfg["train"].get("onecycle_div_factor", 25.0)),
            final_div_factor=float(self.cfg["train"].get("onecycle_final_div_factor", 1e4)),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def main():
    torch.set_float32_matmul_precision("high")
    cfg = load_config("configs/config.yaml")

    seed = int(cfg["train"].get("seed", 42))
    L.seed_everything(seed, workers=True)

    dm = ThreeDDataModule(
        data_dir=cfg["data"]["root_dir"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        target_size=tuple(cfg["data"]["target_size"]),
        grid_size=tuple(cfg["data"]["grid_size"]),
        train_split=float(cfg["data"].get("train_split", 0.8)),
        range_x=tuple(cfg["data"]["range_x"]),
        range_z=tuple(cfg["data"]["range_z"]),
        gaussian_sigma=float(cfg["model"].get("gaussian_sigma", 1.0)),
        z_shift=float(cfg["data"].get("z_shift", 1.5)),
    )

    logger = None
    if cfg["logging"].get("use_wandb", False):
        logger = WandbLogger(
            project=cfg["project_name"],
            name=cfg.get("experiment_name", "fusion_3d_run"),
        )

    os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg["logging"]["save_dir"],
        filename="fusion-3d-{epoch:02d}-{val_iou:.4f}",
        monitor="val/iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=int(cfg["logging"].get("checkpoint_every_n_epochs", 1)),
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs=int(cfg["train"]["max_epochs"]),
        accelerator=cfg["train"].get("accelerator", "gpu"),
        devices=cfg["train"].get("devices", 1),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=float(cfg["train"].get("gradient_clip_val", 1.0)),
        accumulate_grad_batches=int(cfg["train"].get("accumulate_grad_batches", 1)),
        precision=cfg["train"].get("precision", "16-mixed"),
        log_every_n_steps=int(cfg["logging"].get("log_every_n_steps", 10)),
        deterministic=bool(cfg["train"].get("deterministic", False)),
    )

    model = BBoxTask(cfg)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()