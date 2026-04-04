import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Custom modules
from configs.load_config import load_config
from data.data_module import ThreeDDataModule
from models.pipeline_main import Fusion3DDetector
from utils.metrics import get_3d_iou
from utils.geometry import params_10_to_params_7

class BBoxTask(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = Fusion3DDetector()
        
        # Balanced Focal-like weight for sparse heatmap
        self.register_buffer("pos_weight", torch.tensor([float(cfg['train'].get('pos_weight', 10.0))]))
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.reg_loss = nn.HuberLoss(delta=1.0) # Robust to PC outliers

    def training_step(self, batch, batch_idx):
        # 1. Forward Pass
        pred_logits, pred_boxes = self.model(batch['rgb'], batch['pc'])
        target_map = batch['target_map'] 
        
        gt_logits = target_map[:, 0:1, :, :]
        gt_boxes = target_map[:, 1:, :, :] # [B, 10, H, W]
        
        # 2. Heatmap Loss
        loss_cls = self.cls_loss(pred_logits, gt_logits)
        
        # 3. Expanded Mask Regression (Stability Fix)
        # We train on the Gaussian "shoulders" to get more than 1 pixel of signal
        mask = (gt_logits > 0.1) 
        
        if mask.sum() > 0:
            mask_bool = mask.squeeze(1) 
            masked_pred = pred_boxes.permute(0, 2, 3, 1)[mask_bool]
            masked_gt = gt_boxes.permute(0, 2, 3, 1)[mask_bool]
            loss_reg = self.reg_loss(masked_pred, masked_gt)
        else:
            loss_reg = pred_boxes.sum() * 0.0

        # 4. Total Loss
        reg_w = float(self.cfg['train'].get('reg_weight', 40.0)) # Increased for depth focus
        total_loss = loss_cls + (reg_w * loss_reg)
        
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/reg_loss", loss_reg)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pred_logits, pred_boxes = self.model(batch['rgb'], batch['pc'])
        target_map = batch['target_map']
        grid_size = tuple(self.cfg['data']['grid_size'])
        
        # Monitor heatmaps
        loss_cls = self.cls_loss(pred_logits, target_map[:, 0:1, :, :])
        self.log("val/loss", loss_cls, prog_bar=True)
        
        # Extract peak for IoU calculation
        mask = target_map[:, 0:1, :, :] == 1.0
        if mask.sum() > 0:
            indices = torch.where(mask)
            b, _, y, x = [idx[0] for idx in indices]
            
            # Predict & Target (10 channels: 8 params + 2 offsets)
            p10 = pred_boxes[b, :, y, x].detach().cpu().numpy()
            g10 = target_map[b, 1:, y, x].detach().cpu().numpy()
            
            # Reconstruct World Coordinates (Life-sized meters)
            p7 = params_10_to_params_7(p10, (y.item(), x.item()), grid_size=grid_size)
            g7 = params_10_to_params_7(g10, (y.item(), x.item()), grid_size=grid_size)
            
            # Undo Z-shift for true world placement
            p7[2] += 1.5
            g7[2] += 1.5
            
            iou = get_3d_iou(torch.from_numpy(p7).to(self.device), 
                             torch.from_numpy(g7).to(self.device))
            self.log("val/iou", iou, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.cfg['train']['learning_rate']),
            weight_decay=float(self.cfg['train'].get('weight_decay', 0.01))
        )
        # Warm-up is critical for the Attention layers in the head
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(self.cfg['train']['learning_rate']),
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

def main():
    torch.set_float32_matmul_precision('high')
    cfg = load_config("configs/config.yaml")
    
    dm = ThreeDDataModule(
        data_dir=cfg['data']['root_dir'],
        batch_size=cfg['data']['batch_size'], 
        num_workers=cfg['data']['num_workers'],
        grid_size=tuple(cfg['data']['grid_size']) 
    )
    
    logger = WandbLogger(project=cfg['project_name']) if cfg['logging']['use_wandb'] else None
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg['logging']['save_dir'],
        filename="fusion-3d-{epoch:02d}-{val_iou:.2f}",
        monitor="val/iou",
        mode="max",
        save_top_k=3,
        every_n_epochs=5
    )
    
    trainer = L.Trainer(
        max_epochs=cfg['train']['max_epochs'],
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        gradient_clip_val=1.0,
        accumulate_grad_batches=int(cfg['train'].get('accumulate_grad_batches', 1)),
        precision="16-mixed"
    )
    trainer.fit(BBoxTask(cfg), datamodule=dm)

if __name__ == "__main__":
    main()