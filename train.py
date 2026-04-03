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
from utils.geometry import params_8_to_params_7

class BBoxTask(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        self.model = Fusion3DDetector()
        
        # Reduced pos_weight for Gaussian: Since multiple pixels are now "positive"
        # per object, a weight of 5.0-8.0 is usually sufficient to balance the grid.
        self.register_buffer("pos_weight", torch.tensor([float(cfg['train'].get('pos_weight', 8.0))]))
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        self.reg_loss = nn.HuberLoss(delta=1.0)

    def training_step(self, batch, batch_idx):
        # 1. Forward Pass
        pred_logits, pred_boxes = self.model(batch['rgb'], batch['pc'])
        target_map = batch['target_map'] 
        
        gt_logits = target_map[:, 0:1, :, :]
        gt_boxes = target_map[:, 1:, :, :]
        
        # 2. Classification Loss (Heatmap)
        loss_cls = self.cls_loss(pred_logits, gt_logits)
        
        # 3. Masked Regression Loss
        # Only compute regression loss at the peak of the Gaussian (1.0)
        mask = (gt_logits == 1.0)
        
        if mask.sum() > 0:
            masked_pred = pred_boxes.permute(0, 2, 3, 1)[mask.squeeze(1)]
            masked_gt = gt_boxes.permute(0, 2, 3, 1)[mask.squeeze(1)]
            loss_reg = self.reg_loss(masked_pred, masked_gt)
        else:
            loss_reg = pred_boxes.sum() * 0.0

        # 4. Total Loss
        reg_weight = float(self.cfg['train'].get('reg_weight', 20.0))
        total_loss = loss_cls + (reg_weight * loss_reg)
        
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/reg_loss", loss_reg)
        self.log("train/cls_loss", loss_cls)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pred_logits, pred_boxes = self.model(batch['rgb'], batch['pc'])
        target_map = batch['target_map']
        
        loss_cls = self.cls_loss(pred_logits, target_map[:, 0:1, :, :])
        self.log("val/loss", loss_cls, prog_bar=True)
        
        mask = target_map[:, 0:1, :, :] == 1.0
        if mask.sum() > 0:
            indices = torch.where(mask)
            b, _, y, x = [idx[0] for idx in indices]
            
            p8 = pred_boxes[b, :, y, x].detach().cpu().numpy()
            g8 = target_map[b, 1:, y, x].detach().cpu().numpy()
            
            p7 = params_8_to_params_7(p8)
            g7 = params_8_to_params_7(g8)
            
            iou = get_3d_iou(torch.from_numpy(p7).to(self.device), 
                             torch.from_numpy(g7).to(self.device))
            self.log("val/iou", iou, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.cfg['train']['learning_rate']),
            weight_decay=float(self.cfg['train'].get('weight_decay', 0.01))
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(self.cfg['train']['learning_rate']),
            total_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

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
    
    # MODIFIED: Checkpoint every 10 epochs
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg['logging']['save_dir'],
        filename="3d-fusion-{epoch:02d}-{val_loss:.2f}",
        monitor="val/loss",
        mode="min",
        save_top_k=5,           # Keep more models since we save less often
        every_n_epochs=10,      # <--- KEY CHANGE
        save_on_train_epoch_end=True # Ensures it triggers exactly at epoch end
    )
    
    trainer = L.Trainer(
        max_epochs=cfg['train']['max_epochs'],
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        gradient_clip_val=float(cfg['train'].get('gradient_clip_val', 1.0)),
        accumulate_grad_batches=int(cfg['train'].get('accumulate_grad_batches', 1)),
        precision="16-mixed"
    )

    trainer.fit(BBoxTask(cfg), datamodule=dm)

if __name__ == "__main__":
    main()