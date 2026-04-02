import os
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Import your custom modules
from configs.load_config import load_config
from data.data_module import ThreeDDataModule
from models.pipeline_main import Fusion3DDetector
from utils.metrics import get_3d_iou

class BBoxTask(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Initialize the Fusion Model
        self.model = Fusion3DDetector()
        
        # Loss functions
        self.reg_loss = torch.nn.SmoothL1Loss() # Robust to outliers in PC
        self.cls_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, rgb, pc):
        return self.model(rgb, pc)

    def training_step(self, batch, batch_idx):
        rgb, pc, gt_box = batch['rgb'], batch['pc'], batch['gt_box']
        presence = batch['presence']
        
        logits, pred_box = self.model(rgb, pc)
        
        # 1. Classification Loss (Is an object there?)
        c_loss = self.cls_loss(logits, presence)
        
        # 2. Regression Loss (Box parameters)
        # We only calculate regression loss for samples where an object exists
        r_loss = self.reg_loss(pred_box, gt_box)
        
        total_loss = c_loss + (self.cfg['train']['reg_weight'] * r_loss)
        
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/reg_loss", r_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        rgb, pc, gt_box = batch['rgb'], batch['pc'], batch['gt_box']
        logits, pred_box = self.model(rgb, pc)
        
        val_loss = self.reg_loss(pred_box, gt_box)
        
        # Calculate 3D IoU for the batch
        iou = get_3d_iou(pred_box[0], gt_box[0]) # Simplified for single object
        
        self.log("val/loss", val_loss, prog_bar=True)
        self.log("val/iou", iou, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.cfg['train']['learning_rate']),
            weight_decay=float(self.cfg['train']['weight_decay'])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg['train']['max_epochs'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():
    # 1. Load Configuration
    cfg = load_config("configs/config.yaml")
    
    # 2. Setup DataModule
    dm = ThreeDDataModule(
        data_dir=cfg['data']['root_dir'],
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers']
    )
    
    # 3. Setup Logger & Callbacks
    logger = None
    if cfg['logging']['use_wandb']:
        logger = WandbLogger(project=cfg['project_name'], name=cfg['experiment_name'])
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg['logging']['save_dir'],
        filename="best-bbox-model-{epoch:02d}-{val_iou:.2f}",
        monitor="val/iou",
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 4. Initialize Task
    task = BBoxTask(cfg)

    # 5. Initialize Trainer
    trainer = L.Trainer(
        max_epochs=cfg['train']['max_epochs'],
        accelerator=cfg['train']['accelerator'],
        devices=cfg['train']['devices'],
        precision=cfg['train']['precision'],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg['logging']['log_every_n_steps']
    )

    # 6. Start Training
    trainer.fit(task, datamodule=dm)

if __name__ == "__main__":
    main()