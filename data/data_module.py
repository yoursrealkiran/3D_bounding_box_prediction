import lightning as L
from torch.utils.data import DataLoader, random_split
from .dataset import ThreeDObjectDataset
import torch

class ThreeDDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=4, grid_size=(15, 20)):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grid_size = grid_size

    def setup(self, stage=None):
        full_dataset = ThreeDObjectDataset(self.data_dir, grid_size=self.grid_size)
        
        train_len = int(0.8 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        
        generator = torch.Generator().manual_seed(42)
        self.train_ds, self.val_ds = random_split(full_dataset, [train_len, val_len], generator=generator)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True # Better for BatchNorm stability
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True
        )