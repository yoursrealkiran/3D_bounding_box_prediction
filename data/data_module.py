import lightning as L
from torch.utils.data import DataLoader, random_split
from .dataset import ThreeDObjectDataset

class ThreeDDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = ThreeDObjectDataset(self.data_dir)
        # 80/20 split for small datasets
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        self.train_ds, self.val_ds = random_split(dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )