import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

from .dataset import ThreeDObjectDataset


class ThreeDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=16,
        num_workers=4,
        target_size=(480, 640),
        grid_size=(15, 20),
        train_split=0.8,
        range_x=(-0.3, 0.3),
        range_z=(-0.8, 0.0),
        gaussian_sigma=1.0,
        z_shift=1.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = tuple(target_size)
        self.grid_size = tuple(grid_size)
        self.train_split = float(train_split)
        self.range_x = tuple(range_x)
        self.range_z = tuple(range_z)
        self.gaussian_sigma = float(gaussian_sigma)
        self.z_shift = float(z_shift)

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        full_dataset = ThreeDObjectDataset(
            root_dir=self.data_dir,
            target_size=self.target_size,
            grid_size=self.grid_size,
            training=False,
            range_x=self.range_x,
            range_z=self.range_z,
            gaussian_sigma=self.gaussian_sigma,
            z_shift=self.z_shift,
        )

        train_len = int(self.train_split * len(full_dataset))
        val_len = len(full_dataset) - train_len

        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(
            full_dataset, [train_len, val_len], generator=generator
        )

        train_dataset = ThreeDObjectDataset(
            root_dir=self.data_dir,
            target_size=self.target_size,
            grid_size=self.grid_size,
            training=True,
            range_x=self.range_x,
            range_z=self.range_z,
            gaussian_sigma=self.gaussian_sigma,
            z_shift=self.z_shift,
        )

        val_dataset = ThreeDObjectDataset(
            root_dir=self.data_dir,
            target_size=self.target_size,
            grid_size=self.grid_size,
            training=False,
            range_x=self.range_x,
            range_z=self.range_z,
            gaussian_sigma=self.gaussian_sigma,
            z_shift=self.z_shift,
        )

        self.train_ds = torch.utils.data.Subset(train_dataset, train_subset.indices)
        self.val_ds = torch.utils.data.Subset(val_dataset, val_subset.indices)

    def train_dataloader(self):
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(self.train_ds, **loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(self.val_ds, **loader_kwargs)