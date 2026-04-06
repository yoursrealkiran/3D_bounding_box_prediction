import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from .utils import corners_8_to_params_8, draw_msra_gaussian, world_to_grid


class ThreeDObjectDataset(Dataset):
    """
    Full-scene multi-object dataset for dense grid prediction.

    Returned dict:
        rgb:        (3, H, W)
        pc:         (3, H, W)
        target_map: (11, Gh, Gw)
                    [0]    heatmap
                    [1:9]  [x, y, z, w, l, h, sin, cos] at center cells only
                    [9:11] [off_x, off_z] at center cells only
        reg_mask:   (Gh, Gw) 1 only at exact object centers
        mask:       (1, H, W) foreground segmentation mask
    """

    def __init__(
        self,
        root_dir,
        target_size=(480, 640),
        grid_size=(15, 20),
        training=True,
        range_x=(-10.0, 10.0),
        range_z=(0.0, 20.0),
        gaussian_sigma=1.0,
        z_shift=1.5,
    ):
        self.root_dir = root_dir
        self.target_size = tuple(target_size)
        self.grid_size = tuple(grid_size)
        self.training = training
        self.range_x = tuple(range_x)
        self.range_z = tuple(range_z)
        self.gaussian_sigma = float(gaussian_sigma)
        self.z_shift = float(z_shift)

        self.scenes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

    def __len__(self):
        return len(self.scenes)

    def _load_scene(self, scene_path):
        rgb_np = np.array(
            Image.open(os.path.join(scene_path, "rgb.jpg")).convert("RGB")
        ).astype(np.float32) / 255.0

        pc_np = np.load(os.path.join(scene_path, "pc.npy")).astype(np.float32).copy()
        bbox3d_np = np.load(os.path.join(scene_path, "bbox3d.npy")).astype(np.float32).copy()

        
        mask_np = np.load(os.path.join(scene_path, "mask.npy"))
        mask_np = mask_np.astype(bool)

        if mask_np.ndim == 3 and mask_np.shape[0] > 0:
            mask_fg = np.any(mask_np, axis=0).astype(np.float32)
        elif mask_np.ndim == 2:
            mask_fg = mask_np.astype(np.float32)
        else:
            mask_fg = np.zeros(rgb_np.shape[:2], dtype=np.float32)

        return rgb_np, pc_np, bbox3d_np, mask_fg

    def _apply_flip(self, rgb_np, pc_np, bbox3d_np, mask_fg):
        """
        Horizontal image flip + corresponding 3D X flip.
        Must be applied consistently to all aligned modalities.
        """
        
        rgb_np = np.flip(rgb_np, axis=1).copy()
        mask_fg = np.flip(mask_fg, axis=1).copy()
        pc_np = np.flip(pc_np, axis=2).copy()
        pc_np[0, :, :] = -pc_np[0, :, :]
        bbox3d_np[:, :, 0] = -bbox3d_np[:, :, 0]

        return rgb_np, pc_np, bbox3d_np, mask_fg

    def _align_coordinates(self, pc_np, bbox3d_np):
        """
        Apply the same coordinate alignment to both point cloud and boxes.
        """
        pc_np[0, :, :] = -pc_np[0, :, :]
        pc_np[1, :, :] = -pc_np[1, :, :]
        pc_np[2, :, :] -= self.z_shift

        bbox3d_np[:, :, 0] = -bbox3d_np[:, :, 0]
        bbox3d_np[:, :, 1] = -bbox3d_np[:, :, 1]
        bbox3d_np[:, :, 2] -= self.z_shift

        return pc_np, bbox3d_np

    def _resize_inputs(self, rgb_np, pc_np, mask_fg):
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)  # [3, H, W]
        rgb_tensor = F.interpolate(
            rgb_tensor.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pc_tensor = torch.from_numpy(pc_np)  # [3, H, W]
        pc_tensor = F.interpolate(
            pc_tensor.unsqueeze(0),
            size=self.target_size,
            mode="nearest",
        ).squeeze(0)

        mask_tensor = torch.from_numpy(mask_fg).unsqueeze(0)  # [1, H, W]
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=self.target_size,
            mode="nearest",
        ).squeeze(0)

        return (
            rgb_tensor.contiguous(),
            pc_tensor.contiguous(),
            mask_tensor.contiguous(),
        )

    def _build_targets(self, bbox3d_np):
        grid_h, grid_w = self.grid_size

        target_map = np.zeros((11, grid_h, grid_w), dtype=np.float32)
        reg_mask = np.zeros((grid_h, grid_w), dtype=np.float32)

        for obj_corners in bbox3d_np:
            params = corners_8_to_params_8(obj_corners)

            gx_f, gz_f = world_to_grid(
                center_x=params[0],
                center_z=params[2],
                grid_size=self.grid_size,
                range_x=self.range_x,
                range_z=self.range_z,
            )

            gx = int(np.floor(gx_f))
            gz = int(np.floor(gz_f))

            if not (0 <= gx < grid_w and 0 <= gz < grid_h):
                continue

            off_x = gx_f - gx
            off_z = gz_f - gz

            target_map[0] = draw_msra_gaussian(
                target_map[0],
                center=(gx, gz),
                sigma=self.gaussian_sigma,
            )

            if reg_mask[gz, gx] == 0:
                target_map[1:9, gz, gx] = params
                target_map[9, gz, gx] = off_x
                target_map[10, gz, gx] = off_z
                reg_mask[gz, gx] = 1.0

        return target_map, reg_mask

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.scenes[idx])

        rgb_np, pc_np, bbox3d_np, mask_fg = self._load_scene(scene_path)

        if self.training and np.random.rand() > 0.5:
            rgb_np, pc_np, bbox3d_np, mask_fg = self._apply_flip(
                rgb_np, pc_np, bbox3d_np, mask_fg
            )

        pc_np, bbox3d_np = self._align_coordinates(pc_np, bbox3d_np)
        rgb_tensor, pc_tensor, mask_tensor = self._resize_inputs(rgb_np, pc_np, mask_fg)
        target_map, reg_mask = self._build_targets(bbox3d_np)

        return {
            "rgb": rgb_tensor,
            "pc": pc_tensor,
            "target_map": torch.from_numpy(target_map).contiguous(),
            "reg_mask": torch.from_numpy(reg_mask).contiguous(),
            "mask": mask_tensor,  
        }