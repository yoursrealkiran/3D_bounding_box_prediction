import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from .utils import corners_8_to_params_7

class ThreeDObjectDataset(Dataset):
    def __init__(self, root_dir, target_size=(480, 640), transform=None):
        """
        Args:
            root_dir: Path to the dataset
            target_size: (Height, Width) to resize all inputs to for batching
            transform: Optional transforms
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.transform = transform

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.scenes[idx])
        
        # 1. Load data and force into fresh memory buffers
        rgb_np = np.array(Image.open(os.path.join(scene_path, "rgb.jpg"))).astype(np.float32) / 255.0
        pc_np = np.load(os.path.join(scene_path, "pc.npy")).astype(np.float32).copy()
        bbox3d_np = np.load(os.path.join(scene_path, "bbox3d.npy")).astype(np.float32).copy()
        
        # 2. Convert to Tensors
        # Shape: [3, H, W]
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        pc_tensor = torch.from_numpy(pc_np)
        
        # 3. Resize to target_size to allow batch stacking
        # interpolate expects [Batch, Channel, H, W], so we unsqueeze and squeeze
        rgb_tensor = F.interpolate(
            rgb_tensor.unsqueeze(0), 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        pc_tensor = F.interpolate(
            pc_tensor.unsqueeze(0), 
            size=self.target_size, 
            mode='nearest'  # CRITICAL: preserves coordinate integrity
        ).squeeze(0)
        
        # 4. Extract 7D parameters from ground truth
        # Note: 3D coordinates (meters) do not change when we resize the image map
        target_7d = corners_8_to_params_7(bbox3d_np[0])
        target_tensor = torch.from_numpy(target_7d).float()
        
        presence = torch.tensor([1.0], dtype=torch.float32)

        return {
            "rgb": rgb_tensor.contiguous(),
            "pc": pc_tensor.contiguous(),
            "gt_box": target_tensor.contiguous(),
            "presence": presence
        }