import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from .utils import corners_8_to_params_8, draw_msra_gaussian

class ThreeDObjectDataset(Dataset):
    def __init__(self, root_dir, target_size=(480, 640), grid_size=(15, 20), training=True):
        self.root_dir = root_dir
        self.target_size = target_size
        self.grid_size = grid_size
        self.training = training
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.scenes[idx])
        
        # 1. Load Data
        rgb_np = np.array(Image.open(os.path.join(scene_path, "rgb.jpg"))).astype(np.float32) / 255.0
        pc_np = np.load(os.path.join(scene_path, "pc.npy")).astype(np.float32).copy()
        bbox3d_np = np.load(os.path.join(scene_path, "bbox3d.npy")).astype(np.float32).copy()

        # --- AUGMENTATION: Random Horizontal Flip ---
        do_flip = False
        if self.training and np.random.rand() > 0.5:
            do_flip = True
            rgb_np = np.flip(rgb_np, axis=1).copy()
            # Negate X values in 3D
            pc_np[0, :, :] = -pc_np[0, :, :] 
            # Mirror the projected grid spatially
            pc_np = np.flip(pc_np, axis=2).copy() 

        # --- COORDINATE ALIGNMENT ---
        # Align with Camera FOV (Flip X and Y)
        pc_np[0, :, :] = -pc_np[0, :, :] 
        pc_np[1, :, :] = -pc_np[1, :, :] 
        # Shift Z to center data (Fixed at 1.5m)
        pc_np[2, :, :] -= 1.5   
        # pc_np /= 2.0  <-- REMOVED: No more squashing

        # 2. Preprocess Tensors
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        rgb_tensor = F.interpolate(rgb_tensor.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
        
        pc_tensor = torch.from_numpy(pc_np)
        pc_tensor = F.interpolate(pc_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)

        # 3. Create Target Map (11 Channels)
        # [0: Heatmap, 1-8: Params (8D), 9-10: Sub-pixel Offsets]
        target_map = np.zeros((11, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        
        for obj_corners in bbox3d_np:
            if do_flip:
                obj_corners[:, 0] = -obj_corners[:, 0]

            # Match alignment flips
            obj_corners[:, 0] = -obj_corners[:, 0]
            obj_corners[:, 1] = -obj_corners[:, 1]
            
            params = corners_8_to_params_8(obj_corners)
            params[2] -= 1.5   # Match PC shift
            # params[:6] /= 2.0 <-- REMOVED

            # --- HIGH PRECISION OFFSET CALCULATION ---
            # Center X and Z normalized to [0, 1] relative to the grid extent
            norm_x = (params[0] + 0.5) * self.grid_size[1]
            norm_z = (params[2] + 0.5) * self.grid_size[0]
            
            grid_x, grid_z = int(norm_x), int(norm_z)
            
            # The residual is the target for the offset head
            off_x = norm_x - grid_x
            off_z = norm_z - grid_z

            if 0 <= grid_x < self.grid_size[1] and 0 <= grid_z < self.grid_size[0]:
                target_map[0] = draw_msra_gaussian(target_map[0], (grid_x, grid_z), sigma=1)
                target_map[1:9, grid_z, grid_x] = params
                target_map[9, grid_z, grid_x] = off_x
                target_map[10, grid_z, grid_x] = off_z

        return {
            "rgb": rgb_tensor.contiguous(),
            "pc": pc_tensor.contiguous(),
            "target_map": torch.from_numpy(target_map).contiguous()
        }