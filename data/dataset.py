import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from .utils import corners_8_to_params_8, draw_msra_gaussian

class ThreeDObjectDataset(Dataset):
    def __init__(self, root_dir, target_size=(480, 640), grid_size=(15, 20)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.grid_size = grid_size
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Define physical boundaries (Meters)
        # Standardized range based on your typical sensor FOV
        self.range_x = [-10.0, 10.0]
        self.range_z = [0.0, 20.0]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.scenes[idx])
        
        # 1. Load Data
        rgb_np = np.array(Image.open(os.path.join(scene_path, "rgb.jpg"))).astype(np.float32) / 255.0
        # Use .copy() to ensure the arrays are writable for the flip operation
        pc_np = np.load(os.path.join(scene_path, "pc.npy")).astype(np.float32).copy()
        bbox3d_np = np.load(os.path.join(scene_path, "bbox3d.npy")).astype(np.float32).copy()

        # --- FIX: COORDINATE ALIGNMENT ---
        # Rotate Point Cloud 180° around Z-axis (Flip X and Y) 
        # to match Camera Forward orientation found in your visualizer
        pc_np[0, :, :] = -pc_np[0, :, :] # Flip X
        pc_np[1, :, :] = -pc_np[1, :, :] # Flip Y
        
        # Normalization (Matches the shift/scale used in your previous stable runs)
        pc_np[2, :, :] -= 1.5   # Shift Depth (Z)
        pc_np /= 2.0            # Global Scale

        # 2. Preprocess Tensors
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        rgb_tensor = F.interpolate(rgb_tensor.unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)
        
        pc_tensor = torch.from_numpy(pc_np)
        pc_tensor = F.interpolate(pc_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)

        # 3. Create Target Map (Heatmap + Regression)
        target_map = np.zeros((9, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        
        for obj_corners in bbox3d_np:
            # --- FIX: ALIGN GROUND TRUTH ---
            # Flip the box corners exactly like the point cloud
            obj_corners[:, 0] = -obj_corners[:, 0]
            obj_corners[:, 1] = -obj_corners[:, 1]
            
            # Convert corners to [x, y, z, w, l, h, sin, cos]
            params = corners_8_to_params_8(obj_corners)
            
            # Apply same normalization to box params as we did to Point Cloud
            # This makes the regression targets much easier for the model to learn
            params[2] -= 1.5   # Shift Z
            params[:6] /= 2.0  # Scale positions and dimensions (x, y, z, w, l, h)

            center_x, center_z = params[0], params[2]

            # Map World Coords (-1.0 to 1.0 roughly) to Grid Indices (0 to 19, 0 to 14)
            # Since we normalized by 2.0, the world center (0,0) maps to 0.5 in normalized space
            norm_x = (center_x + 0.5) 
            norm_z = (center_z + 0.5)
            
            grid_x = int((norm_x * self.grid_size[1]))
            grid_y = int((norm_z * self.grid_size[0]))

            # Bounds check to ensure object is within the 15x20 grid
            if 0 <= grid_x < self.grid_size[1] and 0 <= grid_y < self.grid_size[0]:
                # Channel 0: Objectness Heatmap (Gaussian peak at 1.0)
                target_map[0] = draw_msra_gaussian(target_map[0], (grid_x, grid_y), sigma=1)
                
                # Channels 1-8: 3D Regression parameters
                # We store these at the center cell only
                target_map[1:, grid_y, grid_x] = params

        return {
            "rgb": rgb_tensor.contiguous(),
            "pc": pc_tensor.contiguous(),
            "target_map": torch.from_numpy(target_map).contiguous()
        }