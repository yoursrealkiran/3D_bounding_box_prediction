import torch
import torch.nn as nn
from .backbone_rgb import RGBBackbone
from .backbone_pc import PCBackbone
from .fusion_node import BBoxHead

class Fusion3DDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize backbones
        self.rgb_stream = RGBBackbone(freeze_layers=True)
        self.pc_stream = PCBackbone(in_channels=3)
        
        # Fusion Head (512 from RGB + 512 from PC = 1024)
        self.head = BBoxHead(input_channels=1024)

    def forward(self, rgb, pc):
        """
        rgb: [Batch, 3, 480, 640]
        pc:  [Batch, 3, 480, 640] (Projected PC map)
        """
        # 1. Extract features from both modalities
        feat_rgb = self.rgb_stream(rgb) # [B, 512, 15, 20]
        feat_pc = self.pc_stream(pc)   # [B, 512, 15, 20]

        # 2. Concatenate along channel dimension
        fused_feat = torch.cat([feat_rgb, feat_pc], dim=1) # [B, 1024, 15, 20]
        
        # 3. Final Dense Prediction
        logits, bboxes = self.head(fused_feat)
        
        return logits, bboxes