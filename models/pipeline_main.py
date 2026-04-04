import torch
import torch.nn as nn
from .backbone_rgb import RGBBackbone
from .backbone_pc import PCBackbone
from .fusion_node import BBoxHead

class Fusion3DDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_stream = RGBBackbone(freeze_layers=True)
        self.pc_stream = PCBackbone(in_channels=3)
        
        # 512 (RGB) + 512 (PC) = 1024
        self.head = BBoxHead(input_channels=1024)

    def forward(self, rgb, pc):
        # Extract features
        feat_rgb = self.rgb_stream(rgb) 
        feat_pc = self.pc_stream(pc)   

        # Concatenate Features
        fused_feat = torch.cat([feat_rgb, feat_pc], dim=1) 
        
        # Dense Grid Prediction
        logits, bboxes = self.head(fused_feat)
        
        return logits, bboxes