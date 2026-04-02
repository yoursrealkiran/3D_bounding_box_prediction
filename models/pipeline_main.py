import torch
import torch.nn as nn
from .backbone_rgb import RGBBackbone
from .backbone_pc import PCBackbone
from .fusion_node import BBoxHead

class Fusion3DDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_stream = RGBBackbone()
        self.pc_stream = PCBackbone()
        
        # Global pooling to collapse spatial dimensions to vectors
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = BBoxHead(input_dim=1024)

    def forward(self, rgb_img, pc_map):
        # 1. Process streams
        feat_rgb = self.rgb_stream(rgb_img)
        feat_pc = self.pc_stream(pc_map)
        
        # 2. Flatten/Pool
        feat_rgb = self.pool(feat_rgb).view(feat_rgb.size(0), -1)
        feat_pc = self.pool(feat_pc).view(feat_pc.size(0), -1)
        
        # 3. Concatenate (Fusion)
        fused_feat = torch.cat([feat_rgb, feat_pc], dim=1)
        
        # 4. Predict
        logits, bbox3d = self.head(fused_feat)
        
        return logits, bbox3d