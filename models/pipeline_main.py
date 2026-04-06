import torch
import torch.nn as nn

from .backbone_rgb import RGBBackbone
from .backbone_pc import PCBackbone
from .fusion_node import BBoxHead


class Fusion3DDetector(nn.Module):
    """
    Two-stream dense detector for RGB + organized point cloud.

    Outputs:
        logits: [B, 1, Gh, Gw]   -> center heatmap
        bboxes: [B, 10, Gh, Gw]  -> bbox regression
        seg:    [B, 1, Gh, Gw]   -> foreground segmentation
    """
    def __init__(
        self,
        freeze_rgb_layers=True,
        head_dropout=0.2,
        output_grid_size=(50, 70),
    ):
        super().__init__()

        self.rgb_stream = RGBBackbone(freeze_layers=freeze_rgb_layers)
        self.pc_stream = PCBackbone(in_channels=3)

        self.head = BBoxHead(
            input_channels=1024,
            dropout=head_dropout,
            output_grid_size=output_grid_size,
        )

    def forward(self, rgb, pc):
        # Extract features
        feat_rgb = self.rgb_stream(rgb)
        feat_pc = self.pc_stream(pc)

        # Safety check for feature map size mismatch before fusion
        if feat_rgb.shape[-2:] != feat_pc.shape[-2:]:
            raise RuntimeError(
                f"Feature map size mismatch: RGB {feat_rgb.shape}, PC {feat_pc.shape}"
            )

        # Fuse features
        fused_feat = torch.cat([feat_rgb, feat_pc], dim=1)

        # Head outputs
        logits, bboxes, seg = self.head(fused_feat)

        return logits, bboxes, seg