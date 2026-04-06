import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(num_channels: int, max_groups: int = 16) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled = self.avg_pool(x).view(b, c)
        weights = self.fc(pooled).view(b, c, 1, 1)
        return x * weights.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.0):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            make_gn(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BBoxHead(nn.Module):
    """
    Dense multi-task prediction head with configurable output grid.

    Inputs:
        x: [B, C, Hf, Wf]  typically [B, 1024, 15, 20]

    Outputs:
        logits: [B, 1, Gh, Gw]   -> center heatmap
        bboxes: [B, 10, Gh, Gw]  -> bbox regression
        seg:    [B, 1, Gh, Gw]   -> foreground segmentation
    """
    def __init__(self, input_channels=1024, dropout=0.2, output_grid_size=(50, 70)):
        super().__init__()
        self.output_grid_size = tuple(output_grid_size)

        self.refine = nn.Sequential(
            ConvGNAct(input_channels, 512, kernel_size=3, dropout=0.0),
            ChannelAttention(512),
            SpatialAttention(),
        )

        self.shared_conv = nn.Sequential(
            ConvGNAct(512, 256, kernel_size=3, dropout=dropout),
            ConvGNAct(256, 256, kernel_size=3, dropout=0.0),
        )

        # Light refinement before upsampling
        self.pre_upsample = nn.Sequential(
            ConvGNAct(256, 256, kernel_size=3, dropout=0.0),
        )

        # Light refinement after upsampling
        self.post_upsample = nn.Sequential(
            ConvGNAct(256, 256, kernel_size=3, dropout=0.0),
            ConvGNAct(256, 256, kernel_size=3, dropout=0.0),
        )

        # Detection heads
        self.classifier = nn.Conv2d(256, 1, kernel_size=1)
        self.regressor = nn.Conv2d(256, 10, kernel_size=1)

        # segmentation head
        self.seg_head = nn.Conv2d(256, 1, kernel_size=1)

        self._init_heads()

    def _init_heads(self):
        # Sparse positives for heatmap
        nn.init.constant_(self.classifier.bias, -2.19)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.001)

        nn.init.normal_(self.regressor.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.regressor.bias, 0.0)

        # Segmentation head initialization
        nn.init.normal_(self.seg_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.seg_head.bias, 0.0)

    def forward(self, x):
        x = self.refine(x)
        x = self.shared_conv(x)
        x = self.pre_upsample(x)

        # Explicit resize to target grid size
        x = F.interpolate(
            x,
            size=self.output_grid_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.post_upsample(x)

        logits = self.classifier(x)
        bboxes = self.regressor(x)
        seg = self.seg_head(x)

        return logits, bboxes, seg