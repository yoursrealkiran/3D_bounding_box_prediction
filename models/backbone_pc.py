import torch
import torch.nn as nn


def make_gn(num_channels: int, max_groups: int = 16) -> nn.GroupNorm:
    """
    Create a GroupNorm layer with a valid number of groups for the channel count.
    """
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation if kernel_size == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            make_gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm instead of BatchNorm.
    Better suited for small batch sizes.
    """

    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        padding = dilation

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = make_gn(out_ch)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = make_gn(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                make_gn(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.out_act(out)
        return out


class PCBackbone(nn.Module):
    """
    Backbone for organized point cloud maps (XYZ image).
    """

    def __init__(self, in_channels=3):
        super().__init__()
        if in_channels != 3:
            raise ValueError("PCBackbone expects 3 input channels: X, Y, Z")

        # Internally, one validity channel is appended - 4 channels in total.
        stem_in = 4

        self.stem = nn.Sequential(
            ConvGNAct(stem_in, 32, kernel_size=3, stride=1),
            ConvGNAct(32, 32, kernel_size=3, stride=1),
        )

        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2, dilation=1)
        self.layer5 = ResidualBlock(512, 512, stride=2, dilation=2)

    def _build_validity_mask(self, x):
        """
        x: [B, 3, H, W]
        Returns:
            valid: [B, 1, H, W] float mask
        """
        finite_mask = torch.isfinite(x).all(dim=1, keepdim=True)

        # Treating all-zero XYZ pixels as invalid background if present.
        nonzero_mask = (x.abs().sum(dim=1, keepdim=True) > 1e-8)

        valid = (finite_mask & nonzero_mask).float()
        return valid

    def forward(self, x):
        # cleaning invalid numeric values
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        valid = self._build_validity_mask(x)
        x = torch.cat([x, valid], dim=1)

        x = self.stem(x)
        x = self.layer1(x)   
        x = self.layer2(x)   
        x = self.layer3(x)   
        x = self.layer4(x)   
        x = self.layer5(x)   
        return x