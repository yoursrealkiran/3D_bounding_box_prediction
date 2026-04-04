import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        # Use dilation to increase receptive field for sparse point clouds
        padding = dilation 
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.conv(x)
        res = self.shortcut(x)
        out += res 
        return torch.relu_(out)

class PCBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Stages 1-3: Standard Downsampling
        self.layer1 = ResidualBlock(in_channels, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        # Stages 4-5: Use Dilation to capture wider geometric context without losing resolution
        self.layer4 = ResidualBlock(256, 512, stride=2, dilation=1)
        self.layer5 = ResidualBlock(512, 512, stride=2, dilation=2) 

    def forward(self, x):
        return self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))