import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), # FIX: inplace=True saves memory on 4GB GPUs
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
        # We calculate the residual path and the identity path
        out = self.conv(x)
        res = self.shortcut(x)
        
        # FIX: Addition followed by inplace ReLU is the standard ResNet pattern
        out += res 
        return torch.relu_(out) # inplace ReLU

class PCBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Layer 1: [B, 3, 480, 640] -> [B, 64, 240, 320]
        self.layer1 = ResidualBlock(in_channels, 64, stride=2)
        # Layer 2: [B, 64, 240, 320] -> [B, 128, 120, 160]
        self.layer2 = ResidualBlock(64, 128, stride=2)
        # Layer 3: [B, 128, 120, 160] -> [B, 256, 60, 80]
        self.layer3 = ResidualBlock(128, 256, stride=2)
        # Layer 4: [B, 256, 60, 80] -> [B, 512, 30, 40]
        self.layer4 = ResidualBlock(256, 512, stride=2)
        # Layer 5: [B, 512, 30, 40] -> [B, 512, 15, 20]
        self.layer5 = ResidualBlock(512, 512, stride=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x