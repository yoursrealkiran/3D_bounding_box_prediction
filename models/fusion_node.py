import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BBoxHead(nn.Module):
    def __init__(self, input_channels=1024):
        super().__init__()
        
        # Fusion refinement
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ChannelAttention(512) # Learn to weigh RGB vs PC
        )
        
        self.shared_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.classifier = nn.Conv2d(256, 1, kernel_size=1)
        # 10 Channels: [x, y, z, w, l, h, sin, cos, off_x, off_z]
        self.regressor = nn.Conv2d(256, 10, kernel_size=1)

    def forward(self, x):
        x = self.refine(x)
        x = self.shared_conv(x)
        return self.classifier(x), self.regressor(x)