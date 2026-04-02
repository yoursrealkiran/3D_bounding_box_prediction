import torch
import torch.nn as nn

class PCBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN to extract geometric features from XYZ channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)) # Ensure fixed output size
        )

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))