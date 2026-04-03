import torch
import torch.nn as nn

class BBoxHead(nn.Module):
    def __init__(self, input_channels=1024):
        super().__init__()
        
        # The shared conv now uses kernel 3 for spatial context
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2), 
            nn.Conv2d(512, 256, kernel_size=1), # Squeeze back to 256
            nn.ReLU(inplace=True)
        )
        
        # Heatmap prediction (Objectness)
        self.classifier = nn.Conv2d(256, 1, kernel_size=1)
        
        # Box parameter regression [x, y, z, w, l, h, sin, cos]
        self.regressor = nn.Conv2d(256, 8, kernel_size=1)

    def forward(self, x):
        x = self.shared_conv(x)
        logits = self.classifier(x)
        bboxes = self.regressor(x)
        return logits, bboxes