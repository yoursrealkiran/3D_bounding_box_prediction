import torch
import torch.nn as nn
import torchvision.models as models

class RGBBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the global average pool and fully connected layers
        # This leaves us with a feature map of shape (Batch, 512, H/32, W/32)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.feature_extractor(x)