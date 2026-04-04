import torch
import torch.nn as nn
import torchvision.models as models

class RGBBackbone(nn.Module):
    def __init__(self, freeze_layers=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_layers:
            # Freeze up to layer 3 of ResNet (Index 7)
            # This forces the model to use existing edge/texture detectors
            # while only training the high-level semantic fusion layers.
            for i, child in enumerate(self.feature_extractor):
                if i < 7: 
                    for param in child.parameters():
                        param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)