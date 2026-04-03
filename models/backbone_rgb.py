import torch
import torch.nn as nn
import torchvision.models as models

class RGBBackbone(nn.Module):
    def __init__(self, freeze_layers=True):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Output (480, 640) -> (15, 20) with 512 channels
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_layers:
            # We only freeze the very early stem (Conv1, BN, Layer1)
            # This allows the deeper semantic layers to adapt to your 3D scenes
            for i, child in enumerate(self.feature_extractor):
                if i < 5: 
                    for param in child.parameters():
                        param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)