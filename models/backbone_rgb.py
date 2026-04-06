import torch.nn as nn
import torchvision.models as models


def make_gn(num_channels: int, max_groups: int = 16) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class RGBBackbone(nn.Module):
    """
    ResNet18 backbone with pretrained weights.
    """

    def __init__(self, freeze_layers=True):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  #  [B, 512, 15, 20]

        if freeze_layers:
            # Freezing early and mid-level layers. Keeping later layers trainable.
            for i, child in enumerate(self.feature_extractor):
                if i < 7:
                    for param in child.parameters():
                        param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            make_gn(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.proj(feat)
        return feat