import torch
import torch.nn as nn

class BBoxHead(nn.Module):
    def __init__(self, input_dim=1024): # 512 (RGB) + 512 (PC)
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Output: 7 parameters [x, y, z, w, h, l, yaw]
        self.regressor = nn.Linear(256, 7)
        
        # Output: Class probability (e.g., is there an object?)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc(x)
        box_params = self.regressor(x)
        logits = self.classifier(x)
        return logits, box_params