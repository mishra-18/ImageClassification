import torch.nn as nn
import torchvision

class Resnet50Flower102(nn.Module):
    def __init__(self, device, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.device = device
        
        if pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        else: 
            weights = None
            
        self.model = torchvision.models.resnet50(weights=weights)
        
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 102),
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x)