import torch
import torch.nn as nn
from torchvision import models

class ImageModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.efficientnet_b4(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = ImageModel()
    print(model)
