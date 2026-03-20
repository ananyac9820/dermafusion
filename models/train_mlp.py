import torch
import torch.nn as nn

class MetadataModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = MetadataModel(input_size=10)
    print(model)
