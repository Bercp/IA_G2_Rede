import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.body(x))


class PolicyNet(nn.Module):
    def __init__(self, channels=32, blocks=2):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks}
        self.stem = nn.Sequential(nn.Conv2d(19, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 320, 1)

    def forward(self, x):
        x = self.head(self.blocks(self.stem(x)))
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], 20480)
