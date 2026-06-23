"""Stage 2: visual feature encoder for mouth ROI frames."""

import torch
import torch.nn as nn


class CNNFeatureEncoder(nn.Module):
    """Per-frame CNN encoder producing a feature vector from a mouth ROI."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, feature_dim)
        feats = self.net(x).flatten(1)
        return self.proj(feats)
