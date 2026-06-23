"""Stage 2: temporal sequence model over per-frame visual features."""

import torch
import torch.nn as nn

from ssr.models.encoder import CNNFeatureEncoder


class CNNLSTMLipReader(nn.Module):
    """MVP sequence model: per-frame CNN encoder + LSTM over time, producing
    per-timestep token logits for CTC-style decoding."""

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        vocab_size: int = 40,
    ):
        super().__init__()
        self.encoder = CNNFeatureEncoder(feature_dim=feature_dim)
        self.rnn = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W) -> token logits (B, T, vocab_size)
        b, t, c, h, w = frames.shape
        feats = self.encoder(frames.view(b * t, c, h, w)).view(b, t, -1)
        seq_out, _ = self.rnn(feats)
        return self.classifier(seq_out)
