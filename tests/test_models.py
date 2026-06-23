import torch

from ssr.models.encoder import CNNFeatureEncoder
from ssr.models.sequence_model import CNNLSTMLipReader


def test_cnn_feature_encoder_shape():
    encoder = CNNFeatureEncoder(feature_dim=128)
    x = torch.randn(4, 3, 96, 96)
    out = encoder(x)
    assert out.shape == (4, 128)


def test_cnn_lstm_lipreader_shape():
    model = CNNLSTMLipReader(feature_dim=64, hidden_size=32, vocab_size=28)
    frames = torch.randn(2, 10, 3, 96, 96)
    logits = model(frames)
    assert logits.shape == (2, 10, 28)
