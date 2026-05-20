import torch

from src.train_ae import FeedForwardAutoencoder, TabularTransformerAutoencoder


def test_autoencoder_models_keep_input_shape() -> None:
    """两个自编码器都应该输出和输入相同的形状。"""

    batch = torch.randn(4, 10)
    vanilla_model = FeedForwardAutoencoder(feature_count=10)
    transformer_model = TabularTransformerAutoencoder(feature_count=10, d_model=16, nhead=4, num_layers=1)

    assert vanilla_model(batch).shape == batch.shape
    assert transformer_model(batch).shape == batch.shape

