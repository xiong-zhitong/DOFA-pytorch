"""Base classes."""

import torch.nn as nn
from torch import Tensor


activation_fns = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
}


class LinearHead(nn.Module):
    """Linear Classification Head that is attached on top of feature extractors."""

    def __init__(self, in_features: int, num_classes: int, activation: str | None = None) -> None:
        """
        Args:
            in_features: Number of input features.
            num_classes: Number of output classes.
            activation: Activation function to apply on top of encoded input feature
        """
        super().__init__()
        self.head = nn.Sequential(
            activation_fns[activation]() if activation else nn.Identity(),
            nn.BatchNorm1d(in_features, affine=False, eps=1e-6),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of linear head.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.head(x)
