from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SimpleCricketCNN(nn.Module):
    """
    Simple 1D CNN classifier for ViSig cricket umpire signals.

    Expects input of shape (batch_size, seq_len, feature_dim).

    Architecture:
      - Permute to (batch_size, feature_dim, seq_len)
      - Stack of Conv1d + ReLU (+ optional dropout)
      - Global max-pool over time
      - Linear layer to num_classes
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_channels: int = 128,
        num_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd so padding keeps length")

        layers = []
        in_channels = input_dim
        padding = kernel_size // 2

        layers.append(nn.Conv1d(in_channels, num_channels, kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 1):
            layers.append(
                nn.Conv1d(num_channels, num_channels, kernel_size, padding=padding)
            )
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.conv_stack = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, L, F), got {x.shape}")

        x = x.permute(0, 2, 1)
        x = self.conv_stack(x)               
        x = x.max(dim=-1).values             
        logits = self.classifier(x)          
        return logits

