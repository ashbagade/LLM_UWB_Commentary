from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SimpleCricketLSTM(nn.Module):
    """
    Simple LSTM-based classifier for ViSig cricket umpire signals.

    Expects input of shape (batch_size, seq_len, feature_dim).
    Architecture:
      - LSTM (batch_first=True) -> take last hidden state
      - Optional dropout
      - 1–2 layer MLP head -> class logits
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        head_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, L, F), got {x.shape}")

        output, _ = self.lstm(x)                  
        mask = (x.abs().sum(dim=2) > 0).unsqueeze(-1)  
        output = output.masked_fill(~mask, float('-inf'))
        pooled = output.max(dim=1).values       
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        return logits


