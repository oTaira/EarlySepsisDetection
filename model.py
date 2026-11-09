"""A small causal 1D CNN.

Three kernel-8 convolutions look only backward, so the logit at an hour uses
the current hour and the previous 21 hours. That is the proposal's trailing
window, kept to one small stack.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class SimpleCausalCNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, kernel_size: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(n_layers):
            layers.append(CausalConv1d(channels, hidden, kernel_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            channels = hidden
        self.stack = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)
        self.receptive_field = 1 + n_layers * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stack(x)).squeeze(1)
