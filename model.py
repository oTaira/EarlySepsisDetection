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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class SWInSiLU(nn.Module):
    """SiLU wrapped by the SWIn ripple.

    Ran et al., Neural Networks, 28 October 2026. The published default is
    amplitude 0.1 and frequency 1: silu(x) + (a / w) * sin(w * silu(x)).
    """

    def __init__(self, amplitude: float = 0.1, omega: float = 1.0):
        super().__init__()
        self.amplitude = float(amplitude)
        self.omega = float(omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.silu(x)
        return base + (self.amplitude / self.omega) * torch.sin(self.omega * base)


class SimpleCausalCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 32,
        kernel_size: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: nn.Module | None = None,
        dilation: int = 1,
    ):
        super().__init__()
        layers = []
        channels = in_channels
        act = nn.ReLU() if activation is None else activation
        for _ in range(n_layers):
            layers.append(CausalConv1d(channels, hidden, kernel_size, dilation=dilation))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            channels = hidden
        self.stack = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)
        self.dilation = dilation
        self.receptive_field = 1 + n_layers * (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stack(x)).squeeze(1)


class CausalConv2d(nn.Module):
    """Convolution over variables and time. Time is padded only on the left."""

    def __init__(self, in_channels: int, out_channels: int, kernel_h: int, kernel_w: int):
        super().__init__()
        if kernel_h % 2 == 0:
            raise ValueError("feature kernel must be odd")
        self.pad_h = kernel_h // 2
        self.pad_w = kernel_w - 1
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_w, 0, self.pad_h, self.pad_h))
        return self.conv(x)


class FeatureTimeCNN(nn.Module):
    """2D CNN on the 34 variables by time.

    Each variable carries its value, observation mask, and time since
    measurement. A short feature kernel only mixes neighboring rows of that
    list, so distant variables meet only after several layers. A tall kernel
    mixes most of the panel at once.
    """

    def __init__(
        self,
        hidden: int = 32,
        kernel_h: int = 3,
        kernel_w: int = 5,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: nn.Module | None = None,
        n_variables: int = 34,
        static: int = 7,
    ):
        super().__init__()
        act = nn.ReLU() if activation is None else activation
        layers = []
        channels = 3
        for _ in range(n_layers):
            layers.append(CausalConv2d(channels, hidden, kernel_h, kernel_w))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            channels = hidden
        self.stack = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden + static, 1, kernel_size=1)
        self.time_receptive_field = 1 + n_layers * (kernel_w - 1)
        self.feature_receptive_field = 1 + n_layers * (kernel_h - 1)
        self.n_variables = n_variables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x[:, : self.n_variables]
        observed = x[:, self.n_variables : 2 * self.n_variables]
        since = x[:, 2 * self.n_variables : 3 * self.n_variables]
        context = x[:, 3 * self.n_variables :]
        grid = torch.stack((values, observed, since), dim=1)
        hidden = self.stack(grid).mean(dim=2)
        return self.head(torch.cat((hidden, context), dim=1)).squeeze(1)


class CausalAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float, ffn_ratio: int = 2):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(hidden)
        self.norm_2 = nn.LayerNorm(hidden)
        self.expand = nn.Linear(hidden, hidden * ffn_ratio)
        self.project = nn.Linear(hidden * ffn_ratio, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x.transpose(1, 2)
        length = series.shape[1]
        future = torch.triu(torch.ones(length, length, device=series.device, dtype=torch.bool), diagonal=1)
        causal = torch.zeros(length, length, device=series.device)
        causal = causal.masked_fill(future, float("-inf"))
        updated, _ = self.attention(series, series, series, attn_mask=causal, need_weights=False)
        series = self.norm_1(series + self.dropout(updated))
        mixed = self.project(self.dropout(F.gelu(self.expand(series))))
        series = self.norm_2(series + mixed)
        return series.transpose(1, 2)


class CausalAttentionNet(nn.Module):
    """A causal convolution mixes every feature, then attention mixes any past hour."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        activation: nn.Module | None = None,
        stem_kernel: int = 5,
        ffn_ratio: int = 2,
    ):
        super().__init__()
        self.stem = CausalConv1d(in_channels, hidden, kernel_size=stem_kernel)
        self.activation = nn.ReLU() if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            CausalAttentionBlock(hidden, heads, dropout, ffn_ratio) for _ in range(n_layers)
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)
        self.heads = heads
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.stem(x)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden).squeeze(1)


class CausalLSTM(nn.Module):
    """Unidirectional LSTM. The logit at an hour uses only that hour and the past."""

    def __init__(self, in_features: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collate supplies (batch, channels, time), matching the CNN.
        series = x.transpose(1, 2)
        output, _ = self.lstm(series)
        return self.head(self.dropout(output)).squeeze(-1)


class DenseGAT(nn.Module):
    """Graph attention for one shared 34-node graph, applied to every hour at once."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int,
        dropout: float,
        concat: bool,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        width = heads * out_features if concat else out_features
        self.bias = nn.Parameter(torch.zeros(width))
        self.score_dst = nn.Parameter(torch.empty(heads, out_features))
        self.score_src = nn.Parameter(torch.empty(heads, out_features))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.score_dst)
        nn.init.xavier_uniform_(self.score_src)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(nodes).view(*nodes.shape[:-1], self.heads, self.out_features)
        dst = (hidden * self.score_dst).sum(-1)
        src = (hidden * self.score_src).sum(-1)
        logits = dst.unsqueeze(-2) + src.unsqueeze(-3)
        logits = F.leaky_relu(logits, self.negative_slope)
        mask = adjacency.view(*([1] * (logits.ndim - 3)), 34, 34, 1)
        weights = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-2)
        weights = F.dropout(weights, self.dropout, training=self.training)
        mixed = torch.einsum("...ijh,...jhd->...ihd", weights, hidden)
        mixed = mixed.reshape(*mixed.shape[:-2], -1) if self.concat else mixed.mean(dim=-2)
        return mixed + self.bias


class PhysioGAT(nn.Module):
    """Graph attention over the 34 physiological variables at one hour.

    Node features are the current value, the observation mask, and the time
    since the last measurement. A static vector is concatenated after pooling.
    """

    def __init__(
        self,
        hidden: int = 32,
        heads: int = 4,
        dropout: float = 0.4,
        static: int = 7,
        n_layers: int = 2,
        activation: nn.Module | None = None,
        negative_slope: float = 0.2,
        pool: str = "mean",
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("PhysioGAT needs at least two layers")
        if pool not in {"mean", "max"}:
            raise ValueError(pool)
        layers = []
        in_features = 3
        for _ in range(n_layers - 1):
            layers.append(DenseGAT(in_features, hidden, heads, dropout, True, negative_slope))
            in_features = hidden * heads
        layers.append(DenseGAT(in_features, hidden, 1, dropout, False, negative_slope))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ELU() if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + static, 1)
        self.hidden = hidden
        self.heads = heads
        self.n_layers = n_layers
        self.pool = pool

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :34]
        observed = x[:, 34:68]
        since = x[:, 68:102]
        context = x[:, 102:].transpose(1, 2)
        nodes = torch.stack((values, observed, since), dim=-1).permute(0, 2, 1, 3)
        adjacency = x.new_zeros(34, 34, dtype=torch.bool)
        adjacency.fill_diagonal_(True)
        if edge_index.numel():
            edges = edge_index.to(x.device)
            adjacency[edges[0], edges[1]] = True
        hidden = nodes
        for layer in self.layers[:-1]:
            hidden = self.dropout(self.activation(layer(hidden, adjacency)))
        hidden = self.layers[-1](hidden, adjacency)
        pooled = hidden.mean(dim=2) if self.pool == "mean" else hidden.max(dim=2).values
        return self.head(torch.cat((pooled, context), dim=-1)).squeeze(-1)
