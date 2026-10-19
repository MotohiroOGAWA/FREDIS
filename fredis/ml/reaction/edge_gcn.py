from __future__ import annotations

import torch
from torch import Tensor, nn


class EdgeAwareGCN(nn.Module):
    """Small message-passing network that explicitly consumes bond features."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.input_projection = nn.Linear(node_dim, hidden_dim)
        self.messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        )
        self.output_dim = hidden_dim

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.input_projection(x)
        for message, update in zip(self.messages, self.updates):
            aggregate = h.new_zeros(h.shape)
            if edge_index.numel():
                source, target = edge_index
                value = message(torch.cat((h[source], h[target], edge_attr), dim=-1))
                aggregate.index_add_(0, target, value)
            h = h + update(torch.cat((h, aggregate), dim=-1))
        return h


def graph_mean_pool(
    x: Tensor, batch: Tensor | None, num_graphs: int | None = None
) -> Tensor:
    if batch is None:
        return x.mean(0, keepdim=True) if x.numel() else x.new_zeros((1, x.shape[-1]))
    count = int(batch.max()) + 1 if batch.numel() else int(num_graphs or 0)
    result = x.new_zeros((count, x.shape[-1]))
    counts = x.new_zeros(count)
    if batch.numel():
        result.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return result / counts.clamp_min(1).unsqueeze(-1)
