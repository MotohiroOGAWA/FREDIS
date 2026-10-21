"""Order-invariant reaction-set encoding, including the empty BOS state."""

from __future__ import annotations
import torch
from torch import Tensor, nn
from ....common.layers.set_transformer import SAB


class ReactionSequenceEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError(
                "State encoder requires positive layers and dropout in [0,1)"
            )
        self.bos = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            SAB(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(
        self,
        reaction_h_pool: Tensor,
        state_reaction_index: Tensor,
        state_sample_index: Tensor,
    ) -> Tensor:
        if state_reaction_index.shape[0] == 0:
            return reaction_h_pool.new_empty((0, self.bos.shape[-1]))
        if reaction_h_pool.shape[1] == 0:
            reaction_h_pool = reaction_h_pool.new_zeros(
                (reaction_h_pool.shape[0], 1, self.bos.shape[-1])
            )
        tokens = reaction_h_pool[
            state_sample_index[:, None], state_reaction_index.clamp_min(0)
        ]
        padding = state_reaction_index < 0
        tokens = tokens.masked_fill(padding[:, :, None], 0)
        tokens = torch.cat((self.bos.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        padding = torch.cat((padding.new_zeros((padding.shape[0], 1)), padding), dim=1)
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=padding)
        return tokens[:, 0]
