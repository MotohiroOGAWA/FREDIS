from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .reactant_graphormer import ReactantGraphormerEncoder


class PatternReactantEncoder(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, context_dim: int,
                 reactant_smarts: str, node_dim: int, graph_dim: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.encoder = ReactantGraphormerEncoder(
            reactant_smarts, atom_dim=atom_dim + context_dim, bond_dim=bond_dim,
            node_dim=node_dim, graph_dim=graph_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout)
        self.position_count = self.encoder.position_count
        self.output_dim = node_dim
        self.graph_dim = graph_dim

    def encode_with_nodes(self, graph: Data, context_by_atom: Tensor,
                          position: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if context_by_atom.shape[0] != graph.x.shape[0]:
            raise ValueError('one context embedding is required per reactant atom')
        data = graph.clone()
        data.x = torch.cat((graph.x, context_by_atom), dim=-1)
        if position is not None:
            data.smarts_position = position
        return self.encoder(data)

    def forward(self, graph: Data, context_by_atom: Tensor, position: Tensor | None = None) -> Tensor:
        return self.encode_with_nodes(graph, context_by_atom, position)[0]


class ReactantEncoderRegistry(nn.Module):
    """Own one SMARTS-sized parameter set per ReactionPattern."""
    def __init__(self, pattern_smarts: Mapping[int, str], atom_dim: int, bond_dim: int,
                 context_dim: int, node_dim: int, graph_dim: int, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.output_dim = node_dim
        self.encoders = nn.ModuleDict({str(int(pid)): PatternReactantEncoder(
            atom_dim, bond_dim, context_dim, smarts, node_dim, graph_dim,
            num_layers, num_heads, dropout) for pid, smarts in pattern_smarts.items()})

    def forward(self, pattern_id: int, graph: Data, context_by_atom: Tensor,
                position: Tensor | None = None) -> Tensor:
        try:
            encoder = self.encoders[str(int(pattern_id))]
        except KeyError as exc:
            raise KeyError(f'Unknown reaction pattern ID: {pattern_id}') from exc
        return encoder(graph, context_by_atom, position)
