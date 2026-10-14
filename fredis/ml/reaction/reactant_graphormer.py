"""Fixed-capacity Graphormer derived from one reactant SMARTS query."""
from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from rdkit import Chem

from ..common.layers.graphormer import GraphormerEncoder, PaddedGraphBatch


class ReactantGraphormerEncoder(GraphormerEncoder):
    """No bucketing/growth or attention-routing settings in the public API.

    atom_dim is fixed by input features; hidden_dim is the attention width.
    SMARTS query positions are concatenated as fixed one-hot input features.
    """

    def __init__(self, reactant_smarts: str, *, atom_dim: int, bond_dim: int,
                 hidden_dim: int | None = None, graph_dim: int, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.0,
                 node_dim: int | None = None, position_encoding: str = "one_hot"):
        query = Chem.MolFromSmarts(reactant_smarts)
        if query is None or query.GetNumAtoms() < 1:
            raise ValueError(f'Invalid reactant SMARTS: {reactant_smarts}')
        count = query.GetNumAtoms()
        if num_layers < 1:
            raise ValueError('num_layers must be positive')
        if hidden_dim is None:
            hidden_dim = node_dim
        elif node_dim is not None:
            raise ValueError('Specify hidden_dim, not both hidden_dim and node_dim')
        if hidden_dim is None:
            raise ValueError('hidden_dim is required')
        if position_encoding not in ('one_hot','learned'):
            raise ValueError('Unknown SMARTS position encoding')
        self.atom_dim = atom_dim
        self.position_encoding = position_encoding
        super().__init__(node_dim=atom_dim + (count if position_encoding == 'one_hot' else 0), hidden_dim=hidden_dim, edge_dim=bond_dim,
                         graph_dim=graph_dim, num_layers=num_layers, num_heads=num_heads,
                         max_degree=max(atom.GetDegree() for atom in query.GetAtoms()),
                         max_spatial_dist=max(1, count), max_edge_dist=max(1, count - 1),
                         dropout=dropout, start_cap=count)
        self.reactant_smarts = reactant_smarts
        self.position_count = count
        if position_encoding == 'learned':
            self.positions = torch.nn.Embedding(count, hidden_dim)
        edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in query.GetBonds()]
        edges += [(v, u) for u, v in edges]
        self.register_buffer('query_edge_index', torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(), persistent=False)

    def _padded_batches(self, data):
        batch = data if isinstance(data, Batch) else Batch.from_data_list([data])
        if torch.any(batch.ptr[1:] - batch.ptr[:-1] != self.position_count):
            raise ValueError('Each reactant graph must contain exactly the SMARTS atom count')
        padded = PaddedGraphBatch.from_graph_batch(batch, cap=self.position_count)
        return [padded], [torch.arange(batch.num_graphs, device=batch.x.device)]

    def _project_and_centrality_bucket(self, bucket):
        position = getattr(bucket.batch, 'smarts_position', None)
        if position is None:
            position = torch.arange(self.position_count, device=bucket.x.device).expand(bucket.B, -1)
        else:
            position = position.reshape(bucket.B, self.position_count)
        if torch.any((position < 0) | (position >= self.position_count)):
            raise ValueError('SMARTS position is out of range')
        if self.position_encoding == 'learned':
            return super()._project_and_centrality_bucket(bucket) + self.positions(position)
        one_hot = torch.nn.functional.one_hot(position, self.position_count).to(bucket.x.dtype)
        x = torch.cat((bucket.x, one_hot), dim=-1)
        return self.centrality(self.node_proj(x), edge_index=bucket.edge_index,
                               edge_mask=bucket.edge_mask, node_mask=bucket.node_mask)

    def encode_matches(self, atom_h: Tensor) -> tuple[Tensor, Tensor]:
        """Encode [matches, SMARTS atoms, features] on the SMARTS topology."""
        if atom_h.ndim != 3 or atom_h.shape[1] != self.position_count:
            raise ValueError('Reactant matches must have shape [matches, SMARTS atoms, features]')
        count = atom_h.shape[0]
        if count == 0:
            return atom_h.new_empty((0, self.position_count, self.hidden_dim)), atom_h.new_empty((0, self.graph_dim))
        offsets = torch.arange(count, device=atom_h.device) * self.position_count
        edges = (self.query_edge_index[None] + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)
        batch = Batch(x=atom_h.reshape(-1, atom_h.shape[-1]), edge_index=edges,
                      batch=torch.repeat_interleave(torch.arange(count, device=atom_h.device), self.position_count),
                      ptr=torch.arange(count + 1, device=atom_h.device) * self.position_count)
        batch.num_graphs = count
        nodes, graphs = self(batch)
        return nodes.reshape(count, self.position_count, self.hidden_dim), graphs
