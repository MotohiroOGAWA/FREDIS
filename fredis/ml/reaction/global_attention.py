"""Pattern-owned bond-aware molecular memory and SMARTS atom cross-attention."""
from __future__ import annotations
from collections import deque
import torch
from torch import nn
from .edge_gcn import EdgeAwareGCN


class GlobalStructureAttention(nn.Module):
    """Q: ordered reactant atoms; K/V: all atoms of that source molecule.

    Grouping by source keeps other molecules out of attention without padding.
    Returned channels retain the match and SMARTS atom axes.
    """
    def __init__(self, atom_dim, bond_dim, position_count, hidden_dim, num_layers=3,
                 num_heads=4, dropout=0.0):
        super().__init__()
        for name, value in [('hidden_dim', hidden_dim), ('num_layers', num_layers), ('num_heads', num_heads)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'global_{name} must be a positive integer')
        if hidden_dim % num_heads:
            raise ValueError('global_hidden_dim must be divisible by global_num_heads')
        self.memory = EdgeAwareGCN(atom_dim, bond_dim, hidden_dim, num_layers, dropout)
        self.query = nn.Linear(atom_dim + position_count, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.position_count = position_count
        # Distances to each queried atom distinguish remote bond positions.
        self.distance = nn.Embedding(66, hidden_dim)

    def forward(self, atoms, edge_index, edge_attr, source_batch, sites, source_ids):
        if sites.shape[1] != self.position_count:
            raise ValueError('Global attention SMARTS atom count mismatch')
        if source_batch.shape != (atoms.shape[0],):
            raise ValueError('Global attention requires source molecule ownership for every atom')
        if torch.any(source_batch[sites] != source_ids[:, None]):
            raise ValueError('Reactant atoms must belong to their source molecule')
        memory = self.memory(atoms, edge_index, edge_attr)
        positions = torch.eye(self.position_count, device=atoms.device, dtype=atoms.dtype)
        query = self.query(torch.cat((atoms[sites], positions.expand(sites.shape[0], -1, -1)), dim=-1))
        result = torch.zeros_like(query)
        adjacency = [[] for _ in range(atoms.shape[0])]
        for u, v in edge_index.detach().cpu().T.tolist():
            adjacency[u].append(v)
        distances = {}
        for anchor in sites.detach().cpu().unique().tolist():
            distance = {anchor: 0}; queue = deque([anchor])
            while queue:
                current = queue.popleft()
                if distance[current] >= 64:
                    continue
                for neighbor in adjacency[current]:
                    if neighbor not in distance:
                        distance[neighbor] = distance[current] + 1; queue.append(neighbor)
            distances[anchor] = distance
        for source in source_ids.unique():
            rows = (source_ids == source).nonzero().flatten()
            atom_ids = (source_batch == source).nonzero().flatten()
            anchors = sites[rows].detach().cpu().reshape(-1).tolist()
            indices = atom_ids.detach().cpu().tolist()
            relative = torch.tensor([[distances[anchor].get(atom, 65) for atom in indices]
                                     for anchor in anchors], device=atoms.device)
            values = memory[atom_ids].unsqueeze(0) + self.distance(relative)
            queries = query[rows].reshape(-1, 1, query.shape[-1])
            attended, _ = self.attention(queries, values, values, need_weights=False)
            result[rows] = self.norm(queries + attended).reshape_as(query[rows])
        return result
