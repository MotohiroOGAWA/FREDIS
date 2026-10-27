"""Pattern-specific one-hop molecular memory and structural cross attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from rdkit import Chem

from ....common.layers.graphormer import (
    CentralityEncoding, SpatialEncoding, EdgeEncoding, PaddedGraphBatch,
)


def build_molecular_batch(atoms, edge_index, edge_attr=None, graph_ptr=None):
    """Pack raw source tensors once, preserving disconnected atoms in each molecule."""
    if graph_ptr is None:
        graph_ptr = torch.tensor([0, atoms.shape[0]], device=atoms.device)
    if graph_ptr.ndim != 1 or graph_ptr.numel() < 2:
        raise ValueError("source_graph_ptr must contain at least two offsets")
    if int(graph_ptr[0]) != 0 or int(graph_ptr[-1]) != atoms.shape[0] or bool((graph_ptr.diff() <= 0).any()):
        raise ValueError("source_graph_ptr must partition all source atoms into nonempty molecules")
    owners = torch.repeat_interleave(torch.arange(graph_ptr.numel() - 1, device=atoms.device), graph_ptr.diff())
    if edge_index.numel() and bool((owners[edge_index[0]] != owners[edge_index[1]]).any()):
        raise ValueError("Source edges must not connect different molecules")
    graph = Batch(x=atoms, edge_index=edge_index, edge_attr=edge_attr, ptr=graph_ptr, batch=owners)
    graph.num_graphs = graph_ptr.numel() - 1
    return PaddedGraphBatch.from_graph_batch(graph, cap=int(graph_ptr.diff().max()))


def rectangular_structural_bias(molecular, query_atoms, spatial, edge_encoding, max_distance):
    """Compute only unique reactant-to-all-atom rows using sparse bond propagation.

    Molecular chemistry stores both directions of each bond. Walk counts and
    edge-value sums propagate over bond lists, without an all-atom square matrix.
    Each layer reaches only previously unvisited atoms, so all tied shortest
    paths contribute and longer walks never contaminate a path average.
    """
    raw = molecular.batch
    owners = raw.batch[query_atoms]
    centers = query_atoms - raw.ptr[owners]
    count, capacity = query_atoms.numel(), molecular.cap
    heads = spatial.dist_emb.embedding_dim
    distance = torch.full((count, capacity), max_distance + 1, dtype=torch.long, device=query_atoms.device)
    rows = torch.arange(count, device=query_atoms.device)
    distance[rows, centers] = 0
    counts = torch.zeros((count, capacity), dtype=torch.float32, device=query_atoms.device)
    counts[rows, centers] = 1
    visited = counts.bool()
    source = molecular.edge_index[owners, 0]
    target = molecular.edge_index[owners, 1]
    mask = molecular.edge_mask[owners]
    target_heads = target[..., None].expand(-1, -1, heads)
    sums = counts.new_zeros((count, capacity, heads))
    edge_bias = torch.zeros_like(sums)
    for depth in range(1, max_distance + 1):
        edge_counts = counts.gather(1, source) * mask
        next_counts = torch.zeros_like(counts).scatter_add(1, target, edge_counts)
        nearest = (next_counts > 0) & ~visited
        distance = torch.where(nearest, depth, distance)
        visited = visited | nearest
        if edge_encoding is not None and depth <= edge_encoding.max_dist:
            # Keep shortest-path counts and sums stable under mixed precision.
            with torch.autocast(device_type=query_atoms.device.type, enabled=False):
                values = molecular.edge_attr.float() @ edge_encoding.w_pos[depth - 1].float()
                if edge_encoding.undirected:
                    # Preserve EdgeEncoding's mirroring convention for bidirectional bonds.
                    values = values * 2
                prefix = sums.gather(1, source[..., None].expand(-1, -1, heads))
                messages = (prefix + edge_counts[..., None] * values[owners]) * mask[..., None]
                next_sums = torch.zeros_like(sums).scatter_add(1, target_heads, messages)
                average = next_sums / next_counts.clamp_min(1)[..., None] / depth
                edge_bias = edge_bias + average * nearest[..., None]
                sums = next_sums * nearest[..., None]
        counts = next_counts * nearest
    # SpatialEncoding supports rectangular query/key axes through a singleton query axis.
    return spatial(distance[:, None, :]).squeeze(-2) + edge_bias.transpose(1, 2)


class OneHopAtomGNN(nn.Module):
    """One simultaneous message-passing step using raw neighbors and bond features."""

    def __init__(self, atom_dim, bond_dim, hidden_dim):
        super().__init__()
        self.bond_dim = bond_dim
        self.self_projection = nn.Linear(atom_dim, hidden_dim)
        self.message_projection = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atoms, edge_index, edge_attr=None):
        if self.bond_dim and (edge_attr is None or edge_attr.shape != (edge_index.shape[1], self.bond_dim)):
            raise ValueError("The one-hop GNN requires the configured bond features")
        src, dst = edge_index
        inputs = atoms[src]
        if self.bond_dim:
            inputs = torch.cat((inputs, edge_attr), dim=-1)
        messages = self.message_projection(inputs)
        pooled = messages.new_zeros((atoms.shape[0], messages.shape[-1])).index_add(0, dst, messages)
        counts = torch.bincount(dst, minlength=atoms.shape[0]).clamp_min(1).unsqueeze(-1)
        return self.norm(F.gelu(self.self_projection(atoms) + pooled / counts))


class ReactionCrossAttentionBlock(nn.Module):
    """Cross attention, residual/normalization, then FFN and residual/normalization."""

    def __init__(self, hidden_dim, num_heads, ffn_mult, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.last_attention = None
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, memory, bias, molecule, valid, *, need_weights=False):
        matches, positions, dim = query.shape
        batch, capacity, _ = memory.shape
        # Project each molecule once, then process every real match in this batch.
        keys = self.k_proj(memory).reshape(batch, capacity, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(memory).reshape(batch, capacity, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.q_proj(query).reshape(matches, positions, self.num_heads, self.head_dim).transpose(1, 2)
        local_bias = bias.to(queries.dtype).masked_fill(~valid[molecule, None, None, :], float('-inf'))
        self.last_attention = None
        if need_weights:
            with torch.no_grad():
                scores = queries.detach() @ keys[molecule].detach().transpose(-1, -2)
                self.last_attention = (scores * (self.head_dim ** -0.5) + local_bias.detach()).softmax(-1)
        attended = F.scaled_dot_product_attention(
            queries, keys[molecule], values[molecule], attn_mask=local_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(matches, positions, dim)
        update = self.out_proj(attended).to(query.dtype)
        query = self.attention_norm(query + self.attention_dropout(update))
        return self.ffn_norm(query + self.ffn_dropout(self.ffn(query)))


class ReactionAtomEncoder(nn.Module):
    """Independent raw-atom encoder for one reaction pattern, without Graphormer.

    Query inputs concatenate all MolGraphBuilder atom features and SMARTS-position
    one-hot labels. A single molecular GNN supplies fixed keys/values for every cross
    attention layer. Centrality is added to both sides; spatial and edge-path
    encodings bias each reactant-to-all-atom attention row.
    """

    def __init__(
        self, smarts, *, atom_dim, bond_dim, hidden_dim, output_dim,
        num_heads=4, num_layers=2, dropout=0.0, ffn_mult=4, max_degree=4,
        max_spatial_dist=4, max_edge_dist=4,
    ):
        super().__init__()
        query = Chem.MolFromSmarts(smarts)
        if query is None or query.GetNumAtoms() < 1:
            raise ValueError("A reaction pattern must contain a valid, nonempty reactant SMARTS")
        for name, value in dict(hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads,
                                num_layers=num_layers, ffn_mult=ffn_mult, max_degree=max_degree,
                                max_spatial_dist=max_spatial_dist, max_edge_dist=max_edge_dist).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("Reaction hidden_dim must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        self.reactant_smarts = smarts
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.position_count = query.GetNumAtoms()
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_spatial_dist = max_spatial_dist
        self.max_edge_dist = max_edge_dist
        self.reactant_projection = nn.Linear(atom_dim + self.position_count, hidden_dim)
        self.atom_gnn = OneHopAtomGNN(atom_dim, bond_dim, hidden_dim)
        self.centrality = CentralityEncoding(hidden_dim, max_degree)
        self.spatial = SpatialEncoding(num_heads, max_spatial_dist)
        self.edge_encoding = EdgeEncoding(bond_dim, num_heads, max_edge_dist) if bond_dim else None
        self.layers = nn.ModuleList(
            ReactionCrossAttentionBlock(hidden_dim, num_heads, ffn_mult, dropout) for _ in range(num_layers)
        )
        self.readout = nn.Linear(self.position_count * hidden_dim, output_dim)

    def forward(self, molecular, sites):
        if sites.ndim != 2 or sites.shape[1] != self.position_count:
            raise ValueError("Reaction sites must follow the reactant SMARTS position order")
        raw = molecular.batch
        if not sites.shape[0]:
            return raw.x.new_empty((0, self.output_dim))
        molecule = raw.batch[sites[:, 0]]
        if bool((raw.batch[sites] != molecule[:, None]).any()):
            raise ValueError("Reaction sites must belong to the same source molecule")
        centers = sites - raw.ptr[molecule, None]
        position = torch.eye(self.position_count, device=sites.device, dtype=raw.x.dtype)
        labels = raw.x[sites]
        query = self.reactant_projection(torch.cat((labels, position.expand(sites.shape[0], -1, -1)), dim=-1))
        all_atoms = self.atom_gnn(raw.x, raw.edge_index, raw.edge_attr)
        memory = all_atoms[molecular.node_ids.clamp_min(0)]
        degree_features = self.centrality(
            memory.new_zeros(memory.shape), molecular.edge_index, molecular.edge_mask, molecular.node_mask,
        )
        memory = memory + degree_features
        query = query + degree_features[molecule[:, None], centers]
        # Reuse structural rows whenever multiple reactions query the same source atom.
        unique_atoms, inverse = torch.unique(sites.reshape(-1), return_inverse=True)
        unique_bias = rectangular_structural_bias(
            molecular, unique_atoms, self.spatial, self.edge_encoding,
            max(self.max_spatial_dist, self.max_edge_dist),
        )
        bias = unique_bias[inverse.reshape(sites.shape)].permute(0, 2, 1, 3)
        for layer in self.layers:
            query = layer(query, memory, bias, molecule, molecular.node_mask)
        return self.readout(query.flatten(1))
