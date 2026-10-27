"""Reaction relation encoding, memory encoding, and reaction-set decoding.

Distance and chemical-relation terms follow GRPE (Park et al., 2022).
The encoder is run once per source, never once per state or collision energy.
"""

import torch
from torch import nn


from ....common.layers.transformer_encoder import TransformerEncoder
from ....common.layers.transformer_decoder import TransformerDecoder


class ReactionRelationEncoding(nn.Module):
    """Query/key-aware distances, chemical relations and nearest SMARTS positions."""

    def __init__(self, dim, heads, max_distance=4):
        super().__init__()
        self.heads, self.width = heads, dim // heads
        self.spatial = nn.ModuleList(
            nn.Embedding(max_distance + 2, dim) for _ in range(3)
        )
        self.relations = nn.ModuleList(nn.Linear(4, dim, bias=False) for _ in range(3))

    def attention_bias(self, q, k, features):
        distance, relations = features[:2]

        def product(tokens, index, transpose=False):
            positions = distance.transpose(1, 2) if transpose else distance
            edges = relations.transpose(1, 2) if transpose else relations
            table = self.spatial[index].weight.reshape(-1, self.heads, self.width)
            scores = torch.einsum("bhid,rhd->bhir", tokens, table)
            spatial = scores.gather(
                3, positions[:, None].expand(-1, self.heads, -1, -1)
            )
            edge = torch.einsum(
                "bhid,rhd->bhir",
                tokens,
                self.relations[index].weight.T.reshape(4, self.heads, self.width),
            )
            return spatial + torch.einsum("bhir,bijr->bhij", edge, edges.to(edge.dtype))

        bias = product(q, 0) + product(k, 1, True).transpose(-1, -2)
        if len(features) == 4:
            positions, endpoints = features[2:]
            bias = bias + self.endpoint_attention_bias(q, k, positions, endpoints)
        return bias

    def value_encoding(self, attention, features):
        distance, relations = features[:2]
        b, h, n, _ = attention.shape
        mass = attention.new_zeros((b, h, n, self.spatial[2].num_embeddings))
        mass.scatter_add_(3, distance[:, None].expand(-1, h, -1, -1), attention)
        value = torch.einsum(
            "bhir,rhd->bhid", mass, self.spatial[2].weight.reshape(-1, h, self.width)
        )
        edge_mass = torch.einsum(
            "bhij,bijr->bhir", attention, relations.to(attention.dtype)
        )
        value = value + torch.einsum(
            "bhir,rhd->bhid",
            edge_mass,
            self.relations[2].weight.T.reshape(4, h, self.width),
        )
        if len(features) == 4:
            positions, endpoints = features[2:]
            value = value + self.endpoint_value_encoding(attention, positions, endpoints)
        return value

    def endpoint_attention_bias(self, q, k, positions, endpoints):
        """Sum every tied endpoint's one-hot projection, without dense pair embeddings."""
        batch, _, count, _ = q.shape
        pair, atom = endpoints
        source_batch = pair // (count * count)
        source = pair // count % count
        target = pair % count
        embedded = positions[atom].reshape(-1, self.heads, self.width).to(q.dtype)
        query_scores = (q[source_batch, :, source] * embedded).sum(-1)
        key_scores = (k[source_batch, :, source] * embedded).sum(-1)
        reverse = (source_batch * count + target) * count + source
        bias = q.new_zeros((batch * count * count, self.heads))
        bias = bias.index_add(0, pair, query_scores).index_add(0, reverse, key_scores)
        return bias.reshape(batch, count, count, self.heads).permute(0, 3, 1, 2)

    def endpoint_value_encoding(self, attention, positions, endpoints):
        batch, _, count, _ = attention.shape
        pair, atom = endpoints
        source_batch = pair // (count * count)
        source = pair // count % count
        target = pair % count
        embedded = positions[atom].reshape(-1, self.heads, self.width).to(attention.dtype)
        target_values = attention[source_batch, :, source, target, None] * embedded
        source_values = attention[source_batch, :, target, source, None] * embedded
        value = attention.new_zeros((batch * count, self.heads, self.width))
        value = value.index_add(0, source_batch * count + source, target_values)
        value = value.index_add(0, source_batch * count + target, source_values)
        return value.reshape(batch, count, self.heads, self.width).permute(0, 2, 1, 3)


class ReactionMemoryEncoder(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        layers=2,
        max_distance=4,
        dropout=0.0,
        ffn_mult=4,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            dim,
            heads,
            layers,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )
        for layer in self.encoder.layers:
            layer.self_attn.relative_encoding = (
                ReactionRelationEncoding(
                    dim, heads, max_distance
                )
            )

    def forward(self, x, distance, relations, valid, endpoint_positions=None, endpoints=None):
        if not x.shape[1]:
            return x
        if (endpoint_positions is None) != (endpoints is None):
            raise ValueError("Endpoint positions and endpoint indices must be supplied together")
        features = (distance, relations)
        if endpoint_positions is not None:
            features += (endpoint_positions, endpoints)
        return self.encoder(x, key_padding_mask=~valid, relative_features=features)[0]


class ReactionStateDecoder(nn.Module):
    def __init__(self, dim, heads=4, layers=2, dropout=0.0, ffn_mult=4):
        super().__init__()
        self.bos = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.token_types = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.decoder = TransformerDecoder(
            dim,
            heads,
            layers,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )

    def forward(self, memory, state, compound, main_adduct, root_molecule, valid=None):
        if not compound.numel():
            return memory.new_empty((0, memory.shape[-1]))
        if memory.shape[1] == 0:
            memory = memory.new_zeros((memory.shape[0], 1, memory.shape[-1]))
            valid = torch.zeros(
                memory.shape[:2], dtype=torch.bool, device=memory.device
            )
        tokens = memory[compound[:, None], state.clamp_min(0)]
        padding = state < 0
        tokens = torch.cat(
            (
                self.bos.expand(len(compound), -1, -1) + self.token_types[0],
                main_adduct[:, None] + self.token_types[1],
                root_molecule[:, None] + self.token_types[2],
                (tokens + self.token_types[3]).masked_fill(padding[..., None], 0),
            ),
            1,
        )
        padding = torch.cat((padding.new_zeros((len(compound), 3)), padding), 1)
        memory_padding = None if valid is None else ~valid[compound].clone()
        if memory_padding is not None:
            memory_padding[:, 0] &= ~memory_padding.all(1)
        hidden, _ = self.decoder(
            tokens,
            memory[compound],
            query_padding_mask=padding,
            memory_key_padding_mask=memory_padding,
        )
        # BOS is invariant under permutations of applied reactions.
        return hidden[:, 0]


@torch.no_grad()
def reaction_geometry(data, max_distance=4, *, device=None):
    """Expand matched substructures through sparse atom adjacency by BFS depth.

    The adjacency is block diagonal across source molecules. Columns represent
    source-local reaction slots, so propagation uses [N_atoms, max_reactions]
    rather than [N_atoms, N_reactions_total]. Only the depth loop uses Python;
    sparse matrix products expand the frontier and aggregate target matches.
    """
    if type(max_distance) is not int or max_distance < 1:
        raise ValueError("max_distance must be a positive integer")
    registry = data.reactions
    device = registry.reaction_type.device if device is None else device
    owners = registry.reaction_source_compound_index.to(device)
    source_count = data.source_graph.ptr.numel() - 1
    atom_count = data.source_graph.num_nodes
    count = owners.numel()
    counts = torch.bincount(owners, minlength=source_count)
    width = int(counts.max()) if source_count else 0
    order = torch.argsort(owners, stable=True)
    starts = counts.cumsum(0) - counts
    slots = torch.empty_like(owners)
    slots[order] = torch.arange(count, device=device) - starts[owners[order]]
    ids = owners.new_full((source_count, width), -1)
    ids[owners, slots] = torch.arange(count, device=device)
    distance = owners.new_full((source_count, width, width), -1)
    relations = torch.zeros(
        (source_count, width, width, 4), dtype=torch.bool, device=device
    )
    for channel, pairs in enumerate(
        (registry.reaction_conflict_index, registry.reaction_invalidation_index)
    ):
        left, right = pairs.to(device)
        relations[owners[left], slots[left], slots[right], channel] = True
    if not count:
        return ids, distance, relations, owners.new_empty((2, 0))
    atom_ptr = registry.reaction_source_atom_ptr.to(device)
    atoms = registry.reaction_source_atom_index.to(device)
    matched_owner = torch.repeat_interleave(
        torch.arange(count, device=device), atom_ptr.diff()
    )
    edges = data.source_graph.edge_index.to(device)
    # Multiplication moves information from edge source to edge destination.
    adjacency = torch.sparse_coo_tensor(
        edges.flip(0),
        torch.ones(edges.shape[1], device=device, dtype=torch.float32),
        (atom_count, atom_count),
    ).coalesce()
    membership = torch.sparse_coo_tensor(
        torch.stack((matched_owner, atoms)),
        torch.ones(atoms.numel(), device=device, dtype=torch.float32),
        (count, atom_count),
    ).coalesce()

    def multiply(matrix, active):
        # Sparse CUDA kernels require float32 even during mixed-precision forward.
        with torch.autocast(device_type=owners.device.type, enabled=False):
            return torch.sparse.mm(matrix, active)

    def reaction_overlap(active):
        # Rows of membership are target reactions; columns are source slots.
        hits = multiply(membership, active) > 0
        packed = torch.zeros_like(distance, dtype=torch.bool)
        packed[owners, slots] = hits
        return packed.transpose(1, 2)

    frontier = torch.zeros((atom_count, width), device=device, dtype=torch.float32)
    frontier[atoms, slots[matched_owner]] = 1
    endpoint_rows = []

    def record_nearest_positions(active, depth):
        atom_position, source_slot = (active[atoms] > 0).nonzero(as_tuple=True)
        target = matched_owner[atom_position]
        source = owners[target]
        target_slot = slots[target]
        nearest = distance[source, source_slot, target_slot] == depth
        pair = (source * width + source_slot) * width + target_slot
        endpoint_rows.append(torch.stack((pair[nearest], atom_position[nearest])))

    visited = frontier.bool()
    shared = reaction_overlap(frontier)
    distance.masked_fill_(shared, 0)
    relations[..., 2] = shared
    record_nearest_positions(frontier, 0)
    for depth in range(1, max_distance + 1):
        frontier_nodes = (multiply(adjacency, frontier) > 0) & ~visited
        visited |= frontier_nodes
        frontier = frontier_nodes.float()
        hits = reaction_overlap(frontier)
        # A shortest distance is written once, at the first reaching depth.
        distance.masked_fill_(hits & (distance == -1), depth)
        record_nearest_positions(frontier, depth)
    distance.masked_fill_(distance == -1, max_distance + 1)
    retained_ptr = registry.reaction_retained_atom_ptr.to(device)
    retained_atoms = registry.reaction_retained_atom_index.to(device)
    retained_owner = torch.repeat_interleave(
        torch.arange(count, device=device), retained_ptr.diff()
    )
    frontier.zero_()
    frontier[retained_atoms, slots[retained_owner]] = 1
    relations[..., 3] = reaction_overlap(frontier)
    return ids, distance, relations, torch.cat(endpoint_rows, dim=1)
