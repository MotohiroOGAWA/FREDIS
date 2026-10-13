"""Condition-independent reaction embeddings from Original Source only."""
from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from ....reaction.reactant_graphormer import ReactantGraphormerEncoder
from collections.abc import Mapping

NEIGHBORHOOD_MODES = ("hop_pooling", "none")


def csr_mean_pool(source_atom_h: Tensor, ptr: Tensor, index: Tensor) -> Tensor:
    """Mean-pool source_atom_h rows named by a CSR (ptr, index) pair; empty rows are zero.

    Tensor-only (no Python-side loop over rows): counts/owner/index_add_ scale
    with the number of (row, neighbor) pairs actually present, never a dense
    [num_rows, num_atoms] matrix.
    """
    counts = ptr[1:] - ptr[:-1]
    owner = torch.repeat_interleave(torch.arange(counts.numel(), device=source_atom_h.device), counts)
    pooled = source_atom_h.new_zeros(counts.numel(), source_atom_h.shape[-1])
    if index.numel():
        pooled.index_add_(0, owner, source_atom_h[index])
    return pooled / counts.clamp_min(1).unsqueeze(-1)


class PatternReactionReadout(nn.Module):
    """Pattern-owned category/context fusion and final reaction feature readout."""
    def __init__(self, graph_dim, mol_dim, hidden_dim, rule_count, product_count):
        super().__init__()
        self.rule_embedding = nn.Embedding(rule_count, hidden_dim)
        self.product_embedding = nn.Embedding(product_count, hidden_dim)
        self.context = nn.Linear(mol_dim + hidden_dim + 6, hidden_dim)
        self.output = nn.Sequential(nn.Linear(graph_dim + hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, atoms, source, reaction_type, static):
        category = self.rule_embedding(reaction_type[:, 1]) + self.product_embedding(reaction_type[:, 2])
        context = self.context(torch.cat((source, category, static), dim=1))
        return self.output(torch.cat((atoms, context), dim=1))


class ReactionInstanceEncoder(nn.Module):
    def __init__(self, atom_dim: int, mol_dim: int, hidden_dim: int,
                 category_sizes: tuple[int, int, int], num_heads: int = 4,
                 max_roles: int = 64, reaction_neighborhood_mode: str = "hop_pooling",
                 reaction_neighborhood_max_hop: int = 3,
                 pattern_smarts: Mapping[int, str] | None = None,
                 reactant_num_layers: int = 2, reactant_dropout: float = 0.0,
                 reactant_node_dim: int | None = None, reactant_graph_dim: int | None = None,
                 pattern_category_sizes: Mapping[int, tuple[int, int]] | None = None) -> None:
        super().__init__()
        if reaction_neighborhood_mode not in NEIGHBORHOOD_MODES:
            raise ValueError(f"Unsupported reaction_neighborhood_mode: {reaction_neighborhood_mode!r}")
        if reaction_neighborhood_max_hop < 1:
            raise ValueError("reaction_neighborhood_max_hop must be a positive integer")
        self.reaction_neighborhood_mode = reaction_neighborhood_mode
        self.reaction_neighborhood_max_hop = reaction_neighborhood_max_hop
        self.output_dim = hidden_dim
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        # SMARTS query roles are chemical features, never reaction-list positions.
        self.role_status = nn.Linear(6,hidden_dim,bias=False)
        self.roles = nn.Embedding(max_roles, hidden_dim)
        # One projection per hop distance (1..max_hop): each hop has a distinct
        # meaning, so they are not shared. bias=False and zero initialization
        # make an all-empty neighborhood (or a freshly fine-tuned adapter)
        # contribute exactly zero until gradients move it.
        self.neighborhood_projections = nn.ModuleList(nn.Linear(atom_dim, hidden_dim, bias=False) for _ in range(reaction_neighborhood_max_hop))
        for layer in self.neighborhood_projections:
            nn.init.zeros_(layer.weight)
        if not pattern_smarts:
            raise ValueError("pattern_smarts is required for pattern-specific reactant Graphormers")
        reactant_node_dim = hidden_dim if reactant_node_dim is None else reactant_node_dim
        self.reactant_graph_dim = hidden_dim if reactant_graph_dim is None else reactant_graph_dim
        self.reactant_encoders = nn.ModuleDict({str(int(pid)): ReactantGraphormerEncoder(
            smarts, atom_dim=hidden_dim, bond_dim=0, node_dim=reactant_node_dim, graph_dim=self.reactant_graph_dim,
            num_heads=num_heads, num_layers=reactant_num_layers, dropout=reactant_dropout)
            for pid, smarts in pattern_smarts.items()})
        self.pattern_heads = nn.ModuleDict({str(int(pid)): PatternReactionReadout(
            self.reactant_graph_dim, mol_dim, hidden_dim,
            *(pattern_category_sizes or {}).get(pid, category_sizes[1:])) for pid in pattern_smarts})
        self.frozen_pattern_ids = set()

    def train(self, mode=True):
        super().train(mode)
        for pid in self.frozen_pattern_ids:
            self.reactant_encoders[pid].eval()
            self.pattern_heads[pid].eval()
        return self

    def forward(self, *, source_atom_h: Tensor, source_mol_h: Tensor, reaction_type: Tensor,
                reaction_tree_index: Tensor, reaction_source_atom_ptr: Tensor,
                reaction_source_atom_index: Tensor, reaction_static_features: Tensor,
                reaction_source_atom_hops: Sequence[tuple[Tensor, Tensor]],
                reaction_source_atom_features: Tensor | None = None) -> Tensor:
        a = reaction_type.shape[0]
        if a == 0:
            return source_atom_h.new_empty((0, self.output_dim))
        counts = reaction_source_atom_ptr[1:] - reaction_source_atom_ptr[:-1]
        owner = torch.repeat_interleave(torch.arange(a, device=counts.device), counts)
        role = torch.arange(owner.numel(), device=counts.device) - reaction_source_atom_ptr[owner]
        # Query role identity is independent of the order of reactions.
        selected = self.atom_projection(source_atom_h[reaction_source_atom_index]) + self.roles(role.clamp_max(self.roles.num_embeddings - 1))
        if reaction_source_atom_features is not None:
            selected = selected + self.role_status(reaction_source_atom_features)
        if self.reaction_neighborhood_mode == "hop_pooling":
            if len(reaction_source_atom_hops) != len(self.neighborhood_projections):
                raise ValueError(f"Expected {len(self.neighborhood_projections)} hop distances, got {len(reaction_source_atom_hops)}")
            neighborhood = None
            for projection, (ptr, index) in zip(self.neighborhood_projections, reaction_source_atom_hops):
                pooled = projection(csr_mean_pool(source_atom_h, ptr, index))
                neighborhood = pooled if neighborhood is None else neighborhood + pooled
            selected = selected + neighborhood
        result = selected.new_zeros((a, self.output_dim))
        known = torch.zeros(a, dtype=torch.bool, device=counts.device)
        for pattern_id, encoder in self.reactant_encoders.items():
            rows = (reaction_type[:, 0] == int(pattern_id)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            if torch.any(counts[rows] != encoder.position_count):
                raise ValueError(f"Reactant atom count does not match SMARTS for pattern {pattern_id}")
            slots = torch.arange(encoder.position_count, device=counts.device)
            index = reaction_source_atom_ptr[rows, None] + slots[None, :]
            _, graph_h = encoder.encode_matches(selected[index])
            result[rows] = self.pattern_heads[pattern_id](graph_h, source_mol_h[reaction_tree_index[rows]],
                                                        reaction_type[rows], reaction_static_features[rows])
            known[rows] = True
        if not torch.all(known):
            raise ValueError("Unknown reaction pattern ID in prepared data")
        return result
