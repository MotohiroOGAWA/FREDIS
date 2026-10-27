"""Condition-independent reaction embeddings from local source chemistry."""

from __future__ import annotations
import torch
from torch import Tensor, nn
from collections.abc import Mapping
from .reactant_atom_attention import ReactionAtomEncoder, build_molecular_batch


class PatternReactionReadout(nn.Module):
    """Pattern-owned additive rule/product features after the direct linear readout."""

    def __init__(
        self, hidden_dim, rule_count, product_count, *, dropout=0.0
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.rule_embedding = nn.Embedding(rule_count, hidden_dim)
        self.product_embedding = nn.Embedding(product_count, hidden_dim)

    def forward(self, atoms, reaction_type):
        category = self.rule_embedding(reaction_type[:, 1]) + self.product_embedding(
            reaction_type[:, 2]
        )
        return self.dropout(atoms + category)


class ReactionInstanceEncoder(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int,
        category_sizes: tuple[int, int, int],
        num_heads: int = 4,
        pattern_smarts: Mapping[int, str] | None = None,
        dropout: float = 0.0,
        pattern_category_sizes: Mapping[int, tuple[int, int]] | None = None,
        pattern_model_params: Mapping[str, dict] | None = None,
        raw_bond_dim: int = 0,
        frozen_pattern_ids=(),
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        if not pattern_smarts:
            raise ValueError(
                "pattern_smarts is required for pattern-specific reaction atom encoders"
            )
        self.reactant_encoders = nn.ModuleDict()
        self.pattern_heads = nn.ModuleDict()
        self.endpoint_projections = nn.ModuleDict()
        self.pattern_model_params = {}
        overrides = pattern_model_params or {}
        frozen = set(map(str, frozen_pattern_ids))
        unknown = set(map(str, overrides)) - set(map(str, pattern_smarts))
        if unknown:
            raise ValueError(f"Unknown pattern architecture IDs: {sorted(unknown)}")
        for pid, smarts in pattern_smarts.items():
            config = dict(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=2,
                dropout=dropout,
                ffn_mult=4,
                max_degree=4,
                max_spatial_dist=4,
                max_edge_dist=4,
            )
            override = dict(overrides.get(str(pid), overrides.get(pid, {})))
            if set(override) - set(config):
                raise ValueError(
                    f"Unknown pattern model parameters: {set(override) - set(config)}"
                )
            config.update(override)
            if str(pid) not in frozen:
                config["dropout"] = dropout
            for key in ("hidden_dim", "num_heads", "num_layers", "ffn_mult",
                        "max_degree", "max_spatial_dist", "max_edge_dist"):
                if type(config[key]) is not int or config[key] < 1:
                    raise ValueError(f"{key} must be a positive integer")
            if not 0 <= config["dropout"] < 1:
                raise ValueError("dropout must be between 0 and 1")
            self.pattern_model_params[str(pid)] = config
            self.reactant_encoders[str(pid)] = ReactionAtomEncoder(
                smarts, atom_dim=atom_dim,
                bond_dim=raw_bond_dim, output_dim=hidden_dim, **config,
            )
            self.endpoint_projections[str(pid)] = nn.Linear(
                self.reactant_encoders[str(pid)].position_count, hidden_dim, bias=False,
            )
            self.pattern_heads[str(pid)] = PatternReactionReadout(
                hidden_dim,
                *(pattern_category_sizes or {}).get(pid, category_sizes[1:]),
                dropout=config["dropout"],
            )
        self.frozen_pattern_ids = set()

    def train(self, mode=True):
        super().train(mode)
        for pid in self.frozen_pattern_ids:
            self.reactant_encoders[pid].eval()
            self.pattern_heads[pid].eval()
            self.endpoint_projections[pid].eval()
        return self

    def encode_endpoint_positions(self, reaction_type, atom_ptr):
        """Map each SMARTS-position one-hot to the shared reaction hidden width.

        A weight-column lookup is exactly a bias-free Linear applied to one-hot
        input. Pattern-owned weights are exported, remapped and frozen together
        with each reaction feature encoder; no separate embedding width is used.
        """
        counts = atom_ptr.diff()
        owners = torch.repeat_interleave(torch.arange(counts.numel(), device=atom_ptr.device), counts)
        positions = torch.arange(owners.numel(), device=atom_ptr.device) - atom_ptr[owners]
        prototype = next(self.endpoint_projections.parameters())
        features = prototype.new_zeros((owners.numel(), self.output_dim))
        patterns = reaction_type[owners, 0]
        for pid, projection in self.endpoint_projections.items():
            rows = (patterns == int(pid)).nonzero(as_tuple=True)[0]
            features = features.index_copy(0, rows, projection.weight.T[positions[rows]])
        return features

    def forward(
        self,
        *,
        reaction_type: Tensor,
        reaction_source_atom_ptr: Tensor,
        reaction_source_atom_index: Tensor,
        source_atom_features: Tensor,
        source_edge_index: Tensor,
        source_edge_attr: Tensor | None = None,
        source_graph_ptr: Tensor | None = None,
    ) -> Tensor:
        a = reaction_type.shape[0]
        if a == 0:
            return source_atom_features.new_empty((0, self.output_dim))
        counts = reaction_source_atom_ptr[1:] - reaction_source_atom_ptr[:-1]
        molecular = build_molecular_batch(source_atom_features, source_edge_index, source_edge_attr, source_graph_ptr)
        result = source_atom_features.new_zeros((a, self.output_dim))
        known = torch.zeros(a, dtype=torch.bool, device=counts.device)
        for pattern_id, encoder in self.reactant_encoders.items():
            rows = (reaction_type[:, 0] == int(pattern_id)).nonzero().flatten()
            if rows.numel() == 0:
                continue
            if torch.any(counts[rows] != encoder.position_count):
                raise ValueError(
                    f"Reactant atom count does not match SMARTS for pattern {pattern_id}"
                )
            slots = torch.arange(encoder.position_count, device=counts.device)
            index = reaction_source_atom_ptr[rows, None] + slots[None, :]
            sites = reaction_source_atom_index[index]
            graph_h = encoder(molecular, sites)
            result[rows] = self.pattern_heads[pattern_id](
                graph_h,
                reaction_type[rows],
            )
            known[rows] = True
        if not torch.all(known):
            raise ValueError("Unknown reaction pattern ID in prepared data")
        return result
