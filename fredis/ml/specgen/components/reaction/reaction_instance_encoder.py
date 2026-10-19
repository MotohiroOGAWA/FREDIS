"""Condition-independent reaction embeddings from Original Source only."""

from __future__ import annotations
import torch
from torch import Tensor, nn
from ....reaction.reactant_graphormer import ReactantGraphormerEncoder
from collections.abc import Mapping
from ....reaction.external_neighborhood import ExternalNeighborhoodEncoder
from ....reaction.global_attention import GlobalStructureAttention
from rdkit import Chem


class PatternReactionReadout(nn.Module):
    """Pattern-owned category/context fusion and final reaction feature readout."""

    def __init__(self, graph_dim, mol_dim, hidden_dim, rule_count, product_count):
        super().__init__()
        self.rule_embedding = nn.Embedding(rule_count, hidden_dim)
        self.product_embedding = nn.Embedding(product_count, hidden_dim)
        self.context = nn.Linear(mol_dim + hidden_dim + 6, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(graph_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, atoms, source, reaction_type, static):
        category = self.rule_embedding(reaction_type[:, 1]) + self.product_embedding(
            reaction_type[:, 2]
        )
        inputs = [source, category, static]
        context = self.context(torch.cat(inputs, dim=1))
        return self.output(torch.cat((atoms, context), dim=1))


class ReactionInstanceEncoder(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        mol_dim: int,
        hidden_dim: int,
        category_sizes: tuple[int, int, int],
        num_heads: int = 4,
        pattern_smarts: Mapping[int, str] | None = None,
        reactant_num_layers: int = 2,
        reactant_dropout: float = 0.0,
        reactant_node_dim: int | None = None,
        reactant_graph_dim: int | None = None,
        pattern_category_sizes: Mapping[int, tuple[int, int]] | None = None,
        pattern_model_params: Mapping[str, dict] | None = None,
        raw_bond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        if not pattern_smarts:
            raise ValueError(
                "pattern_smarts is required for pattern-specific reactant Graphormers"
            )
        reactant_node_dim = (
            hidden_dim if reactant_node_dim is None else reactant_node_dim
        )
        self.reactant_graph_dim = (
            hidden_dim if reactant_graph_dim is None else reactant_graph_dim
        )
        self.reactant_encoders = nn.ModuleDict()
        self.pattern_heads = nn.ModuleDict()
        self.neighborhood_encoders = nn.ModuleDict()
        self.global_encoders = nn.ModuleDict()
        self.pattern_model_params = {}
        overrides = pattern_model_params or {}
        unknown = set(map(str, overrides)) - set(map(str, pattern_smarts))
        if unknown:
            raise ValueError(f"Unknown pattern architecture IDs: {sorted(unknown)}")
        for pid, smarts in pattern_smarts.items():
            config = dict(
                hidden_dim=reactant_node_dim,
                graph_dim=self.reactant_graph_dim,
                num_heads=num_heads,
                num_layers=reactant_num_layers,
                dropout=reactant_dropout,
                descriptors=[],
                input_mode="chemical",
                neighborhood_hops=3,
                neighborhood_hidden_dim=32,
                global_num_layers=0,
                global_hidden_dim=32,
                global_num_heads=4,
                neighborhood_edge_features=True,
            )
            override = dict(overrides.get(str(pid), overrides.get(pid, {})))
            if set(override) - set(config):
                raise ValueError(
                    f"Unknown pattern model parameters: {set(override) - set(config)}"
                )
            config.update(override)
            for key in ("hidden_dim", "graph_dim", "num_heads", "num_layers"):
                if type(config[key]) is not int or config[key] < 1:
                    raise ValueError(f"{key} must be a positive integer")
            if not 0 <= config["dropout"] < 1:
                raise ValueError("dropout must be between 0 and 1")
            self.pattern_model_params[str(pid)] = config
            if (
                config["input_mode"] != "chemical"
                or not config["neighborhood_edge_features"]
            ):
                raise ValueError(
                    "Only chemical, bond-aware reaction inputs are supported"
                )
            if config["descriptors"]:
                raise ValueError(
                    "Reaction descriptors are unsupported; use spectrum descriptors"
                )
            if config["neighborhood_hops"]:
                self.neighborhood_encoders[str(pid)] = ExternalNeighborhoodEncoder(
                    atom_dim,
                    config["neighborhood_hidden_dim"],
                    config["neighborhood_hops"],
                    bond_dim=raw_bond_dim,
                )
            else:
                # Validate even when no neighborhood parameters are allocated.
                if (
                    type(config["neighborhood_hops"]) is not int
                    or config["neighborhood_hops"] != 0
                ):
                    raise ValueError("neighborhood_hops must be a non-negative integer")
                if (
                    type(config["neighborhood_hidden_dim"]) is not int
                    or config["neighborhood_hidden_dim"] < 1
                ):
                    raise ValueError("neighborhood_hidden_dim must be positive")
            if (
                type(config["global_num_layers"]) is not int
                or config["global_num_layers"] < 0
            ):
                raise ValueError("global_num_layers must be a non-negative integer")
            if config["global_num_layers"]:
                self.global_encoders[str(pid)] = GlobalStructureAttention(
                    atom_dim,
                    raw_bond_dim,
                    Chem.MolFromSmarts(smarts).GetNumAtoms(),
                    config["global_hidden_dim"],
                    config["global_num_layers"],
                    config["global_num_heads"],
                    config["dropout"],
                )
            input_dim = atom_dim + (
                config["neighborhood_hidden_dim"] if config["neighborhood_hops"] else 0
            )
            if config["global_num_layers"]:
                input_dim += config["global_hidden_dim"]
            self.reactant_encoders[str(pid)] = ReactantGraphormerEncoder(
                smarts,
                atom_dim=input_dim,
                bond_dim=0,
                hidden_dim=config["hidden_dim"],
                graph_dim=config["graph_dim"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                dropout=config["dropout"],
            )
            self.pattern_heads[str(pid)] = PatternReactionReadout(
                config["graph_dim"],
                mol_dim,
                hidden_dim,
                *(pattern_category_sizes or {}).get(pid, category_sizes[1:]),
            )
        self.frozen_pattern_ids = set()

    def train(self, mode=True):
        super().train(mode)
        for pid in self.frozen_pattern_ids:
            self.reactant_encoders[pid].eval()
            self.pattern_heads[pid].eval()
            if pid in self.neighborhood_encoders:
                self.neighborhood_encoders[pid].eval()
            if pid in self.global_encoders:
                self.global_encoders[pid].eval()
        return self

    def forward(
        self,
        *,
        source_mol_h: Tensor,
        reaction_type: Tensor,
        reaction_tree_index: Tensor,
        reaction_source_atom_ptr: Tensor,
        reaction_source_atom_index: Tensor,
        reaction_static_features: Tensor,
        source_atom_features: Tensor | None = None,
        source_edge_index: Tensor | None = None,
        source_edge_attr: Tensor | None = None,
        source_batch: Tensor | None = None,
    ) -> Tensor:
        a = reaction_type.shape[0]
        if a == 0:
            return source_mol_h.new_empty((0, self.output_dim))
        counts = reaction_source_atom_ptr[1:] - reaction_source_atom_ptr[:-1]
        result = source_mol_h.new_zeros((a, self.output_dim))
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
            if source_atom_features is None or source_edge_index is None:
                raise ValueError(
                    "Raw atom chemical features and source edges are required"
                )
            sites = reaction_source_atom_index[index]
            inputs = source_atom_features[sites]
            if pattern_id in self.neighborhood_encoders:
                outside = self.neighborhood_encoders[pattern_id](
                    source_atom_features, source_edge_index, sites, source_edge_attr
                )
                inputs = torch.cat((inputs, outside), dim=-1)
            if pattern_id in self.global_encoders:
                if source_edge_attr is None or source_batch is None:
                    raise ValueError(
                        "Global attention requires source bond features and molecule ownership"
                    )
                global_h = self.global_encoders[pattern_id](
                    source_atom_features,
                    source_edge_index,
                    source_edge_attr,
                    source_batch,
                    sites,
                    reaction_tree_index[rows],
                )
                inputs = torch.cat((inputs, global_h), dim=-1)
            _, graph_h = encoder.encode_matches(inputs)
            result[rows] = self.pattern_heads[pattern_id](
                graph_h,
                source_mol_h[reaction_tree_index[rows]],
                reaction_type[rows],
                reaction_static_features[rows],
            )
            known[rows] = True
        if not torch.all(known):
            raise ValueError("Unknown reaction pattern ID in prepared data")
        return result
