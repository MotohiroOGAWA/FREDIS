"""Selection features have no dependency on materialized target graphs."""

from __future__ import annotations
import torch
from dataclasses import dataclass
from torch import Tensor, nn
from .components.reaction.reaction_transformer import (
    ReactionMemoryEncoder,
    ReactionStateDecoder,
    reaction_geometry,
)
from .components.condition.adduct_embedding import AdductEmbeddingLayer
from .components.condition.collision_energy_feature import CollisionEnergyFeatureLayer
from .components.reaction.reaction_instance_encoder import ReactionInstanceEncoder


@dataclass(frozen=True)
class SourceReactionFeatures:
    reaction_h: Tensor
    branch_main_adduct_h: Tensor
    source_embeddings: Tensor | None = None
    collision_energy_h: Tensor | None = None
    memory: Tensor | None = None
    memory_ids: Tensor | None = None
    state_h: Tensor | None = None
    local_reaction_h: Tensor | None = None


class SourceReactionFeatureModel(nn.Module):
    def __init__(
        self,
        mol_encoder: nn.Module,
        category_sizes: tuple[int, int, int],
        hidden_dim: int = 128,
        branch_main_adduct_dim: int = 128,
        max_reaction_count: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        adduct_encoder: nn.Module | None = None,
        collision_energy_encoder: nn.Module | None = None,
        pattern_smarts: dict[int, str] | None = None,
        pattern_category_sizes: dict[int, tuple[int, int]] | None = None,
        pattern_model_params: dict[str, dict] | None = None,
        frozen_pattern_ids=(),
        reaction_transformer=None,
    ) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.reaction_instance_encoder = ReactionInstanceEncoder(
            mol_encoder.graph_builder.atom_dim,
            hidden_dim,
            category_sizes,
            num_heads,
            pattern_smarts=pattern_smarts,
            dropout=dropout,
            frozen_pattern_ids=frozen_pattern_ids,
            pattern_category_sizes=pattern_category_sizes,
            pattern_model_params=pattern_model_params,
            raw_bond_dim=mol_encoder.graph_builder.bond_dim,
        )
        if not isinstance(adduct_encoder, AdductEmbeddingLayer):
            raise TypeError("Branching requires an explicit main-adduct encoder")
        if not isinstance(collision_energy_encoder, CollisionEnergyFeatureLayer):
            raise TypeError(
                "Intensity prediction requires a separate collision-energy encoder"
            )
        self.adduct_encoder = adduct_encoder
        self.collision_energy_encoder = collision_energy_encoder
        self.main_adduct_projection = nn.Sequential(
            nn.Linear(adduct_encoder.feature_dim, branch_main_adduct_dim),
            nn.GELU(),
            nn.LayerNorm(branch_main_adduct_dim),
        )
        config = dict(reaction_transformer or {})
        if "max_spatial_distance" in config:
            old_distance = config.pop("max_spatial_distance")
            if config.get("max_reaction_distance", old_distance) != old_distance:
                raise ValueError("Conflicting reaction-instance distance limits")
            config["max_reaction_distance"] = old_distance
        allowed = {
            "encoder_layers",
            "decoder_layers",
            "num_heads",
            "ffn_mult",
            "max_reaction_distance",
        }
        if set(config) - allowed:
            raise ValueError(
                f"Unknown reaction_transformer settings: {set(config) - allowed}"
            )
        if any(type(value) is not int or value < 1 for value in config.values()):
            raise ValueError(
                "Reaction transformer dimensions and layer counts must be positive integers"
            )
        heads = config.get("num_heads", num_heads)
        ff = config.get("ffn_mult", 4)
        self.max_reaction_distance = config.get("max_reaction_distance", 4)
        self.memory_encoder = ReactionMemoryEncoder(
            hidden_dim,
            heads,
            config.get("encoder_layers", 2),
            self.max_reaction_distance,
            dropout,
            ff,
        )
        self.state_decoder = ReactionStateDecoder(
            hidden_dim, heads, config.get("decoder_layers", 2), dropout, ff
        )
        self.decoder_adduct_projection = nn.Linear(branch_main_adduct_dim, hidden_dim)
        self.decoder_root_projection = nn.Linear(mol_encoder.graph_dim, hidden_dim)
        self.branch_query = nn.Linear(hidden_dim + branch_main_adduct_dim, hidden_dim)
        self.branch_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5
        self.max_reaction_count = max_reaction_count

    def freeze_mol_encoder(self) -> None:
        self._freeze_mol_encoder = True
        self.mol_encoder.eval()
        for parameter in self.mol_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_mol_encoder", False):
            self.mol_encoder.eval()
        return self

    def encode_static(self, data, max_graphs=128):
        """Bound source encoding, then share reaction tokens across prediction chunks."""
        from torch_geometric.data import Batch

        device = next(self.parameters()).device
        graphs = data.source_graph.to_data_list()
        molecules = []
        for start in range(0, len(graphs), max_graphs):
            encoded = self.mol_encoder(
                Batch.from_data_list(graphs[start : start + max_graphs]).to(device)
            )
            molecules.append(encoded)
        source_h = torch.cat(molecules)
        kwargs = {
            name: getattr(data.reactions, name).to(device)
            for name in (
                "reaction_type",
                "reaction_source_atom_ptr",
                "reaction_source_atom_index",
            )
        }
        kwargs["source_atom_features"] = data.source_graph.x.to(device)
        kwargs["source_edge_attr"] = data.source_graph.edge_attr.to(device)
        kwargs["source_edge_index"] = data.source_graph.edge_index.to(device)
        kwargs["source_graph_ptr"] = data.source_graph.ptr.to(device)
        reactions = self.reaction_instance_encoder(**kwargs)
        memory, ids = self.encode_memory(reactions, data)
        local_reactions = reactions
        reactions = self.flatten_memory(reactions, memory, ids)
        return reactions, source_h, memory, ids, local_reactions

    def forward(
        self,
        data: object,
        static_features: tuple[Tensor, ...] | None = None,
    ) -> SourceReactionFeatures:
        source = (
            self.mol_encoder(data.source_graph) if static_features is None else None
        )
        reaction = (
            self.reaction_instance_encoder(
                reaction_type=data.reactions.reaction_type,
                reaction_source_atom_ptr=data.reactions.reaction_source_atom_ptr,
                reaction_source_atom_index=data.reactions.reaction_source_atom_index,
                source_atom_features=data.source_graph.x,
                source_edge_index=data.source_graph.edge_index,
                source_edge_attr=data.source_graph.edge_attr,
                source_graph_ptr=data.source_graph.ptr,
            )
            if static_features is None
            else static_features[0]
        )
        local_reaction = reaction if static_features is None else static_features[4]
        source_embeddings = source if source is not None else static_features[1]
        memory, memory_ids = (
            self.encode_memory(reaction, data)
            if static_features is None
            else (static_features[2], static_features[3])
        )
        if static_features is None:
            reaction = self.flatten_memory(reaction, memory, memory_ids)
        collision_energy = self.collision_energy_encoder(data.collision_energy_eV)
        group_adduct_index = data.branch_groups.main_adduct_index
        condition = self.main_adduct_projection(self.adduct_encoder(group_adduct_index))
        # Static reaction tokens are shared by all CE/adduct conditions.
        return SourceReactionFeatures(
            reaction,
            condition,
            source_embeddings,
            collision_energy,
            memory,
            memory_ids,
            self.encode_execution_states(
                data, memory, memory_ids, condition, source_embeddings
            ),
            local_reaction,
        )

    def encode_compounds(self, data, features, max_graphs=128):
        from torch_geometric.data import Batch

        graphs = data.compounds.graph.to_data_list()
        cached = {
            int(cid): features.source_embeddings[i]
            for i, cid in enumerate(data.source_compounds.compound_index.tolist())
        }
        missing = [i for i in range(len(graphs)) if i not in cached]
        for offset in range(0, len(missing), max_graphs):
            ids = missing[offset : offset + max_graphs]
            encoded = self.mol_encoder(
                Batch.from_data_list([graphs[i] for i in ids]).to(data.device)
            )
            cached.update((cid, encoded[j]) for j, cid in enumerate(ids))
        return torch.stack([cached[i] for i in range(len(graphs))])

    @staticmethod
    def flatten_memory(reaction, memory, ids):
        valid = ids >= 0
        return torch.zeros_like(reaction).index_copy(0, ids[valid], memory[valid])

    def encode_memory(self, reaction, data):
        cache_key = (self.max_reaction_distance, str(reaction.device))
        cache = data.__dict__.get("_grpe_geometry")
        if cache is None or cache[0] != cache_key or len(cache[1]) != 4:
            geometry = reaction_geometry(
                data, self.max_reaction_distance, device=reaction.device
            )
            data.__dict__["_grpe_geometry"] = (cache_key, geometry)
        ids, distance, relations, endpoints = data.__dict__["_grpe_geometry"][1]
        ids, distance, relations = (
            t.to(reaction.device) for t in (ids, distance, relations)
        )
        endpoint_positions = self.reaction_instance_encoder.encode_endpoint_positions(
            data.reactions.reaction_type.to(reaction.device),
            data.reactions.reaction_source_atom_ptr.to(reaction.device),
        )
        endpoints = endpoints.to(reaction.device)
        valid = ids >= 0
        padded = torch.cat((reaction, reaction.new_zeros((1, reaction.shape[-1]))))
        tokens = padded[ids.masked_fill(~valid, reaction.shape[0])]
        return (
            self.memory_encoder(tokens, distance, relations, valid, endpoint_positions, endpoints),
            ids,
        )

    @staticmethod
    def reaction_slot_lookup(ids, reaction_count):
        """Build the registry-to-memory mapping once for all state chunks."""
        inverse = ids.new_full((ids.shape[0], reaction_count + 1), -1)
        owner, slot = (ids >= 0).nonzero(as_tuple=True)
        inverse[owner, ids[owner, slot] + 1] = slot
        return inverse

    def decode_sets(self, memory, ids, states, owners, main_adduct, root_molecule,
                    *, slot_lookup=None):
        if slot_lookup is None:
            slot_lookup = self.reaction_slot_lookup(
                ids, int(ids.max()) + 1 if ids.numel() else 0
            )
        local = slot_lookup[owners[:, None], states + 1]
        if bool(((states >= 0) & (local < 0)).any()):
            raise ValueError("State contains a reaction from another source")
        return self.state_decoder(
            memory, local, owners, main_adduct, root_molecule, ids >= 0
        )

    def encode_execution_states(self, data, memory, ids, condition, source_embeddings):
        execution = data.execution_graph
        sets = data.reaction_sets
        ptr, values = sets.gather(execution.reaction_set_index)
        counts = ptr.diff()
        width = int(counts.max()) if counts.numel() else 0
        state = torch.full(
            (counts.numel(), width), -1, device=ids.device, dtype=torch.long
        )
        row = torch.repeat_interleave(
            torch.arange(counts.numel(), device=ids.device), counts
        )
        slot = torch.arange(values.numel(), device=ids.device) - ptr[row]
        state[row, slot] = values
        owners = data.branch_groups.source_compound_index[execution.branch_group_index]
        adduct_tokens = self.decoder_adduct_projection(condition)
        root_tokens = self.decoder_root_projection(source_embeddings)
        lookup = self.reaction_slot_lookup(ids, data.reactions.reaction_type.shape[0])
        local = lookup[owners[:, None], state + 1]
        if bool(((state >= 0) & (local < 0)).any()):
            raise ValueError("State contains a reaction from another source")
        valid = ids >= 0
        # Bound cross-attention memory copies; reuse all static mappings.
        return torch.cat(
            [
                self.state_decoder(
                    memory, local[i : i + 128], owners[i : i + 128],
                    adduct_tokens[execution.branch_group_index[i : i + 128]],
                    root_tokens[owners[i : i + 128]], valid,
                )
                for i in range(0, len(state), 128)
            ]
        )

    def score_execution(self, data, features):
        execution = data.execution_graph
        parent = execution.source_index
        context = torch.cat(
            (
                features.state_h,
                features.branch_main_adduct_h[execution.branch_group_index],
            ),
            1,
        )
        query = self.branch_query(context)[parent]
        key = self.branch_key(features.reaction_h)[execution.reaction_index]
        return (query * key).sum(1) * self.scale

    def score_edges(self, data, features, molecular=None):
        # Compound edge diagnostics merge histories using their strongest score.
        # Training uses score_execution, retaining every canonical state context.
        execution = data.execution_graph
        logits = self.score_execution(data, features)
        graph, ctx = data.compound_graph, data.reaction_contexts
        groups = torch.repeat_interleave(
            torch.arange(data.num_branch_groups, device=data.device),
            ctx.edge_ptr.diff(),
        )
        keys = torch.stack(
            (
                groups,
                graph.source_compound_index[ctx.edge_index],
                graph.target_compound_index[ctx.edge_index],
                graph.reaction_index[ctx.edge_index],
            ),
            1,
        )
        ekeys = torch.stack(
            (
                execution.branch_group_index[execution.source_index],
                execution.compound_index[execution.source_index],
                execution.compound_index[execution.target_index],
                execution.reaction_index,
            ),
            1,
        )
        _, inverse = torch.unique(torch.cat((keys, ekeys)), dim=0, return_inverse=True)
        best = (
            logits.new_full((int(inverse.max()) + 1,), -torch.inf)
            if inverse.numel()
            else logits.new_empty(0)
        )
        best.scatter_reduce_(
            0, inverse[len(keys) :], logits, reduce="amax", include_self=True
        )
        return best[inverse[: len(keys)]], (
            self.encode_compounds(data, features) if molecular is None else molecular
        )
