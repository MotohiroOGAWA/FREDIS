"""Selection features have no dependency on materialized target graphs."""

from __future__ import annotations
import torch
from dataclasses import dataclass
from torch import Tensor, nn
from .components.condition.adduct_embedding import AdductEmbeddingLayer
from .components.condition.collision_energy_feature import CollisionEnergyFeatureLayer
from .components.reaction.reaction_instance_encoder import ReactionInstanceEncoder
from .components.reaction.reaction_decoder import (
    ReactionPool,
    BranchingReactionDecoder,
    build_reaction_pool,
)


@dataclass(frozen=True)
class SourceReactionFeatures:
    reaction_h: Tensor
    branch_main_adduct_h: Tensor
    pool: ReactionPool
    source_embeddings: Tensor | None = None
    collision_energy_h: Tensor | None = None


class SourceReactionFeatureModel(nn.Module):
    def __init__(
        self,
        mol_encoder: nn.Module,
        category_sizes: tuple[int, int, int],
        hidden_dim: int = 128,
        branch_main_adduct_dim: int = 128,
        max_reaction_count: int = 3,
        num_heads: int = 4,
        edge_num_layers: int = 2,
        edge_dropout: float = 0.0,
        adduct_encoder: nn.Module | None = None,
        collision_energy_encoder: nn.Module | None = None,
        pattern_smarts: dict[int, str] | None = None,
        reactant_num_layers: int = 2,
        reactant_dropout: float = 0.0,
        reactant_node_dim: int | None = None,
        reactant_graph_dim: int | None = None,
        pattern_category_sizes: dict[int, tuple[int, int]] | None = None,
        pattern_model_params: dict[str, dict] | None = None,
    ) -> None:
        super().__init__()
        self.mol_encoder = mol_encoder
        self.reaction_instance_encoder = ReactionInstanceEncoder(
            mol_encoder.graph_builder.atom_dim,
            mol_encoder.graph_dim,
            hidden_dim,
            category_sizes,
            num_heads,
            pattern_smarts=pattern_smarts,
            reactant_num_layers=reactant_num_layers,
            reactant_dropout=reactant_dropout,
            reactant_node_dim=reactant_node_dim,
            reactant_graph_dim=reactant_graph_dim,
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
        if edge_num_layers < 1:
            raise ValueError("edge_num_layers must be positive")
        layers = []
        width = 2 * mol_encoder.graph_dim + hidden_dim + branch_main_adduct_dim
        for _ in range(edge_num_layers):
            layers.extend(
                [nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(edge_dropout)]
            )
            width = hidden_dim
        self.edge_head = nn.Sequential(*layers, nn.Linear(width, 1))
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
                "reaction_source_compound_index",
                "reaction_source_atom_ptr",
                "reaction_source_atom_index",
            )
        }
        kwargs["source_atom_features"] = data.source_graph.x.to(device)
        kwargs["source_edge_attr"] = data.source_graph.edge_attr.to(device)
        kwargs["source_edge_index"] = data.source_graph.edge_index.to(device)
        reactions = self.reaction_instance_encoder(source_mol_h=source_h, **kwargs)
        return reactions, source_h

    def forward(
        self,
        data: object,
        training_pool: bool = False,
        static_features: tuple[Tensor, Tensor] | None = None,
    ) -> SourceReactionFeatures:
        source = (
            self.mol_encoder(data.source_graph) if static_features is None else None
        )
        reaction = (
            self.reaction_instance_encoder(
                source_mol_h=source,
                reaction_type=data.reactions.reaction_type,
                reaction_source_compound_index=data.reactions.reaction_source_compound_index,
                reaction_source_atom_ptr=data.reactions.reaction_source_atom_ptr,
                reaction_source_atom_index=data.reactions.reaction_source_atom_index,
                source_atom_features=data.source_graph.x,
                source_edge_index=data.source_graph.edge_index,
                source_edge_attr=data.source_graph.edge_attr,
            )
            if static_features is None
            else static_features[0]
        )
        source_embeddings = source if source is not None else static_features[1]
        collision_energy = self.collision_energy_encoder(data.collision_energy_eV)
        group_adduct_index = data.branch_groups.main_adduct_index
        condition = self.main_adduct_projection(self.adduct_encoder(group_adduct_index))
        # Static reaction tokens are shared by all CE/adduct conditions.
        pool = build_reaction_pool(reaction, data)
        return SourceReactionFeatures(
            reaction,
            condition,
            pool,
            source_embeddings,
            collision_energy,
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

    def score_edges(self, data, features, molecular=None):
        molecular = (
            self.encode_compounds(data, features) if molecular is None else molecular
        )
        ctx, graph = data.reaction_contexts, data.compound_graph
        groups = torch.repeat_interleave(
            torch.arange(data.num_branch_groups, device=data.device),
            ctx.edge_ptr.diff(),
        )
        edge = ctx.edge_index
        inputs = torch.cat(
            (
                molecular[graph.source_compound_index[edge]],
                molecular[graph.target_compound_index[edge]],
                features.reaction_h[graph.reaction_index[edge]],
                features.branch_main_adduct_h[groups],
            ),
            1,
        )
        return self.edge_head(inputs).flatten(), molecular
