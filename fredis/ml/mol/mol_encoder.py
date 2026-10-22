from __future__ import annotations

from copy import copy
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from ...libs.mmkit.mmkit import Compound
from .graph_builder import MolGraphBuilder
from .mol_graphormer import (
    DEFAULT_MOL_GRAPHORMER_DROPOUT,
    DEFAULT_MOL_GRAPHORMER_MAX_DEGREE,
    DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST,
    DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
    MolGraphormerEncoder,
)


class MolEncoder(nn.Module):
    """Graphormer with optional caller-supplied graph-level feature vectors.

    ``external_features`` has shape [num_graphs, external_feature_dim]. It may
    contain descriptors or any other numeric features. Selection, scaling and
    missing-value handling belong to the caller. Graph outputs retain graph_dim.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        node_dim: int,
        graph_dim: int,
        num_layers: int,
        num_heads: int,
        max_degree: int = DEFAULT_MOL_GRAPHORMER_MAX_DEGREE,
        max_spatial_dist: int = DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
        max_edge_dist: int = DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST,
        dropout: float = DEFAULT_MOL_GRAPHORMER_DROPOUT,
        external_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if type(external_feature_dim) is not int or external_feature_dim < 0:
            raise ValueError("external_feature_dim must be a non-negative integer")
        self.external_feature_dim = external_feature_dim
        self.graph_builder = MolGraphBuilder(symbols=symbols)
        self.encoder = MolGraphormerEncoder(
            atom_dim=self.graph_builder.atom_dim,
            bond_dim=self.graph_builder.bond_dim,
            node_dim=node_dim,
            graph_dim=graph_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_degree=max_degree,
            max_spatial_dist=max_spatial_dist,
            max_edge_dist=max_edge_dist,
            dropout=dropout,
        )
        self._node_dim = self.encoder.mol_node_dim
        self._graph_dim = self.encoder.mol_graph_dim
        self.external_dropout = nn.Dropout(0.0)
        if external_feature_dim:
            self.external_feature_projection = nn.Sequential(
                nn.Linear(external_feature_dim, self._graph_dim),
                nn.GELU(),
                nn.Linear(self._graph_dim, self._graph_dim),
            )
            self.external_feature_fusion = nn.Linear(self._graph_dim * 2, self._graph_dim)
        else:
            self.external_feature_projection = None
            self.external_feature_fusion = None

    @property
    def node_dim(self) -> int:
        return self._node_dim

    @property
    def graph_dim(self) -> int:
        return self._graph_dim

    @property
    def atom_dim(self) -> int:
        return self.graph_builder.atom_dim

    @property
    def bond_dim(self) -> int:
        return self.graph_builder.bond_dim

    @property
    def symbols(self) -> Tuple[str, ...]:
        return self.graph_builder.symbols

    def encode_components(self, compound: Compound) -> Data:
        device = next(self.parameters()).device
        return self.graph_builder.build(compound, device=device)

    def _encode_graph(
        self, batch: Batch, external_features: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        node_h, graph_h = self.encoder(batch)
        if external_features is None:
            external_features = getattr(batch, "external_features", None)
        if self.external_feature_dim == 0:
            if external_features is not None:
                raise ValueError("external_features requires external_feature_dim > 0")
            return node_h, graph_h
        expected = (graph_h.shape[0], self.external_feature_dim)
        if external_features is None or external_features.shape != expected:
            raise ValueError(f"external_features must have shape {expected}")
        values = external_features.to(device=graph_h.device, dtype=graph_h.dtype)
        if not torch.isfinite(values).all():
            raise ValueError("external_features must contain only finite values")
        feature_h = self.external_dropout(self.external_feature_projection(values))
        return node_h, self.external_dropout(self.external_feature_fusion(
            torch.cat((graph_h, feature_h), dim=-1)
        ))

    def forward(self, batch: Batch, external_features: Tensor | None = None) -> Tensor:
        return self._encode_graph(batch, external_features)[1]

    def encode_with_nodes(
        self, batch: Batch, external_features: Tensor | None = None
    ) -> Batch:
        """Fuse external features into graph outputs; atom outputs stay structural."""
        node_h, graph_h = self._encode_graph(batch, external_features)
        encoded = copy(batch)
        encoded.x = node_h
        encoded.embeddings = graph_h
        return encoded

    def encode(
        self, compound: Compound, external_features: Tensor | None = None
    ) -> Optional[Tensor]:
        try:
            return self(
                Batch.from_data_list([self.encode_components(compound)]), external_features
            )[0]
        except Exception:
            return None

    def encode_batch(
        self, compounds: List[Compound], external_features: Tensor | None = None
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        data_list: list[Data] = []
        indices: list[int] = []
        device = next(self.parameters()).device
        for index, compound in enumerate(compounds):
            try:
                data_list.append(self.graph_builder.build(compound, device=device))
                indices.append(index)
            except Exception:
                pass
        if not data_list:
            return None, None
        features = external_features[indices] if external_features is not None else None
        return self(Batch.from_data_list(data_list), features), torch.tensor(
            indices, dtype=torch.long, device=device
        )
