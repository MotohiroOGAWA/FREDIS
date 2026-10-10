from __future__ import annotations

from copy import copy
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from ...domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
from ...libs.mmkit.mmkit import Compound
from .graph_builder import MolGraphBuilder
from .mol_graphormer import (
    DEFAULT_MOL_GRAPHORMER_DROPOUT,
    DEFAULT_MOL_GRAPHORMER_MAX_DEGREE,
    DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST,
    DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
    MolGraphormerEncoder,
)


DEFAULT_MOL_ENCODER_DESCRIPTORS = ("FormalCharge",)


class MolEncoder(nn.Module):
    """Graphormer molecular encoder with built-in molecular descriptors.

    The graph embedding is concatenated with an embedding of the selected
    molecular ``descriptors`` (computed by :class:`MolGraphBuilder`) and
    projected back to ``graph_dim``.

    :meth:`forward` returns ``[num_graphs, graph_dim]``.  :meth:`encode_with_nodes`
    additionally returns the per-atom hidden states that the reaction instance
    encoder pools over reacting and neighboring atoms.
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
        descriptors: Sequence[str] = DEFAULT_MOL_ENCODER_DESCRIPTORS,
    ) -> None:
        super().__init__()
        names = tuple(str(value) for value in descriptors)
        unknown = [name for name in names if name not in SUPPORTED_DESCRIPTOR_NAMES]
        if unknown:
            raise ValueError(f"Unsupported descriptors: {unknown}. Supported: {list(SUPPORTED_DESCRIPTOR_NAMES)}")
        if len(set(names)) != len(names):
            raise ValueError("descriptors must not contain duplicates")
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
        self.descriptors = names
        self.register_buffer(
            "descriptor_index",
            torch.tensor([SUPPORTED_DESCRIPTOR_NAMES.index(name) for name in names], dtype=torch.long),
            persistent=False,
        )
        if names:
            self.descriptor_projection = nn.Sequential(
                nn.Linear(len(names), self._graph_dim), nn.GELU(), nn.Linear(self._graph_dim, self._graph_dim)
            )
            self.descriptor_fusion = nn.Linear(self._graph_dim * 2, self._graph_dim)
        else:
            self.descriptor_projection = None
            self.descriptor_fusion = None

    @property
    def node_dim(self) -> int:return self._node_dim
    @property
    def graph_dim(self) -> int:return self._graph_dim
    @property
    def atom_dim(self) -> int:return self.graph_builder.atom_dim
    @property
    def bond_dim(self) -> int:return self.graph_builder.bond_dim
    @property
    def symbols(self) -> Tuple[str, ...]:return self.graph_builder.symbols

    def encode_components(self, compound: Compound) -> Data:
        device = next(self.parameters()).device
        return self.graph_builder.build(compound, device=device)

    def _encode_graph(self, batch: Batch) -> tuple[Tensor, Tensor]:
        node_h, graph_h = self.encoder(batch)
        if self.descriptor_projection is None:
            return node_h, graph_h
        values = batch.descriptors.to(graph_h.dtype).index_select(1, self.descriptor_index)
        # Signed log scaling keeps counts, masses and charges in a comparable range.
        values = torch.sign(values) * torch.log1p(values.abs())
        descriptor_h = self.descriptor_projection(values)
        return node_h, self.descriptor_fusion(torch.cat((graph_h, descriptor_h), dim=-1))

    def forward(self, batch: Batch) -> Tensor:
        return self._encode_graph(batch)[1]

    def encode_with_nodes(self, batch: Batch) -> Batch:
        """Encode once, returning atom states as ``x`` and graph states as ``embeddings``."""
        node_h, graph_h = self._encode_graph(batch)
        encoded = copy(batch)
        encoded.x = node_h
        encoded.embeddings = graph_h
        return encoded

    def encode(self, compound: Compound) -> Optional[Tensor]:
        try:
            return self(Batch.from_data_list([self.encode_components(compound)]))[0]
        except Exception:
            return None

    def encode_batch(self, compounds: List[Compound]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        data_list: list[Data] = []
        indices: list[int] = []
        device = next(self.parameters()).device
        for index, compound in enumerate(compounds):
            try:
                data_list.append(self.graph_builder.build(compound, device=device));indices.append(index)
            except Exception:
                pass
        if not data_list:return None,None
        return self(Batch.from_data_list(data_list)),torch.tensor(indices,dtype=torch.long,device=device)
