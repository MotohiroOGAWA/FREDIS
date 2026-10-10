from __future__ import annotations

from torch import Tensor
from torch_geometric.data import Batch, Data

from ..common.layers.graphormer import GraphormerEncoder

DEFAULT_MOL_GRAPHORMER_MAX_DEGREE = 4
DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST = 3
DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST = 3
DEFAULT_MOL_GRAPHORMER_DROPOUT = 0.0


class MolGraphormerEncoder(GraphormerEncoder):
    """Graphormer specialization for molecular graphs.

    The public constructor only exposes molecule-relevant dimensions and
    graph-structure options. Condition tokens are intentionally not part of
    this API; molecules are encoded without sample-level context such as
    collision energy or adduct type.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int,
        node_dim: int,
        graph_dim: int,
        num_layers: int,
        num_heads: int,
        max_degree: int = DEFAULT_MOL_GRAPHORMER_MAX_DEGREE,
        max_spatial_dist: int = DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
        max_edge_dist: int = DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST,
        dropout: float = DEFAULT_MOL_GRAPHORMER_DROPOUT,
        start_cap: int = 64,
        cap_growth: float = 2.0,
    ) -> None:
        node_dim = int(node_dim)
        graph_dim = int(graph_dim)
        if graph_dim <= 0 or graph_dim % node_dim != 0:
            raise ValueError(
                "graph_dim must be a positive integer multiple of node_dim."
            )

        atom_feature_dim = int(atom_dim)
        bond_feature_dim = int(bond_dim)
        graph_token_count = graph_dim // node_dim

        super().__init__(
            node_dim=atom_feature_dim,
            hidden_dim=node_dim,
            edge_dim=bond_feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_degree=max_degree,
            max_spatial_dist=max_spatial_dist,
            max_edge_dist=max_edge_dist,
            graph_token_count=graph_token_count,
            dropout=dropout,
            graph_to_node=True,
            undirected_for_spd=True,
            undirected_for_path=True,
            start_cap=start_cap,
            cap_growth=cap_growth,
        )

        self.atom_feature_dim = atom_feature_dim
        self.bond_feature_dim = bond_feature_dim
        self.mol_node_dim = node_dim
        self.mol_graph_dim = graph_dim
        self.mol_graph_token_count = graph_token_count

    def forward(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        return super().forward(data)
