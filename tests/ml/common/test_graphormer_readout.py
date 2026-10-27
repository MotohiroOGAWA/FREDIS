import pytest
import torch
from torch_geometric.data import Data

from fredis.ml.common.layers.graphormer import GraphormerEncoder
from fredis.ml.mol.mol_graphormer import MolGraphormerEncoder


@pytest.mark.parametrize("graph_dim, tokens", [(5, 1), (8, 1), (11, 2), (24, 3)])
def test_graph_dim_controls_tokens_and_linear_readout(graph_dim, tokens):
    model = GraphormerEncoder(
        node_dim=4, hidden_dim=8, graph_dim=graph_dim, num_heads=2
    )
    graph = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1], [1, 0]]))
    nodes, output = model(graph)
    assert model.max_degree == 4
    assert nodes.shape == (2, 8)
    assert output.shape == (1, graph_dim)
    assert model.graph_tokens.shape == (tokens, 8)
    assert isinstance(model.graph_projection, torch.nn.Linear)
    if tokens > 1:
        assert not torch.equal(model.graph_tokens[0], model.graph_tokens[1])
    output.square().sum().backward()
    assert torch.all(model.graph_tokens.grad.abs().sum(-1) > 0)
    assert model.graph_projection.weight.grad is not None


def test_molecule_graph_dim_need_not_be_a_multiple_of_node_dim():
    model = MolGraphormerEncoder(
        atom_dim=4, bond_dim=2, node_dim=8, graph_dim=11, num_heads=2, num_layers=1
    )
    _, output = model(
        Data(
            x=torch.randn(2, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.randn(2, 2),
        )
    )
    assert output.shape == (1, 11)
    assert model.graph_token_count == 2


