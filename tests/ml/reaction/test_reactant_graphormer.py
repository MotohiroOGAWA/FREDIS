import pytest
import torch
from torch_geometric.data import Data

from fredis.ml.common.layers.graphormer import GraphormerEncoder
from fredis.ml.mol.mol_graphormer import MolGraphormerEncoder
from fredis.ml.reaction import ReactantGraphormerEncoder
from fredis.ml.specgen.components.reaction.reaction_instance_encoder import ReactionInstanceEncoder


@pytest.mark.parametrize('graph_dim, tokens', [(5, 1), (8, 1), (11, 2), (24, 3)])
def test_graph_dim_controls_tokens_and_linear_readout(graph_dim, tokens):
    model = GraphormerEncoder(node_dim=4, hidden_dim=8, graph_dim=graph_dim, num_heads=2)
    graph = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1], [1, 0]]))
    nodes, output = model(graph)
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
    model = MolGraphormerEncoder(atom_dim=4, bond_dim=2, node_dim=8, graph_dim=11, num_heads=2, num_layers=1)
    _, output = model(Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2, 2)))
    assert output.shape == (1, 11)
    assert model.graph_token_count == 2


@pytest.mark.parametrize('smarts, atoms', [('[C:1]-[C:2]', 2), ('O=c1cc(-c2ccccc2)oc2ccccc12', 17), ('[C:1].[O:2]', 2)])
def test_smarts_capacity_and_batched_match_gradients(smarts, atoms):
    model = ReactantGraphormerEncoder(smarts, atom_dim=4, bond_dim=0, hidden_dim=8, graph_dim=11, num_layers=1, num_heads=2)
    assert model.start_cap == model.position_count == atoms
    x = torch.randn(3, atoms, 4, requires_grad=True)
    nodes, graphs = model.encode_matches(x)
    assert nodes.shape == (3, atoms, 8)
    assert graphs.shape == (3, 11)
    torch.testing.assert_close(graphs, torch.cat([model.encode_matches(row[None])[1] for row in x]))
    graphs.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert not hasattr(model,"positions")
    assert model.node_proj.in_features == 4 + atoms
    assert model.node_proj.weight.grad[:, -atoms:].abs().sum() > 0
    with pytest.raises(ValueError, match='SMARTS atom count'):
        model(Data(x=torch.randn(atoms + 1, 4), edge_index=torch.empty((2, 0), dtype=torch.long)))




@pytest.mark.parametrize('node_dim, graph_dim', [(8, 8), (4, 11)])
def test_training_encoder_routes_mixed_patterns_and_restores_reaction_order(node_dim, graph_dim):
    torch.manual_seed(5)
    model = ReactionInstanceEncoder(4, 6, 8, (2, 1, 1), num_heads=2,
                                    pattern_smarts={0: 'CC', 1: 'CCC'},
                                    reactant_node_dim=node_dim, reactant_graph_dim=graph_dim)
    atoms = torch.randn(7, 4, requires_grad=True)
    kwargs = dict(source_mol_h=torch.randn(1, 6),
                  reaction_type=torch.tensor([[1, 0, 0], [0, 0, 0], [1, 0, 0]]),
                  reaction_tree_index=torch.zeros(3, dtype=torch.long),
                  reaction_source_atom_ptr=torch.tensor([0, 3, 5, 8]),
                  reaction_source_atom_index=torch.tensor([0, 1, 2, 3, 4, 4, 5, 6]),
                  reaction_static_features=torch.randn(3, 6), source_atom_features=atoms, source_edge_index=torch.empty((2,0),dtype=torch.long), source_edge_attr=torch.empty((0,0)))
    output = model(**kwargs)
    assert output.shape == (3, 8)
    # Each match is equivalent to processing it alone, even in a mixed-pattern batch.
    for row in range(3):
        start, end = kwargs['reaction_source_atom_ptr'][row:row+2]
        one = {**kwargs, 'reaction_type': kwargs['reaction_type'][row:row+1],
               'reaction_tree_index': torch.zeros(1, dtype=torch.long),
               'reaction_source_atom_ptr': torch.tensor([0, end-start]),
               'reaction_source_atom_index': kwargs['reaction_source_atom_index'][start:end],
               'reaction_static_features': kwargs['reaction_static_features'][row:row+1]}
        torch.testing.assert_close(output[row:row+1], model(**one))
    output.square().sum().backward()
    for encoder in model.reactant_encoders.values():
        assert encoder.graph_tokens.grad is not None
        assert encoder.blocks[0].mha.parameters().__next__().grad is not None
    assert atoms.grad is not None


def test_fine_tuning_remaps_existing_pattern_encoder_and_trains_new_pattern(tmp_path):
    from torch import nn
    from fredis.ml.specgen.fine_tuning import configure_pattern_training
    from fredis.ml.training.fragment_tree_training.fine_tuning import initialize_from_base

    class Feature(nn.Module):
        def __init__(self, smarts):
            super().__init__()
            self.reaction_instance_encoder = ReactionInstanceEncoder(
                4, 6, 8, (len(smarts), 1, 1), num_heads=2, pattern_smarts=smarts)

        def freeze_mol_encoder(self):
            pass

    class Generator(nn.Module):
        def __init__(self, smarts):
            super().__init__()
            self.mol_encoder = nn.Linear(4, 4)
            self.feature_model = Feature(smarts)

    old = Generator({0: 'CC'})
    checkpoint = tmp_path / 'base.pt'
    torch.save({'model_state_dict': old.state_dict()}, checkpoint)
    new = Generator({0: 'CCC', 1: 'CC'})
    config = {'version': 1, 'frozen_pattern_ids': [1], 'pattern_mapping': [[0, 1]], 'source_checkpoint': str(checkpoint),
              }
    configure_pattern_training(new, config)
    initialize_from_base(new, {'fine_tuning': config})
    before = old.feature_model.reaction_instance_encoder.reactant_encoders['0']
    inherited = new.feature_model.reaction_instance_encoder.reactant_encoders['1']
    added = new.feature_model.reaction_instance_encoder.reactant_encoders['0']
    torch.testing.assert_close(before.graph_tokens, inherited.graph_tokens)
    assert not inherited.graph_tokens.requires_grad
    assert all(parameter.requires_grad for parameter in added.parameters())
    # These encoders receive hidden-width source features in the training path.
    x = torch.randn(1, 2, before.atom_dim)
    torch.testing.assert_close(before.encode_matches(x)[1], inherited.encode_matches(x)[1])
