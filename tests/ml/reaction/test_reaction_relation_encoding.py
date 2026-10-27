"""Reaction memory uses distances, chemical relations and nearest SMARTS positions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fredis.ml.specgen.components.reaction.reaction_instance_encoder import ReactionInstanceEncoder
from fredis.ml.specgen.components.reaction.reaction_transformer import (
    ReactionRelationEncoding, ReactionMemoryEncoder, reaction_geometry,
)
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.training.fragment_tree_training.fine_tuning import initialize_from_base
from .TestReactionPipeline import config
from .test_canonical_workflow import prepared


def test_different_smarts_sizes_need_no_connection_role_configuration():
    encoder = ReactionInstanceEncoder(3, 8, (2, 1, 1), num_heads=2,
                                     pattern_smarts={0: 'CC', 1: 'CCCCCC'})
    assert encoder.reactant_encoders['0'].reactant_projection.in_features == 3 + 2
    assert encoder.reactant_encoders['1'].reactant_projection.in_features == 3 + 6
    assert not any('role_encoder' in key or 'role_projections' in key for key in encoder.state_dict())
    generator = create_spectrum_generator(config())
    assert not any('role_encoder' in key or 'role_projections' in key or 'relative_encoding.roles' in key
                   for key in generator.state_dict())


def test_tied_nearest_connections_preserve_all_nearest_smarts_positions():
    registry = SimpleNamespace(
        reaction_type=torch.zeros((2, 3), dtype=torch.long),
        reaction_source_compound_index=torch.tensor([0, 0]),
        reaction_source_atom_ptr=torch.tensor([0, 2, 4]),
        reaction_source_atom_index=torch.tensor([0, 3, 1, 2]),
        reaction_retained_atom_ptr=torch.tensor([0, 2, 4]),
        reaction_retained_atom_index=torch.tensor([0, 3, 1, 2]),
        reaction_conflict_index=torch.empty((2, 0), dtype=torch.long),
        reaction_invalidation_index=torch.empty((2, 0), dtype=torch.long),
    )
    graph = SimpleNamespace(ptr=torch.tensor([0, 4]), num_nodes=4,
                            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]))
    geometry = reaction_geometry(SimpleNamespace(reactions=registry, source_graph=graph))
    assert len(geometry) == 4
    ids, distance, relations, endpoints = geometry
    assert ids.shape == (1, 2) and relations.shape == (1, 2, 2, 4)
    assert distance[0, 0, 1] == distance[0, 1, 0] == 1
    pair, position = endpoints
    assert set(position[pair == 1].tolist()) == {2, 3}
    assert set(position[pair == 2].tolist()) == {0, 1}


def test_distance_and_chemical_relations_still_affect_attention_and_values():
    torch.manual_seed(3)
    encoding = ReactionRelationEncoding(8, 2)
    q, k = torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4)
    distance = torch.ones((1, 2, 2), dtype=torch.long)
    relations = torch.zeros((1, 2, 2, 4))
    baseline = encoding.attention_bias(q, k, (distance, relations))
    changed_distance = distance.clone(); changed_distance[:, 0, 1] = 2
    assert not torch.allclose(baseline, encoding.attention_bias(q, k, (changed_distance, relations)))
    changed_relations = relations.clone(); changed_relations[:, 0, 1, 0] = 1
    assert not torch.allclose(baseline, encoding.attention_bias(q, k, (distance, changed_relations)))
    attention = baseline.softmax(-1)
    values = encoding.value_encoding(attention, (distance, relations))
    changed_values = encoding.value_encoding(attention, (distance, changed_relations))
    assert not torch.allclose(values, changed_values)
    assert not hasattr(encoding, 'roles')
    memory = ReactionMemoryEncoder(8, heads=2).eval()
    out = memory(torch.randn(1, 2, 8), distance, relations, torch.ones(1, 2, dtype=torch.bool))
    assert out.shape == (1, 2, 8) and torch.isfinite(out).all()


def test_fine_tuning_restores_features_without_role_adapters(tmp_path):
    source, data = prepared()
    source.eval()
    path = tmp_path / 'source.pt'
    torch.save({'model_state_dict': source.state_dict()}, path)
    target_config = deepcopy(config())
    patterns = list(source.feature_model.reaction_instance_encoder.reactant_encoders)
    target_config['fine_tuning'] = dict(
        source_checkpoint=str(path), pattern_mapping=[[int(pid), int(pid)] for pid in patterns],
        frozen_pattern_ids=patterns,
    )
    target = create_spectrum_generator(target_config).eval()
    initialize_from_base(target, target_config)
    assert not any('role_encoder' in name or 'relative_encoding.roles' in name for name in target.state_dict())
    before, after = source.feature_model(data), target.feature_model(data)
    torch.testing.assert_close(before.reaction_h, after.reaction_h)
    torch.testing.assert_close(before.state_h, after.state_h)
    for module in target.feature_model.reaction_instance_encoder.reactant_encoders.values():
        assert not any(parameter.requires_grad for parameter in module.parameters())


def test_removed_cli_option_is_rejected(tmp_path):
    from fredis.ml.training.fragment_tree_training.training import build_arg_parser, training_model_config
    parser = build_arg_parser()
    args = ['--dataset-dir', str(tmp_path/'data'), '--output-dir', str(tmp_path/'run')]
    settings = training_model_config(parser.parse_args(args))
    assert 'role_dim' not in settings['reaction_model_params']
    with pytest.raises(SystemExit):
        parser.parse_args(args + ['--reaction-role-dim', '4'])


def test_one_hot_position_projections_use_hidden_width_and_sum_every_shortest_tie():
    torch.manual_seed(13)
    encoder = ReactionInstanceEncoder(3, 8, (2, 1, 1), num_heads=2,
                                     pattern_smarts={0: 'CC', 1: 'NN'})
    types = torch.tensor([[0, 0, 0], [1, 0, 0]])
    ptr = torch.tensor([0, 2, 4])
    positions = encoder.encode_endpoint_positions(types, ptr)
    expected = torch.cat([encoder.endpoint_projections[str(pid)](torch.eye(2)) for pid in (0, 1)])
    torch.testing.assert_close(positions, expected)
    assert positions.shape == (4, 8)
    # A is at atoms {0,3}; B is at {1,2}. Both ends of each match are tied nearest.
    endpoints = torch.tensor([[1, 1, 2, 2], [2, 3, 0, 1]])
    relation = ReactionRelationEncoding(8, 2)
    q, k = torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4)
    bias = relation.endpoint_attention_bias(q, k, positions, endpoints)
    target_sum = (expected[2] + expected[3]).reshape(2, 4)
    source_sum = (expected[0] + expected[1]).reshape(2, 4)
    wanted = (q[0, :, 0] * target_sum).sum(-1) + (k[0, :, 1] * source_sum).sum(-1)
    torch.testing.assert_close(bias[0, :, 0, 1], wanted)
    assert not torch.allclose(bias[0, :, 0, 1], wanted / 2)
    attention = torch.zeros(1, 2, 2, 2)
    attention[:, :, 0, 1] = 1
    values = relation.endpoint_value_encoding(attention, positions, endpoints)
    torch.testing.assert_close(values[0, :, 0], target_sum + source_sum)
    assert not values[0, :, 1].any()
    (bias * torch.randn_like(bias)).sum().add((values * torch.randn_like(values)).sum()).backward()
    assert all(projection.weight.grad.abs().sum() > 0 for projection in encoder.endpoint_projections.values())


def test_endpoint_features_affect_memory_and_are_permutation_equivariant():
    torch.manual_seed(14)
    model = ReactionMemoryEncoder(8, heads=2).eval()
    x = torch.randn(1, 2, 8)
    distance = torch.ones(1, 2, 2, dtype=torch.long)
    relations = torch.zeros(1, 2, 2, 4)
    valid = torch.ones(1, 2, dtype=torch.bool)
    positions = torch.randn(4, 8, requires_grad=True)
    endpoints = torch.tensor([[1, 1, 2, 2], [2, 3, 0, 1]])
    output = model(x, distance, relations, valid, positions, endpoints)
    changed = positions.detach().clone(); changed[2] += torch.arange(8.)
    assert not torch.allclose(output, model(x, distance, relations, valid, changed, endpoints))
    order = torch.tensor([1, 0]); inverse = order.argsort()
    source = endpoints[0] // 2; target = endpoints[0] % 2
    permuted_endpoints = torch.stack((inverse[source] * 2 + inverse[target], endpoints[1]))
    permuted = model(x[:, order], distance[:, order][:, :, order], relations[:, order][:, :, order],
                     valid[:, order], positions, permuted_endpoints)
    torch.testing.assert_close(permuted, output[:, order])
    (output * torch.randn_like(output)).sum().backward()
    assert positions.grad.abs().sum() > 0
