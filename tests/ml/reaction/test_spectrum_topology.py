"""GPU topology packing, reusable propagation indices and batch isolation."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch, Data

from fredis.ml.common.layers.graphormer import GraphormerEncoder
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.specgen.competitive_spectrum_model import propagate_log_probability, segment_log_softmax
from fredis.ml.specgen.spectrum_topology import canonical_spectrum_topology, probability_levels
from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel
from .test_canonical_workflow import prepared

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_topology_preserves_multiple_sources_and_reordered_duplicate_samples(device):
    _, original = prepared()
    structure = SourceReactionStructure.from_structures([original, original]).select_samples([3, 0, 3])
    batch = structure.to(device).spectrum_batch
    actual = canonical_spectrum_topology(batch)
    core = structure.for_prediction()
    execution = core.execution_graph
    rows, samples, roots, source, target, reactions = [], [], [], [], [], []
    physical = []
    for sample, group in enumerate(core.sample_branch_group_index.tolist()):
        members = (execution.branch_group_index == group).nonzero().flatten().tolist()
        remap = {old: len(rows) + index for index, old in enumerate(members)}
        roots.append(remap[int(execution.root_index[group])])
        rows.extend(members)
        samples.extend([sample] * len(members))
        physical.extend((sample, node) for node in sorted(set(execution.compound_index[members].tolist())))
        for parent, child, reaction in zip(execution.source_index.tolist(),
                                           execution.target_index.tolist(), execution.reaction_index.tolist()):
            if parent in remap:
                source.append(remap[parent]); target.append(remap[child]); reactions.append(reaction)
    for index in range(len(rows)):
        source.extend([index, index]); target.extend([index, -1]); reactions.extend([-1, -2])
    for name, expected in [('execution_rows', rows), ('samples', samples), ('roots', roots),
                           ('source', source), ('target', target), ('reactions', reactions), ('fragments', physical)]:
        assert getattr(actual, name).cpu().tolist() == [list(v) if isinstance(v, tuple) else v for v in expected]
    torch.testing.assert_close(actual.fragments[actual.state_fragment],
                               torch.stack((batch.state_sample_index, batch.state_node_index), 1))


@pytest.mark.parametrize('device', DEVICES)
def test_cached_levels_preserve_merged_history_probabilities_and_gradients(device):
    source = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=device)
    target = torch.tensor([1, 2, 0, 2, 1, 2, -1, 2], device=device)
    initial = torch.tensor([0., -torch.inf, -torch.inf], device=device)
    levels = probability_levels(source, target, 3)
    first = torch.randn(8, device=device, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    expected = propagate_log_probability(initial, source, target, segment_log_softmax(first, source, 3))
    actual = propagate_log_probability(initial, source, target, segment_log_softmax(second, source, 3), levels=levels)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)
    sum(value.exp().sum() for value in expected).backward()
    sum(value.exp().sum() for value in actual).backward()
    torch.testing.assert_close(first.grad, second.grad)


@pytest.mark.parametrize('device', DEVICES)
def test_graph_buckets_keep_outputs_and_gradients_without_unpacking_pyg_batch(device):
    torch.manual_seed(71)
    graphs = []
    for size in [2, 9, 1, 5]:
        source = torch.arange(size - 1)
        edges = torch.stack((torch.cat((source, source + 1)), torch.cat((source + 1, source))))
        graphs.append(Data(x=torch.randn(size, 4), edge_index=edges, edge_attr=torch.randn(edges.shape[1], 2)))
    batch = Batch.from_data_list(graphs).to(device)
    batch.x.requires_grad_()
    model = GraphormerEncoder(4, 8, edge_dim=2, num_heads=2, num_layers=1,
                              graph_dim=11, start_cap=2, cap_growth=2).to(device).eval()
    with patch.object(Batch, 'to_data_list', side_effect=AssertionError('GPU graph unpacking')):
        nodes, encoded = model(batch)
    gradients = torch.autograd.grad(encoded.square().sum() + nodes.square().sum(), batch.x)[0]
    expected_nodes, expected_graphs, expected_gradients = [], [], []
    for graph in graphs:
        graph = graph.to(device)
        graph.x.requires_grad_()
        h, output = model(graph)
        expected_nodes.append(h); expected_graphs.append(output)
        expected_gradients.append(torch.autograd.grad(output.square().sum() + h.square().sum(), graph.x)[0])
    torch.testing.assert_close(nodes, torch.cat(expected_nodes), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(encoded, torch.cat(expected_graphs), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(gradients, torch.cat(expected_gradients), rtol=2e-4, atol=3e-5)


def test_static_batch_caches_do_not_retain_autograd_or_cross_replaced_inputs():
    generator, structure = prepared()
    trainer = ReactionFragmentTreeTrainingModel(generator.feature_model, spectrum_model=generator.spectrum_model).eval()
    with patch('fredis.ml.specgen.competitive_spectrum_model.canonical_spectrum_topology',
               wraps=canonical_spectrum_topology) as build:
        for _ in range(2):
            trainer.zero_grad(set_to_none=True)
            result = trainer(structure)
            result.loss.backward()
            assert torch.isfinite(result.loss)
        assert build.call_count == 1
        changed = replace(structure, conditions=replace(structure.conditions,
                          collision_energy_eV=structure.collision_energy_eV + 5))
        trainer(changed).loss.backward()
        assert build.call_count == 2
    assert structure.for_prediction().execution_graph is structure.execution_graph
    assert 'spectrum_batch' not in structure.__getstate__()
