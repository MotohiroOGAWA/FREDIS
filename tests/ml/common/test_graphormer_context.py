import pytest
import torch
from torch_geometric.data import Batch, Data

from fredis.ml.common.layers.graphormer import GraphormerEncoder


def graphs():
    return Batch.from_data_list([
        Data(x=torch.randn(n, 4), edge_index=torch.tensor([[0, 1], [1, 0]]))
        for n in (2, 5)])


@pytest.mark.parametrize('context_dim, tokens', [(0, 0), (3, 1), (8, 1), (11, 2), (24, 3)])
def test_context_width_drives_token_count_and_gradients(context_dim, tokens):
    model = GraphormerEncoder(4, 8, num_heads=2, graph_dim=11, context_dim=context_dim, start_cap=2)
    assert model.context_token_count == tokens
    context = torch.randn(2, context_dim, requires_grad=True) if context_dim else None
    nodes, output = model(graphs(), context_repr=context)
    assert nodes.shape == (7, 8) and output.shape == (2, 11)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    if context is not None:
        assert torch.all(context.grad.abs().sum(-1) > 0)
        assert model.blocks[0].cross_attention.out_proj.weight.grad is not None
    else:
        assert all(block.cross_attention is None for block in model.blocks)


def test_context_affects_graphs_and_preserves_batch_isolation():
    model = GraphormerEncoder(4, 8, num_heads=2, graph_dim=11, context_dim=11, start_cap=2).eval()
    batch = graphs()
    context = torch.randn(2, 11)
    nodes, output = model(batch, context)
    changed = context.clone(); changed[0] += torch.randn(11)
    other_nodes, other_output = model(batch, changed)
    assert not torch.equal(nodes[:2], other_nodes[:2])
    assert not torch.equal(output[0], other_output[0])
    torch.testing.assert_close(nodes[2:], other_nodes[2:])
    torch.testing.assert_close(output[1], other_output[1])
    # Self-attention sees graph/node tokens only. Cross-attention gets separate
    # unchanged context memory; the two softmax domains have different lengths.
    projected = model._context_repr_to_tokens(context)
    captures = []
    self_lengths = []
    hooks = []
    for block in model.blocks:
        hooks.append(block.mha.register_forward_pre_hook(
            lambda module, args: self_lengths.append(args[0].shape[1])))
        hooks.append(block.cross_attention.register_forward_pre_hook(
            lambda module, args: captures.append((args[0].shape[1], args[1].detach().clone()))))
    model(batch, context)
    for hook in hooks:
        hook.remove()
    assert self_lengths == [4, 4, 10, 10]  # two graph tokens + bucket cap
    for bucket_index, expected in enumerate(projected):
        for query_length, memory in captures[bucket_index * 2:bucket_index * 2 + 2]:
            assert query_length == self_lengths[bucket_index * 2]
            torch.testing.assert_close(memory[0], expected)


def test_context_updates_both_graph_and_node_queries_in_one_layer():
    model = GraphormerEncoder(4, 8, num_heads=2, num_layers=1, graph_dim=11, context_dim=11)
    batch = graphs()
    context = torch.randn(2, 11)
    nodes, output = model(batch, context)
    other_nodes, other_output = model(batch, context + torch.randn_like(context))
    assert torch.all((nodes - other_nodes).abs().sum(-1) > 0)
    assert torch.all((output - other_output).abs().sum(-1) > 0)



def test_context_input_errors():
    with pytest.raises(ValueError, match='context_dim'):
        GraphormerEncoder(4, 8, num_heads=2, context_dim=-1)
    model = GraphormerEncoder(4, 8, num_heads=2, context_dim=11)
    for context, message in [(None, 'required'), (torch.randn(2, 10), 'feature size'),
                             (torch.randn(1, 11), 'batch size'), (torch.randn(2, 2, 11), 'shape')]:
        with pytest.raises(ValueError, match=message):
            model(graphs(), context)
    with pytest.raises(ValueError, match='context_dim is 0'):
        GraphormerEncoder(4, 8, num_heads=2)(graphs(), torch.randn(2, 11))
