"""Sparse GAT correctness and outgoing event probability contracts."""

import pytest
import torch
from torch import nn

from fredis.ml.common.layers.gat import (
    GAT,
    GATBlock,
    GATLayer,
    GATTransitionHead,
    segment_log_softmax,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_ragged_log_softmax_matches_torch_and_gradients(device):
    values = torch.tensor(
        [1000.0, -2000.0, 999.0, -1999.0, 3.0],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    groups = torch.tensor([2, 0, 2, 0, 4], device=device)
    actual = segment_log_softmax(values, groups, 6)
    expected = torch.empty_like(actual)
    for group in (0, 2, 4):
        mask = groups == group
        expected[mask] = nn.LogSoftmax(dim=0)(values[mask])
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(1, 6, device=device, dtype=values.dtype)
    grad = torch.autograd.grad((actual * weights).sum(), values, retain_graph=True)[0]
    reference = torch.autograd.grad((expected * weights).sum(), values)[0]
    torch.testing.assert_close(grad, reference)


@pytest.mark.parametrize("device", DEVICES)
def test_gat_matches_explicit_dense_attention(device):
    torch.manual_seed(3)
    model = GATLayer(4, 6, 2, directed=True, add_self_loops=False).to(device).double()
    x = torch.randn(4, 4, device=device, dtype=torch.float64)
    edges = torch.tensor([[0, 2, 1, 0], [1, 1, 2, 2]], device=device)
    attr = torch.randn(4, 2, device=device, dtype=torch.float64)
    out, (returned_edges, weights) = model(x, edges, attr, need_weights=True)
    h = model.node_projection(x)
    scores = model.leaky_relu(
        (h[edges[0]] * model.source_attention).sum(-1)
        + (h[edges[1]] * model.target_attention).sum(-1)
        + (attr * model.edge_attention).sum(-1)
    )
    expected = torch.zeros_like(h)
    for target in (1, 2):
        mask = edges[1] == target
        alpha = scores[mask].softmax(0)
        expected[target] = (
            alpha[:, None] * (h[edges[0, mask]] + model.edge_projection(attr[mask]))
        ).sum(0)
        torch.testing.assert_close(weights[mask], alpha)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(returned_edges, edges)
    assert torch.all(out[[0, 3]] == 0)


def test_self_loops_do_not_duplicate_and_edge_information_is_retained():
    model = GATLayer(4, 4, 2, directed=True)
    x = torch.randn(3, 4)
    edges = torch.tensor([[0, 0], [0, 1]])
    attr = torch.randn(2, 2)
    _, (messages, weights) = model(x, edges, attr, need_weights=True)
    assert messages.shape == (2, 4)
    assert (messages[0] == messages[1]).sum() == 3
    sums = torch.zeros(3).index_add_(0, messages[1], weights)
    torch.testing.assert_close(sums, torch.ones(3))
    single = GATLayer(4, 4, 2, directed=True, add_self_loops=False)
    original, _ = single(x, edges[:, 1:], attr[1:])
    changed, _ = single(x, edges[:, 1:], attr[1:] + 1)
    assert not torch.allclose(original[1], changed[1])


def test_message_direction_allows_parent_to_read_children():
    model = GATLayer(4, 4, 2, directed=True)
    x = torch.randn(3, 4)
    reactions = torch.tensor([[0, 0], [1, 2]])
    attr = torch.randn(2, 2)
    forward, _ = model(x, reactions, attr)
    changed = x.clone()
    changed[1] += 2
    forward_changed, _ = model(changed, reactions, attr)
    torch.testing.assert_close(forward[0], forward_changed[0])
    backward, _ = model(x, reactions.flip(0), attr)
    backward_changed, _ = model(changed, reactions.flip(0), attr)
    assert not torch.allclose(backward[0], backward_changed[0])


@pytest.mark.parametrize("device", DEVICES)
def test_transition_head_normalizes_each_parent_including_stay(device):
    torch.manual_seed(9)
    x = torch.randn(5, 8, device=device, requires_grad=True)
    edge = torch.tensor([[0, 0, 3], [1, 2, 4]], device=device)
    attr = torch.randn(3, 3, device=device, requires_grad=True)
    condition = torch.randn(5, 2, device=device, requires_grad=True)
    # Child-to-parent message edges are separate from physical output events.
    block = GATBlock(8, 3).to(device)
    head = GATTransitionHead(8, 3, condition_dim=2).to(device)
    h, _ = block(x, edge.flip(0), attr)
    events, log_probability = head(h, edge, attr, condition_h=condition)
    torch.testing.assert_close(events[:, :3], edge)
    torch.testing.assert_close(
        events[:, 3:], torch.arange(5, device=device).expand(2, -1)
    )
    totals = x.new_zeros(5).index_add_(0, events[0], log_probability.exp())
    torch.testing.assert_close(totals, torch.ones_like(totals))
    torch.testing.assert_close(
        log_probability[3 + torch.tensor([1, 2, 4], device=device)], x.new_zeros(3)
    )
    (-log_probability[0]).backward()
    for value in (x, attr, condition):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert torch.isfinite(head.stay_edge.grad).all()
    assert x.grad[1].abs().sum() > 0  # The known child's structure affects the score.
    assert block.gat.edge_projection.weight.grad.abs().sum() > 0


def test_disconnected_batch_and_edge_order_do_not_change_outputs():
    torch.manual_seed(1)
    block = GATBlock(8, 2).eval()
    head = GATTransitionHead(8, 2).eval()
    x = torch.randn(6, 8)
    edges = torch.tensor([[0, 0, 3], [1, 2, 4]])
    attr = torch.randn(3, 2)
    h, _ = block(x, edges, attr)
    alone, _ = block(x[:3], edges[:, :2], attr[:2])
    torch.testing.assert_close(h[:3], alone)
    order = torch.tensor([2, 0, 1])
    reordered, _ = block(x, edges[:, order], attr[order])
    torch.testing.assert_close(h, reordered)
    _, logs = head(h, edges, attr)
    _, alone_logs = head(alone, edges[:, :2], attr[:2])
    torch.testing.assert_close(logs[:2], alone_logs[:2])
    _, shuffled_logs = head(reordered, edges[:, order], attr[order])
    torch.testing.assert_close(shuffled_logs[:3], logs[order])
    torch.testing.assert_close(shuffled_logs[3:], logs[3:])


def test_node_permutation_equivariance():
    model = GATBlock(8, 2).eval()
    x = torch.randn(4, 8)
    edges = torch.tensor([[0, 0, 2], [1, 2, 3]])
    attr = torch.randn(3, 2)
    order = torch.tensor([2, 0, 3, 1])
    inverse = order.argsort()
    original, _ = model(x, edges, attr)
    changed, _ = model(x[order], inverse[edges], attr)
    torch.testing.assert_close(changed, original[order])


@pytest.mark.parametrize("count", [0, 3])
def test_empty_edges_and_optional_edge_features(count):
    x = torch.randn(count, 4)
    edges = torch.empty((2, 0), dtype=torch.long)
    out, attention = GATBlock(4, 0)(x, edges)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert attention is None
    events, logs = GATTransitionHead(4, 0)(out, edges)
    assert events.shape == (2, count)
    torch.testing.assert_close(logs, torch.zeros(count))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_log_softmax_is_finite(dtype):
    values = torch.tensor([1000, 999, -1000, -999], dtype=dtype, requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1])
    result = segment_log_softmax(values, groups, 2)
    expected = torch.cat(
        (values[:2].float().log_softmax(0), values[2:].float().log_softmax(0))
    )
    torch.testing.assert_close(result.float(), expected, atol=0.01, rtol=0.01)
    result.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_required_shapes_and_conditions_are_checked():
    x = torch.randn(3, 4)
    edges = torch.tensor([[0], [1]])
    with pytest.raises(ValueError, match="edge_attr is required"):
        GATLayer(4, 4, 2)(x, edges)
    with pytest.raises(ValueError, match="condition_h is required"):
        GATTransitionHead(4, 0, condition_dim=2)(x, edges)
    with pytest.raises(ValueError, match="edge_attr must have shape"):
        GATLayer(4, 4, 2)(x, edges, torch.randn(1, 3))


def test_undirected_layer_appends_reverse_edges_and_preserves_attributes():
    model = GATLayer(4, 4, 2, directed=False, add_self_loops=False)
    x = torch.randn(3, 4)
    edges = torch.tensor([[0, 1], [1, 2]])
    attr = torch.randn(2, 2)
    out, (messages, weights) = model(x, edges, attr, need_weights=True)
    directed = GATLayer(4, 4, 2, directed=True, add_self_loops=False)
    directed.load_state_dict(model.state_dict())
    expected_edges = torch.cat((edges, edges.flip(0)), 1)
    expected, (_, reference) = directed(
        x, expected_edges, attr.repeat(2, 1), need_weights=True
    )
    torch.testing.assert_close(messages, expected_edges)
    torch.testing.assert_close(weights, reference)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_graph_softmax_and_temperature_match_torch(device):
    torch.manual_seed(5)
    x = torch.randn(6, 4, device=device)
    edge = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], device=device)
    attr = torch.randn(4, 2, device=device)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    model = GATLayer(4, 4, 2, directed=True, softmax_per="graph", T=0.3).to(device)
    out, (messages, weights) = model(x, edge, attr, batch=batch, need_weights=True)
    h = model.node_projection(x)
    extended_attr = torch.cat((attr, attr.new_zeros((6, 2))))
    scores = (
        model.leaky_relu(
            (h[messages[0]] * model.source_attention).sum(-1)
            + (h[messages[1]] * model.target_attention).sum(-1)
            + (extended_attr * model.edge_attention).sum(-1)
        )
        / 0.3
    )
    for graph in (0, 1):
        mask = batch[messages[1]] == graph
        torch.testing.assert_close(
            weights[mask], nn.LogSoftmax(dim=0)(scores[mask]).exp()
        )
    alone, (_, alone_weights) = model(x[:3], edge[:, :2], attr[:2], need_weights=True)
    torch.testing.assert_close(out[:3], alone)
    torch.testing.assert_close(weights[batch[messages[1]] == 0], alone_weights)
    # Temperature changes the distribution without changing event connectivity.
    model.T = 2.0
    _, (_, warmer) = model(x, edge, attr, batch=batch, need_weights=True)
    assert not torch.allclose(weights, warmer)


@pytest.mark.parametrize("device", DEVICES)
def test_stack_repeats_full_residual_ffn_blocks_with_independent_parameters(device):
    torch.manual_seed(2)
    model = (
        GAT(8, 2, num_layers=3, ffn_hidden_mult=2, directed=False, T=0.7)
        .to(device)
        .eval()
    )
    x = torch.randn(4, 8, device=device, requires_grad=True)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device)
    attr = torch.randn(3, 2, device=device, requires_grad=True)
    actual, weights = model(x, edges, attr, need_weights=True)
    expected = x
    assert len(weights) == 3
    for layer in model.layers:
        assert layer.feed_forward.linear_1.out_features == 16
        assert layer.gat.directed is False
        assert layer.gat.T == 0.7
        expected, _ = layer(expected, edges, attr)
    torch.testing.assert_close(actual, expected)
    assert (
        model.layers[0].gat.node_projection.weight
        is not model.layers[1].gat.node_projection.weight
    )
    actual.square().sum().backward()
    for layer in model.layers:
        for component in (layer.gat, layer.feed_forward, layer.norm1, layer.norm2):
            assert any(
                p.grad is not None
                and torch.isfinite(p.grad).all()
                and p.grad.abs().sum() > 0
                for p in component.parameters()
            )
    assert model(x.detach(), edges, attr.detach())[1] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"softmax_per": "src"},
        {"T": 0},
        {"T": -1},
        {"T": float("nan")},
        {"T": float("inf")},
    ],
)
def test_invalid_gat_attention_settings(kwargs):
    with pytest.raises(ValueError):
        GATLayer(4, 4, 2, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gat_autocast_backpropagates():
    model = GAT(8, 2, num_layers=2).cuda()
    x = torch.randn(3, 8, device="cuda", requires_grad=True)
    edge = torch.tensor([[0, 0], [1, 2]], device="cuda")
    attr = torch.randn(2, 2, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out, _ = model(x, edge, attr)
        loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(attr.grad).all()
