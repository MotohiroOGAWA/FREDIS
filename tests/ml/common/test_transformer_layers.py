import pytest
import torch

from fredis.ml.common.layers.transformer_decoder import TransformerDecoder
from fredis.ml.common.layers.transformer_encoder import TransformerEncoder


def padding(lengths, width):
    return torch.arange(width)[None, :] >= torch.tensor(lengths)[:, None]


def test_encoder_shapes_weights_and_zeroed_padding():
    torch.manual_seed(0)
    model = TransformerEncoder(8, 2, 2).eval()
    x = torch.randn(2, 4, 8)
    mask = padding([4, 2], 4)
    out, weights = model(x, key_padding_mask=mask, need_weights=True)
    assert out.shape == (2, 4, 8) and torch.all(out[mask] == 0)
    assert len(weights) == 2 and weights[0].shape == (2, 2, 4, 4)
    assert torch.allclose(weights[0].sum(-1).transpose(0, 1)[:, ~mask], torch.ones(1))
    assert torch.all(weights[1][1, :, :, 2:] == 0)
    assert model(x, key_padding_mask=mask)[1] is None


def test_encoder_is_permutation_equivariant_and_padding_invariant():
    torch.manual_seed(0)
    model = TransformerEncoder(8, 2, 2).eval()
    x = torch.randn(1, 5, 8)
    order = torch.tensor([3, 0, 4, 1, 2])
    out, _ = model(x)
    permuted, _ = model(x[:, order])
    assert torch.allclose(permuted, out[:, order], atol=1e-5)
    padded = torch.cat((x, torch.randn(1, 3, 8)), 1)
    out_padded, _ = model(padded, key_padding_mask=padding([5], 8))
    assert torch.allclose(out_padded[:, :5], out, atol=1e-5)


def test_batch_rows_do_not_interact():
    torch.manual_seed(0)
    model = TransformerEncoder(8, 2, 1).eval()
    x = torch.randn(2, 3, 8)
    together, _ = model(x)
    alone, _ = model(x[:1])
    assert torch.allclose(together[:1], alone, atol=1e-5)


def test_decoder_shapes_masks_and_weights():
    torch.manual_seed(0)
    model = TransformerDecoder(8, 2, 2).eval()
    query, memory = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    query_mask, memory_padding = padding([3, 1], 3), padding([5, 4], 5)
    allowed = torch.ones(2, 3, 5, dtype=torch.bool)
    allowed[0, 0, 1] = False
    out, weights = model(
        query,
        memory,
        query_padding_mask=query_mask,
        memory_key_padding_mask=memory_padding,
        memory_mask=allowed,
        need_weights=True,
    )
    assert out.shape == (2, 3, 8) and torch.all(out[query_mask] == 0)
    assert len(weights.self_attention) == len(weights.cross_attention) == 2
    assert weights.self_attention[0].shape == (2, 2, 3, 3)
    cross = weights.cross_attention[1]
    assert cross.shape == (2, 2, 3, 5)
    assert torch.all(cross[0, :, 0, 1] == 0) and torch.all(cross[1, :, :, 4] == 0)
    assert torch.allclose(cross.sum(-1)[:, :, 0], torch.ones(1))


def test_decoder_is_equivariant_over_queries_and_invariant_over_memory():
    torch.manual_seed(0)
    model = TransformerDecoder(8, 2, 2).eval()
    query, memory = torch.randn(1, 3, 8), torch.randn(1, 6, 8)
    out, _ = model(query, memory)
    q_order, m_order = torch.tensor([2, 0, 1]), torch.tensor([5, 3, 1, 0, 2, 4])
    permuted, _ = model(query[:, q_order], memory[:, m_order])
    assert torch.allclose(permuted, out[:, q_order], atol=1e-5)
    padded_memory = torch.cat((memory, torch.randn(1, 2, 8)), 1)
    padded, _ = model(query, padded_memory, memory_key_padding_mask=padding([6], 8))
    assert torch.allclose(padded, out, atol=1e-5)


def test_fully_masked_valid_queries_are_rejected():
    model = TransformerDecoder(8, 2, 1)
    query, memory = torch.randn(1, 2, 8), torch.randn(1, 3, 8)
    allowed = torch.ones(1, 2, 3, dtype=torch.bool)
    allowed[0, 1] = False
    with pytest.raises(ValueError, match="query 1 of batch row 0"):
        model(query, memory, memory_mask=allowed)
    # The same row is accepted when that query is padding.
    model(query, memory, memory_mask=allowed, query_padding_mask=padding([1], 2))
    blocked = torch.ones(1, 2, 2, dtype=torch.bool)
    blocked[0, 0] = False
    with pytest.raises(ValueError, match="no attendable keys"):
        TransformerEncoder(8, 2, 1)(torch.randn(1, 2, 8), attn_mask=blocked)
    # A row made only of padding has no valid queries and is accepted.
    TransformerEncoder(8, 2, 1)(torch.randn(1, 2, 8), key_padding_mask=padding([0], 2))


def test_gradients_reach_queries_and_memory():
    model = TransformerDecoder(8, 2, 1)
    query = torch.randn(2, 3, 8, requires_grad=True)
    memory = torch.randn(2, 4, 8, requires_grad=True)
    out, _ = model(query, memory, memory_key_padding_mask=padding([4, 2], 4))
    out.square().sum().backward()
    assert query.grad.abs().sum() > 0
    assert memory.grad[0].abs().sum() > 0 and torch.all(memory.grad[1, 2:] == 0)
