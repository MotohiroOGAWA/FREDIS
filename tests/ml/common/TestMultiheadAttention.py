from __future__ import annotations

import unittest

import torch

from fredis.ml.common.layers.multihead_attention import (
    MultiheadAttention,
    SelfAttention,
    CrossAttention,
)


class TestMultiheadAttention(unittest.TestCase):
    def test_forward_batch_first(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        seq_len = 5
        embed_dim = 16
        num_heads = 4

        layer = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        x = torch.randn(batch_size, seq_len, embed_dim)

        y, weights = layer(
            query=x,
            key=x,
            value=x,
            need_weights=True,
        )

        self.assertEqual(y.shape, (batch_size, seq_len, embed_dim))
        self.assertEqual(weights.shape, (batch_size, seq_len, seq_len))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(weights).all())

    def test_forward_with_attn_bias_4d(self) -> None:
        torch.manual_seed(0)

        batch_size = 2
        seq_len = 6
        embed_dim = 16
        num_heads = 4

        layer = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        x = torch.randn(batch_size, seq_len, embed_dim)

        attn_bias = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            seq_len,
        ) * 0.01

        y, _ = layer(
            query=x,
            key=x,
            value=x,
            attn_bias=attn_bias,
        )

        self.assertEqual(y.shape, (batch_size, seq_len, embed_dim))
        self.assertTrue(torch.isfinite(y).all())

    def test_key_padding_mask_masks_weights(self) -> None:
        torch.manual_seed(0)

        batch_size = 2
        seq_len = 5
        embed_dim = 16
        num_heads = 4

        layer = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        x = torch.randn(batch_size, seq_len, embed_dim)

        # True means padding.
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        key_padding_mask[:, -1] = True

        _, weights = layer(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        self.assertEqual(weights.shape, (batch_size, seq_len, seq_len))

        # The last key is padded, so attention weight to the last key should be zero.
        self.assertTrue(torch.allclose(weights[:, :, -1], torch.zeros_like(weights[:, :, -1])))

    def test_backward_can_be_called(self) -> None:
        torch.manual_seed(0)

        layer = MultiheadAttention(
            embed_dim=16,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )

        x = torch.randn(2, 5, 16, requires_grad=True)

        y, _ = layer(
            query=x,
            key=x,
            value=x,
        )

        loss = y.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_self_attention_forward(self) -> None:
        torch.manual_seed(0)

        layer = SelfAttention(
            embed_dim=16,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )

        x = torch.randn(2, 5, 16)
        y, _ = layer(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_cross_attention_forward(self) -> None:
        torch.manual_seed(0)

        layer = CrossAttention(
            embed_dim=16,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )

        q = torch.randn(2, 3, 16)
        kv = torch.randn(2, 5, 16)

        y, _ = layer(q=q, kv=kv)

        self.assertEqual(y.shape, q.shape)
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()