from __future__ import annotations

import unittest

import torch

from fredis.ml.common.layers.feed_forward import FeedForwardBlock


class TestFeedForwardBlock(unittest.TestCase):
    def test_forward_keeps_input_shape(self) -> None:
        torch.manual_seed(0)

        batch_size = 4
        seq_len = 7
        embed_dim = 16
        ff_dim = 64

        layer = FeedForwardBlock(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=0.0,
        )

        x = torch.randn(batch_size, seq_len, embed_dim)
        y = layer(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_backward_can_be_called(self) -> None:
        torch.manual_seed(0)

        layer = FeedForwardBlock(
            embed_dim=16,
            ff_dim=64,
            dropout=0.0,
        )

        x = torch.randn(4, 7, 16, requires_grad=True)
        y = layer(x)

        loss = y.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()