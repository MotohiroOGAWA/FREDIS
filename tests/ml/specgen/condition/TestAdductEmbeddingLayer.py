from __future__ import annotations

import unittest

import torch

from fredis.libs.mmkit.mmkit import Adduct
from fredis.ml.specgen.components.condition.adduct_embedding import (
    AdductEmbeddingLayer,
)


class TestAdductEmbeddingLayer(unittest.TestCase):
    def test_properties_are_correct(self) -> None:
        layer = AdductEmbeddingLayer(
            adduct_type_strs=("[M+H]+", "[M+Na]+", "[M-H]-"),
            embedding_dim=8,
        )

        self.assertEqual(layer.feature_dim, 8)
        self.assertEqual(layer.num_adduct_types, 3)
        self.assertEqual(len(layer.adduct_type_strs), 3)
        self.assertEqual(len(layer.adduct_types), 3)
        self.assertEqual(len(layer.adduct2idx), 3)

    def test_adducts_to_idx_accepts_strings(self) -> None:
        layer = AdductEmbeddingLayer(
            adduct_type_strs=("[M+H]+", "[M+Na]+", "[M-H]-"),
            embedding_dim=8,
        )

        idx = layer.adducts_to_idx(["[M+H]+", "[M-H]-"])

        self.assertEqual(idx.dtype, torch.long)
        self.assertEqual(idx.shape, (2,))
        self.assertEqual(idx.tolist(), [
            layer.adduct2idx["[M+H]+"],
            layer.adduct2idx["[M-H]-"],
        ])

    def test_adducts_to_idx_accepts_adduct_objects(self) -> None:
        layer = AdductEmbeddingLayer(
            adduct_type_strs=("[M+H]+", "[M+Na]+", "[M-H]-"),
            embedding_dim=8,
        )

        adducts = [
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
        ]

        idx = layer.adducts_to_idx(adducts)

        self.assertEqual(idx.shape, (2,))
        self.assertEqual(idx.tolist(), [
            layer.adduct2idx["[M+H]+"],
            layer.adduct2idx["[M+Na]+"],
        ])

    def test_forward_returns_embedding(self) -> None:
        torch.manual_seed(0)

        layer = AdductEmbeddingLayer(
            adduct_type_strs=("[M+H]+", "[M+Na]+", "[M-H]-"),
            embedding_dim=8,
        )

        idx = layer.adducts_to_idx(["[M+H]+", "[M+Na]+"])
        out = layer(idx)

        self.assertEqual(out.shape, (2, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_backward_can_be_called(self) -> None:
        torch.manual_seed(0)

        layer = AdductEmbeddingLayer(
            adduct_type_strs=("[M+H]+", "[M+Na]+", "[M-H]-"),
            embedding_dim=8,
        )

        idx = layer.adducts_to_idx(["[M+H]+", "[M+Na]+"])
        out = layer(idx)

        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(layer.embedding.weight.grad)
        self.assertTrue(torch.isfinite(layer.embedding.weight.grad).all())

    def test_unknown_adduct_returns_minus_one_index(self) -> None:
        layer = AdductEmbeddingLayer(
            adduct_type_strs=("[M+H]+", "[M+Na]+"),
            embedding_dim=8,
        )

        idx = layer.adducts_to_idx(["[M+K]+"])

        self.assertEqual(idx.tolist(), [-1])

    def test_unknown_adduct_index_raises_in_forward(self) -> None:
        layer = AdductEmbeddingLayer(
            adduct_type_strs=("[M+H]+", "[M+Na]+"),
            embedding_dim=8,
        )

        idx = layer.adducts_to_idx(["[M+K]+"])

        with self.assertRaises(IndexError):
            layer(idx)


if __name__ == "__main__":
    unittest.main()