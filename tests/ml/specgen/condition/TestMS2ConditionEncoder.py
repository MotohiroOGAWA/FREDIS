from __future__ import annotations

import unittest

import torch

from fredis.libs.mmkit.mmkit import Adduct
from fredis.ml.specgen.components.condition.condition_encoder import (
    MS2ConditionEncoder,
)


class TestMS2ConditionEncoder(unittest.TestCase):
    def _make_encoder(self) -> MS2ConditionEncoder:
        return MS2ConditionEncoder(
            adduct_type_strs=("[M+H]+", "[M+Na]+", "[M-H]-"),
            adduct_embedding_dim=8,
            ce_feature_dim=16,
            ce_fc_dims=(16,),
            feature_dim=20,
            fc_dims=(32,),
        )

    def test_properties_are_correct(self) -> None:
        encoder = self._make_encoder()

        self.assertEqual(encoder.in_feature_dim, 8 + 16)
        self.assertEqual(encoder.feature_dim, 20)
        self.assertEqual(
            encoder.adduct_type_strs,
            ("[M+H]+", "[M+Na]+", "[M-H]-"),
        )
        self.assertEqual(len(encoder.adduct_types), 3)

    def test_adducts_to_idx_accepts_strings(self) -> None:
        encoder = self._make_encoder()

        idx = encoder.adducts_to_idx(["[M+H]+", "[M-H]-"])

        self.assertEqual(idx.dtype, torch.long)
        self.assertEqual(idx.shape, (2,))
        self.assertEqual(
            idx.tolist(),
            [
                encoder.adduct_layer.adduct2idx["[M+H]+"],
                encoder.adduct_layer.adduct2idx["[M-H]-"],
            ],
        )

    def test_adducts_to_idx_accepts_adduct_objects(self) -> None:
        encoder = self._make_encoder()

        adducts = [
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
        ]

        idx = encoder.adducts_to_idx(adducts)

        self.assertEqual(idx.dtype, torch.long)
        self.assertEqual(idx.shape, (2,))
        self.assertEqual(
            idx.tolist(),
            [
                encoder.adduct_layer.adduct2idx["[M+H]+"],
                encoder.adduct_layer.adduct2idx["[M+Na]+"],
            ],
        )

    def test_ce_to_tensor_accepts_list(self) -> None:
        encoder = self._make_encoder()

        ce = encoder.ce_to_tensor([10.0, 20.0, 30.0])

        self.assertEqual(ce.shape, (3,))
        self.assertEqual(ce.dtype, torch.float32)
        self.assertEqual(ce.tolist(), [10.0, 20.0, 30.0])

    def test_ce_to_tensor_accepts_scalar_tensor(self) -> None:
        encoder = self._make_encoder()

        ce = encoder.ce_to_tensor(torch.tensor(25.0))

        self.assertEqual(ce.shape, (1,))
        self.assertEqual(ce.dtype, torch.float32)
        self.assertEqual(ce.item(), 25.0)

    def test_forward_accepts_1d_collision_energy(self) -> None:
        torch.manual_seed(0)

        encoder = self._make_encoder()

        adduct_idx = encoder.adducts_to_idx(
            ["[M+H]+", "[M+Na]+", "[M-H]-"]
        )
        collision_energy = torch.tensor([10.0, 20.0, 30.0])

        out = encoder(
            adduct_idx=adduct_idx,
            collision_energy=collision_energy,
        )

        self.assertEqual(out.shape, (3, 20))
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_accepts_2d_collision_energy(self) -> None:
        torch.manual_seed(0)

        encoder = self._make_encoder()

        adduct_idx = encoder.adducts_to_idx(["[M+H]+", "[M+Na]+"])
        collision_energy = torch.tensor([[10.0], [20.0]])

        out = encoder(
            adduct_idx=adduct_idx,
            collision_energy=collision_energy,
        )

        self.assertEqual(out.shape, (2, 20))
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_rejects_invalid_collision_energy_shape(self) -> None:
        encoder = self._make_encoder()

        adduct_idx = encoder.adducts_to_idx(["[M+H]+", "[M+Na]+"])
        collision_energy = torch.randn(2, 3)

        with self.assertRaises(ValueError):
            encoder(
                adduct_idx=adduct_idx,
                collision_energy=collision_energy,
            )

    def test_forward_rejects_unknown_adduct_index(self) -> None:
        encoder = self._make_encoder()

        adduct_idx = encoder.adducts_to_idx(["[M+K]+"])
        collision_energy = torch.tensor([20.0])

        with self.assertRaises(IndexError):
            encoder(
                adduct_idx=adduct_idx,
                collision_energy=collision_energy,
            )

    def test_backward_can_be_called(self) -> None:
        torch.manual_seed(0)

        encoder = self._make_encoder()

        adduct_idx = encoder.adducts_to_idx(
            ["[M+H]+", "[M+Na]+", "[M-H]-"]
        )
        collision_energy = torch.tensor(
            [10.0, 20.0, 30.0],
            requires_grad=True,
        )

        out = encoder(
            adduct_idx=adduct_idx,
            collision_energy=collision_energy,
        )

        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(collision_energy.grad)
        self.assertTrue(torch.isfinite(collision_energy.grad).all())

        has_parameter_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in encoder.parameters()
            if p.requires_grad
        )

        self.assertTrue(has_parameter_grad)

    def test_batch_size_mismatch_raises_runtime_error(self) -> None:
        encoder = self._make_encoder()

        adduct_idx = encoder.adducts_to_idx(["[M+H]+", "[M+Na]+"])
        collision_energy = torch.tensor([10.0, 20.0, 30.0])

        with self.assertRaises(RuntimeError):
            encoder(
                adduct_idx=adduct_idx,
                collision_energy=collision_energy,
            )


if __name__ == "__main__":
    unittest.main()