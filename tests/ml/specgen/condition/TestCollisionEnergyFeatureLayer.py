from __future__ import annotations

import unittest

import torch

from fredis.ml.specgen.components.condition.collision_energy_feature import (
    CollisionEnergyFeatureLayer,
)


class TestCollisionEnergyFeatureLayer(unittest.TestCase):
    def test_feature_dim_property(self) -> None:
        layer = CollisionEnergyFeatureLayer(
            feature_dim=16,
            fc_dims=(32,),
        )

        self.assertEqual(layer.feature_dim, 16)

    def test_ce_to_tensor_from_list(self) -> None:
        ce = CollisionEnergyFeatureLayer.ce_to_tensor([10.0, 20.0, 30.0])

        self.assertEqual(ce.shape, (3,))
        self.assertEqual(ce.dtype, torch.float32)
        self.assertEqual(ce.tolist(), [10.0, 20.0, 30.0])

    def test_ce_to_tensor_from_scalar_tensor(self) -> None:
        ce = CollisionEnergyFeatureLayer.ce_to_tensor(torch.tensor(25.0))

        self.assertEqual(ce.shape, (1,))
        self.assertEqual(ce.dtype, torch.float32)
        self.assertEqual(ce.item(), 25.0)

    def test_forward_accepts_1d_collision_energy(self) -> None:
        torch.manual_seed(0)

        layer = CollisionEnergyFeatureLayer(
            feature_dim=16,
            fc_dims=(32,),
        )

        ce = torch.tensor([10.0, 20.0, 30.0])
        out = layer(ce)

        self.assertEqual(out.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_accepts_2d_collision_energy(self) -> None:
        torch.manual_seed(0)

        layer = CollisionEnergyFeatureLayer(
            feature_dim=16,
            fc_dims=(32,),
        )

        ce = torch.tensor([[10.0], [20.0], [30.0]])
        out = layer(ce)

        self.assertEqual(out.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_rejects_invalid_shape(self) -> None:
        layer = CollisionEnergyFeatureLayer(
            feature_dim=16,
            fc_dims=(32,),
        )

        ce = torch.randn(2, 3)

        with self.assertRaises(ValueError):
            layer(ce)

    def test_backward_can_be_called(self) -> None:
        torch.manual_seed(0)

        layer = CollisionEnergyFeatureLayer(
            feature_dim=16,
            fc_dims=(32,),
        )

        ce = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        out = layer(ce)

        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(ce.grad)
        self.assertTrue(torch.isfinite(ce.grad).all())

        has_param_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in layer.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_param_grad)


if __name__ == "__main__":
    unittest.main()