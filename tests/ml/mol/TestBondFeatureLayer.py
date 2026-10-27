from __future__ import annotations

import unittest

import torch
from rdkit import Chem

from fredis.ml.mol.bond_feature import BondFeatureLayer


class TestBondFeatureLayer(unittest.TestCase):
    def test_feature_dim_matches_feature_sets(self) -> None:
        layer = BondFeatureLayer()

        expected_dim = sum(len(v) for v in layer.feature_sets.values())

        self.assertEqual(layer.feature_dim, expected_dim)

    def test_encode_single_bond(self) -> None:
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)

        layer = BondFeatureLayer()
        feature = layer.encode(bond)

        self.assertEqual(feature.dim(), 1)
        self.assertEqual(feature.shape[0], layer.feature_dim)
        self.assertEqual(feature.dtype, torch.float32)
        self.assertTrue(torch.isfinite(feature).all())

        bond_types = layer.feature_sets["bond_type"]
        single_index = bond_types.index("SINGLE")

        bond_type_feature = layer.encode_bond_type(bond)
        self.assertEqual(bond_type_feature[single_index].item(), 1.0)
        self.assertEqual(bond_type_feature.sum().item(), 1.0)

    def test_encode_double_bond(self) -> None:
        mol = Chem.MolFromSmiles("C=O")
        bond = mol.GetBondWithIdx(0)

        layer = BondFeatureLayer()

        bond_types = layer.feature_sets["bond_type"]
        double_index = bond_types.index("DOUBLE")

        bond_type_feature = layer.encode_bond_type(bond)

        self.assertEqual(bond_type_feature[double_index].item(), 1.0)
        self.assertEqual(bond_type_feature.sum().item(), 1.0)

    def test_encode_aromatic_bond(self) -> None:
        mol = Chem.MolFromSmiles("c1ccccc1")
        bond = mol.GetBondWithIdx(0)

        layer = BondFeatureLayer()

        bond_types = layer.feature_sets["bond_type"]
        aromatic_index = bond_types.index("AROMATIC")

        bond_type_feature = layer.encode_bond_type(bond)

        self.assertEqual(bond_type_feature[aromatic_index].item(), 1.0)
        self.assertEqual(bond_type_feature.sum().item(), 1.0)

    def test_benzene_bond_is_six_cycle(self) -> None:
        mol = Chem.MolFromSmiles("c1ccccc1")
        bond = mol.GetBondWithIdx(0)

        layer = BondFeatureLayer()

        ring_types = layer.feature_sets["ring_type"]
        six_cycle_index = ring_types.index("6-cycle")

        ring_feature = layer.encode_ring_type(bond)

        self.assertEqual(ring_feature[six_cycle_index].item(), 1.0)
        self.assertEqual(ring_feature.sum().item(), 1.0)

    def test_none_bond_returns_zero_bond_type_and_no_ring(self) -> None:
        layer = BondFeatureLayer()

        bond_type_feature = layer.encode_bond_type(None)
        ring_feature = layer.encode_ring_type(None)

        self.assertEqual(bond_type_feature.sum().item(), 0.0)

        ring_types = layer.feature_sets["ring_type"]
        no_ring_index = ring_types.index("no-ring")

        self.assertEqual(ring_feature[no_ring_index].item(), 1.0)
        self.assertEqual(ring_feature.sum().item(), 1.0)


if __name__ == "__main__":
    unittest.main()