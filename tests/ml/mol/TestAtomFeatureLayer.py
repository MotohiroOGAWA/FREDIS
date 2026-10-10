from __future__ import annotations

import unittest

import torch
from rdkit import Chem

from fredis.ml.mol.atom_feature import AtomFeatureLayer


class TestAtomFeatureLayer(unittest.TestCase):
    def test_feature_dim_matches_feature_sets(self) -> None:
        layer = AtomFeatureLayer(symbols=("C", "H", "O", "N"))

        expected_dim = sum(len(v) for v in layer.feature_sets.values())

        self.assertEqual(layer.feature_dim, expected_dim)

    def test_encode_carbon_atom_returns_one_dimensional_tensor(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)

        atom = mol.GetAtomWithIdx(0)

        layer = AtomFeatureLayer(symbols=("C", "H", "O", "N"))
        feature = layer.encode(atom)

        self.assertEqual(feature.dim(), 1)
        self.assertEqual(feature.shape[0], layer.feature_dim)
        self.assertEqual(feature.dtype, torch.float32)
        self.assertTrue(torch.isfinite(feature).all())

    def test_encode_symbol_sets_exactly_one_symbol_bit(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)

        layer = AtomFeatureLayer(symbols=("C", "H", "O", "N"))
        symbol_feature = layer.encode_symbol(atom)

        self.assertEqual(symbol_feature.sum().item(), 1.0)

        carbon_index = layer.symbols.index("C")
        self.assertEqual(symbol_feature[carbon_index].item(), 1.0)

    def test_unsupported_symbol_raises_value_error(self) -> None:
        mol = Chem.MolFromSmiles("ClC")
        atom = mol.GetAtomWithIdx(0)

        layer = AtomFeatureLayer(symbols=("C", "H", "O", "N"))

        with self.assertRaises(ValueError):
            layer.encode_symbol(atom)

    def test_benzene_atom_is_six_cycle(self) -> None:
        mol = Chem.MolFromSmiles("c1ccccc1")
        atom = mol.GetAtomWithIdx(0)

        layer = AtomFeatureLayer(symbols=("C", "H", "O", "N"))
        ring_feature = layer.encode_ring_type(atom)

        ring_types = layer.feature_sets["ring_type"]
        six_cycle_index = ring_types.index("6-cycle")

        self.assertEqual(ring_feature[six_cycle_index].item(), 1.0)
        self.assertEqual(ring_feature.sum().item(), 1.0)

    def test_non_ring_atom_is_no_ring(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)

        layer = AtomFeatureLayer(symbols=("C", "H", "O", "N"))
        ring_feature = layer.encode_ring_type(atom)

        ring_types = layer.feature_sets["ring_type"]
        no_ring_index = ring_types.index("no-ring")

        self.assertEqual(ring_feature[no_ring_index].item(), 1.0)
        self.assertEqual(ring_feature.sum().item(), 1.0)


if __name__ == "__main__":
    unittest.main()