from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data

from fredis.libs.mmkit.mmkit import Compound
from fredis.ml.mol.graph_builder import MolGraphBuilder


class TestMolGraphBuilder(unittest.TestCase):
    def test_build_ethanol_graph(self) -> None:
        compound = Compound.from_smiles("CCO")

        builder = MolGraphBuilder(symbols=("C", "H", "O", "N"))
        data = builder.build(compound)

        self.assertIsInstance(data, Data)

        # Ethanol has 3 heavy atoms in RDKit default graph.
        self.assertEqual(data.x.shape, (3, builder.atom_dim))

        # Two bonds are represented bidirectionally.
        self.assertEqual(data.edge_index.shape, (2, 4))
        self.assertEqual(data.edge_attr.shape, (4, builder.bond_dim))

        self.assertEqual(data.x.dtype, torch.float32)
        self.assertEqual(data.edge_index.dtype, torch.long)
        self.assertEqual(data.edge_attr.dtype, torch.float32)

        self.assertTrue(torch.isfinite(data.x).all())
        self.assertTrue(torch.isfinite(data.edge_attr).all())

    def test_hydrogen_nodes_are_optional_and_attached_counts_are_preserved(self):
        builder=MolGraphBuilder(symbols=('C','O'))
        plain=builder.build(Compound.from_smiles('C'))
        explicit=builder.build(Compound.from_smiles('C[H]'))
        torch.testing.assert_close(plain.x,explicit.x)
        isotope=Compound.from_smiles('C([2H])O')
        data=builder.build(isotope)
        self.assertEqual(data.num_nodes,2)
        self.assertEqual(data.edge_index.tolist(),[[0,1],[1,0]])
        carbon=next(atom for atom in isotope.mol.GetAtoms() if atom.GetSymbol()=='C')
        counts=builder.atom_layer.encode_num_hydrogens(carbon)
        self.assertEqual(counts.argmax().item(),carbon.GetTotalNumHs(includeNeighbors=True))
        self.assertEqual(carbon.GetTotalNumHs(includeNeighbors=True),3)
        hydrogen=builder.build(Compound.from_smiles('[H]'))
        self.assertEqual(hydrogen.x.shape,(0,builder.atom_dim))
        registered=MolGraphBuilder(symbols=('C','H','O')).build(isotope)
        self.assertEqual(registered.num_nodes,3)

    def test_build_single_atom_graph(self) -> None:
        compound = Compound.from_smiles("C")

        builder = MolGraphBuilder(symbols=("C", "H", "O", "N"))
        data = builder.build(compound)

        self.assertEqual(data.x.shape, (1, builder.atom_dim))
        self.assertEqual(data.edge_index.shape, (2, 0))
        self.assertEqual(data.edge_attr.shape, (0, builder.bond_dim))

    def test_build_graph_on_cpu_by_default(self) -> None:
        compound = Compound.from_smiles("CCO")

        builder = MolGraphBuilder(symbols=("C", "H", "O", "N"))
        data = builder.build(compound)

        self.assertEqual(data.x.device.type, "cpu")
        self.assertEqual(data.edge_index.device.type, "cpu")
        self.assertEqual(data.edge_attr.device.type, "cpu")

    def test_build_unsupported_symbol_raises_value_error(self) -> None:
        compound = Compound.from_smiles("ClC")

        builder = MolGraphBuilder(symbols=("C", "H", "O", "N"))

        with self.assertRaises(ValueError):
            builder.build(compound)

    def test_symbols_are_sorted_unique(self) -> None:
        builder = MolGraphBuilder(symbols=("O", "C", "C", "N"))

        self.assertEqual(builder.symbols, ("C", "N", "O"))


if __name__ == "__main__":
    unittest.main()