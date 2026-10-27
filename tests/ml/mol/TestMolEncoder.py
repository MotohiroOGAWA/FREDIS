from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Batch

from fredis.libs.mmkit.mmkit import Compound
from fredis.ml.mol.mol_encoder import MolEncoder


class TestMolEncoder(unittest.TestCase):
    def test_encode_components_returns_data(self) -> None:
        torch.manual_seed(0)

        compound = Compound.from_smiles("CCO")

        encoder = MolEncoder(
            symbols=("C", "H", "O", "N"),
            node_dim=16,
            graph_dim=32,
            num_layers=1,
            num_heads=4,
            max_degree=4,
            max_spatial_dist=3,
            max_edge_dist=3,
            dropout=0.0,
        )

        data = encoder.encode_components(compound)

        self.assertEqual(data.x.shape[0], 3)
        self.assertEqual(data.x.shape[1], encoder.atom_dim)
        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_attr.shape[1], encoder.bond_dim)

        self.assertEqual(encoder.node_dim, 16)
        self.assertEqual(encoder.graph_dim, 32)

    def test_forward_batch_returns_node_and_graph_embeddings(self) -> None:
        torch.manual_seed(0)

        compounds = [
            Compound.from_smiles("CCO"),
            Compound.from_smiles("c1ccccc1"),
        ]

        encoder = MolEncoder(
            symbols=("C", "H", "O", "N"),
            node_dim=16,
            graph_dim=32,
            num_layers=1,
            num_heads=4,
            max_degree=6,
            max_spatial_dist=6,
            max_edge_dist=3,
            dropout=0.0,
        )

        data_list = [
            encoder.encode_components(compound)
            for compound in compounds
        ]

        batch = Batch.from_data_list(data_list)
        encoded_batch = encoder.encode_with_nodes(batch)

        total_atoms = sum(data.x.size(0) for data in data_list)

        self.assertEqual(encoded_batch.x.shape, (total_atoms, 16))
        self.assertTrue(hasattr(encoded_batch, "embeddings"))
        self.assertEqual(encoded_batch.embeddings.shape, (2, 32))

        self.assertTrue(torch.isfinite(encoded_batch.x).all())
        self.assertTrue(torch.isfinite(encoded_batch.embeddings).all())

    def test_encode_single_compound_returns_embedding_and_atom_states(self) -> None:
        torch.manual_seed(0)

        compound = Compound.from_smiles("CCO")

        encoder = MolEncoder(
            symbols=("C", "H", "O", "N"),
            node_dim=16,
            graph_dim=32,
            num_layers=1,
            num_heads=4,
            max_degree=4,
            max_spatial_dist=3,
            max_edge_dist=3,
            dropout=0.0,
        )

        embedding = encoder.encode(compound)

        self.assertIsNotNone(embedding)
        assert embedding is not None

        self.assertEqual(embedding.shape, (32,))
        self.assertTrue(torch.isfinite(embedding).all())

        encoded = encoder.encode_with_nodes(
            Batch.from_data_list([encoder.encode_components(compound)])
        )
        self.assertEqual(encoded.x.shape, (3, 16))
        self.assertTrue(torch.isfinite(encoded.x).all())

    def test_encode_batch_returns_valid_indices(self) -> None:
        torch.manual_seed(0)

        compounds = [
            Compound.from_smiles("CCO"),
            Compound.from_smiles("c1ccccc1"),
        ]

        encoder = MolEncoder(
            symbols=("C", "H", "O", "N"),
            node_dim=16,
            graph_dim=32,
            num_layers=1,
            num_heads=4,
            max_degree=6,
            max_spatial_dist=6,
            max_edge_dist=3,
            dropout=0.0,
        )

        batch, valid_indices = encoder.encode_batch(compounds)

        self.assertIsNotNone(batch)
        self.assertIsNotNone(valid_indices)
        assert batch is not None
        assert valid_indices is not None

        self.assertEqual(valid_indices.tolist(), [0, 1])
        self.assertEqual(batch.shape, (2, 32))

    def test_encode_batch_skips_invalid_symbol(self) -> None:
        torch.manual_seed(0)

        compounds = [
            Compound.from_smiles("CCO"),
            Compound.from_smiles("ClC"),
        ]

        encoder = MolEncoder(
            symbols=("C", "H", "O", "N"),
            node_dim=16,
            graph_dim=32,
            num_layers=1,
            num_heads=4,
            max_degree=4,
            max_spatial_dist=3,
            max_edge_dist=3,
            dropout=0.0,
        )

        batch, valid_indices = encoder.encode_batch(compounds)

        self.assertIsNotNone(batch)
        self.assertIsNotNone(valid_indices)
        assert batch is not None
        assert valid_indices is not None

        # The second compound contains Cl, which is unsupported by symbols.
        self.assertEqual(valid_indices.tolist(), [0])
        self.assertEqual(batch.shape, (1, 32))

    def test_backward_can_be_called(self) -> None:
        torch.manual_seed(0)

        compounds = [
            Compound.from_smiles("CCO"),
            Compound.from_smiles("c1ccccc1"),
        ]

        encoder = MolEncoder(
            symbols=("C", "H", "O", "N"),
            node_dim=16,
            graph_dim=32,
            num_layers=1,
            num_heads=4,
            max_degree=6,
            max_spatial_dist=6,
            max_edge_dist=3,
            dropout=0.0,
        )

        data_list = [
            encoder.encode_components(compound)
            for compound in compounds
        ]

        batch = Batch.from_data_list(data_list)
        encoded_batch = encoder.encode_with_nodes(batch)

        loss = encoded_batch.x.mean() + encoded_batch.embeddings.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in encoder.parameters()
            if p.requires_grad
        )

        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
