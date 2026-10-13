from __future__ import annotations

import unittest

import torch

try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

if HAS_PYG:
    from fredis.ml.common.layers.graphormer import (
        PaddedGraphBatch,
        CentralityEncoding,
        SpatialEncoding,
        EdgeEncoding,
        GraphormerMHA,
        GraphormerBlock,
        GraphormerEncoder,
    )


@unittest.skipUnless(HAS_PYG, "torch_geometric is required")
class TestPaddedGraphBatch(unittest.TestCase):
    def test_from_graph_batch_pads_nodes_and_edges(self) -> None:
        torch.manual_seed(0)

        graph_1 = Data(
            x=torch.randn(3, 8),
            edge_index=torch.tensor(
                [[0, 1], [1, 2]],
                dtype=torch.long,
            ),
            edge_attr=torch.randn(2, 4),
        )

        graph_2 = Data(
            x=torch.randn(5, 8),
            edge_index=torch.tensor(
                [[0, 1, 2], [1, 2, 3]],
                dtype=torch.long,
            ),
            edge_attr=torch.randn(3, 4),
        )

        batch = Batch.from_data_list([graph_1, graph_2])
        padded = PaddedGraphBatch.from_graph_batch(batch, cap=5)

        self.assertEqual(padded.x.shape, (2, 5, 8))
        self.assertEqual(padded.node_mask.shape, (2, 5))
        self.assertEqual(padded.edge_index.shape[0], 2)
        self.assertEqual(padded.edge_index.shape[1], 2)
        self.assertEqual(padded.edge_attr.shape[0], 2)
        self.assertEqual(padded.edge_attr.shape[-1], 4)

        self.assertEqual(padded.num_nodes.tolist(), [3, 5])
        self.assertEqual(padded.num_edges.tolist(), [2, 3])

        self.assertTrue(padded.node_mask[0, :3].all())
        self.assertFalse(padded.node_mask[0, 3:].any())
        self.assertTrue(padded.node_mask[1, :5].all())

    def test_build_adj(self) -> None:
        torch.manual_seed(0)

        graph = Data(
            x=torch.randn(3, 8),
            edge_index=torch.tensor(
                [[0, 1], [1, 2]],
                dtype=torch.long,
            ),
            edge_attr=torch.randn(2, 4),
        )

        batch = Batch.from_data_list([graph])
        padded = PaddedGraphBatch.from_graph_batch(batch, cap=4)

        adj = padded.build_adj(undirected=True, add_self_loops=False)

        self.assertEqual(adj.shape, (1, 4, 4))
        self.assertTrue(adj[0, 0, 1])
        self.assertTrue(adj[0, 1, 0])
        self.assertTrue(adj[0, 1, 2])
        self.assertTrue(adj[0, 2, 1])

        # Padded node should not be connected.
        self.assertFalse(adj[0, 3].any())
        self.assertFalse(adj[0, :, 3].any())

@unittest.skipUnless(HAS_PYG, "torch_geometric is required")
class TestSpatialEncoding(unittest.TestCase):
    def test_compute_spatial_pos_for_path_graph(self) -> None:
        # 0 - 1 - 2
        adj = torch.tensor(
            [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ],
            dtype=torch.bool,
        )

        dist = SpatialEncoding.compute_spatial_pos(
            adj=adj,
            max_dist=4,
            add_self_loops=True,
        )

        expected = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 1],
                [2, 1, 0],
            ],
            dtype=torch.long,
        )

        self.assertTrue(torch.equal(dist, expected))

    def test_forward_returns_per_head_bias(self) -> None:
        torch.manual_seed(0)

        num_heads = 4
        max_dist = 5
        layer = SpatialEncoding(num_heads=num_heads, max_dist=max_dist)

        spatial_pos = torch.tensor(
            [
                [
                    [0, 1, 2],
                    [1, 0, 1],
                    [2, 1, 0],
                ]
            ],
            dtype=torch.long,
        )

        bias = layer(spatial_pos)

        self.assertEqual(bias.shape, (1, num_heads, 3, 3))
        self.assertTrue(torch.isfinite(bias).all())

@unittest.skipUnless(HAS_PYG, "torch_geometric is required")
class TestGraphormerAttentionBlocks(unittest.TestCase):
    def test_graphormer_mha_forward(self) -> None:
        torch.manual_seed(0)

        batch_size = 2
        num_nodes = 5
        dim = 16
        num_heads = 4

        layer = GraphormerMHA(
            dim=dim,
            num_heads=num_heads,
            dropout=0.0,
        )

        x = torch.randn(batch_size, num_nodes, dim)
        node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

        dist_bias = torch.randn(batch_size, num_heads, num_nodes, num_nodes) * 0.01

        y = layer(
            x,
            node_mask=node_mask,
            dist_bias=dist_bias,
            edge_bias=None,
        )

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_graphormer_block_forward_and_backward(self) -> None:
        torch.manual_seed(0)

        batch_size = 2
        num_nodes = 5
        dim = 16
        num_heads = 4

        block = GraphormerBlock(
            dim=dim,
            num_heads=num_heads,
            dropout=0.0,
        )

        x = torch.randn(batch_size, num_nodes, dim, requires_grad=True)
        node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

        dist_bias = torch.randn(batch_size, num_heads, num_nodes, num_nodes) * 0.01

        y = block(
            x,
            node_mask=node_mask,
            dist_bias=dist_bias,
            edge_bias=None,
        )

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

        loss = y.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

@unittest.skipUnless(HAS_PYG, "torch_geometric is required")
class TestGraphormerEncoder(unittest.TestCase):
    def test_encoder_forward_and_backward(self) -> None:
        torch.manual_seed(0)

        in_dim = 8
        dim = 16
        edge_dim = 4
        num_heads = 4

        graph_1 = Data(
            x=torch.randn(3, in_dim),
            edge_index=torch.tensor(
                [[0, 1], [1, 2]],
                dtype=torch.long,
            ),
            edge_attr=torch.randn(2, edge_dim),
        )

        graph_2 = Data(
            x=torch.randn(5, in_dim),
            edge_index=torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 4]],
                dtype=torch.long,
            ),
            edge_attr=torch.randn(4, edge_dim),
        )

        batch = Batch.from_data_list([graph_1, graph_2])

        encoder = GraphormerEncoder(
            node_dim=in_dim,
            hidden_dim=dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=2,
            max_degree=4,
            max_spatial_dist=5,
            max_edge_dist=5,
            dropout=0.0,
            graph_dim=dim,
            start_cap=4,
            cap_growth=2.0,
        )

        node_h, graph_h = encoder(batch)

        self.assertEqual(node_h.shape, (batch.num_nodes, dim))
        self.assertEqual(graph_h.shape, (batch.num_graphs, dim))
        self.assertTrue(torch.isfinite(node_h).all())
        self.assertTrue(torch.isfinite(graph_h).all())

        loss = node_h.mean() + graph_h.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in encoder.parameters()
            if p.requires_grad
        )

        self.assertTrue(has_grad)
