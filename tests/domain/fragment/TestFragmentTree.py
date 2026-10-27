from __future__ import annotations

import unittest

import numpy as np

from fredis.domain.fragment.tree.FragmentEdge import FragmentEdge
from fredis.domain.fragment.tree.FragmentNode import FragmentNode
from fredis.domain.fragment.tree.FragmentTree import FragmentTree
from fredis.domain.fragment.tree._private._FragmentEdgeStore import (
    _FragmentEdgeStore,
)
from fredis.domain.fragment.tree._private._FragmentNodeStore import (
    _FragmentNodeStore,
)
from fredis.domain.fragment.tree._private._FragmentTreeAdjacency import (
    _FragmentTreeAdjacency,
)
from fredis.domain.fragment.tree._private._FragmentTreeDepths import (
    _FragmentTreeDepths,
)


def make_nodes() -> tuple[FragmentNode, ...]:
    return (
        FragmentNode(index=0, smiles="CCO"),
        FragmentNode(index=1, smiles="CC"),
        FragmentNode(index=2, smiles="O"),
        FragmentNode(index=3, smiles="C"),
    )


def make_edges() -> tuple[FragmentEdge, ...]:
    return (
        FragmentEdge(index=0, source_index=0, target_index=1),
        FragmentEdge(index=1, source_index=0, target_index=2),
        FragmentEdge(index=2, source_index=1, target_index=3),
    )


class TestFragmentNode(unittest.TestCase):

    def test_fragment_node_fields(self) -> None:
        node = FragmentNode(index=0, smiles="CCO")

        self.assertEqual(node.index, 0)
        self.assertEqual(node.smiles, "CCO")

    def test_fragment_node_copy(self) -> None:
        node = FragmentNode(index=0, smiles="CCO")
        copied = node.copy()

        self.assertEqual(copied, node)
        self.assertIsNot(copied, node)


class TestFragmentEdge(unittest.TestCase):

    def test_fragment_edge_fields(self) -> None:
        edge = FragmentEdge(index=0, source_index=0, target_index=1)

        self.assertEqual(edge.index, 0)
        self.assertEqual(edge.source_index, 0)
        self.assertEqual(edge.target_index, 1)
        self.assertEqual(edge.transitions, ())

    def test_fragment_edge_copy(self) -> None:
        edge = make_edges()[0]
        copied = edge.copy()

        self.assertEqual(copied, edge)
        self.assertIsNot(copied, edge)


class TestFragmentNodeStore(unittest.TestCase):

    def test_get_node(self) -> None:
        store = _FragmentNodeStore(
            node_smiles=np.asarray(["CCO", "CC"], dtype=object),
        )

        node = store.get_node(1)

        self.assertEqual(node.index, 1)
        self.assertEqual(node.smiles, "CC")

    def test_num_nodes(self) -> None:
        store = _FragmentNodeStore(
            node_smiles=np.asarray(["CCO", "CC"], dtype=object),
        )

        self.assertEqual(store.num_nodes, 2)

    def test_invalid_node_store_shape_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            _FragmentNodeStore(
                node_smiles=np.asarray([["CCO", "CC"]], dtype=object),
            )


class TestFragmentEdgeStore(unittest.TestCase):

    def test_get_edge(self) -> None:
        store = _FragmentEdgeStore(
            source_indices=np.asarray([0, 0, 1], dtype=np.int64),
            target_indices=np.asarray([1, 2, 3], dtype=np.int64),
        )

        edge = store.get_edge(2)

        self.assertEqual(edge.index, 2)
        self.assertEqual(edge.source_index, 1)
        self.assertEqual(edge.target_index, 3)
        self.assertEqual(edge.transitions, ())

    def test_num_edges(self) -> None:
        store = _FragmentEdgeStore(
            source_indices=np.asarray([], dtype=np.int64),
            target_indices=np.asarray([], dtype=np.int64),
        )

        self.assertEqual(store.num_edges, 0)


class TestFragmentTreeAdjacency(unittest.TestCase):

    def test_from_source_target_indices(self) -> None:
        adjacency = _FragmentTreeAdjacency.from_source_target_indices(
            source_indices=np.asarray([0, 0, 1], dtype=np.int64),
            target_indices=np.asarray([1, 2, 3], dtype=np.int64),
            num_nodes=4,
        )

        np.testing.assert_array_equal(
            adjacency.get_out_edge_indices(0),
            np.asarray([0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            adjacency.get_out_edge_indices(1),
            np.asarray([2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            adjacency.get_in_edge_indices(3),
            np.asarray([2], dtype=np.int64),
        )

    def test_empty_adjacency(self) -> None:
        adjacency = _FragmentTreeAdjacency.from_source_target_indices(
            source_indices=np.asarray([], dtype=np.int64),
            target_indices=np.asarray([], dtype=np.int64),
            num_nodes=2,
        )

        np.testing.assert_array_equal(
            adjacency.in_edge_indptr,
            np.asarray([0, 0, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            adjacency.out_edge_indptr,
            np.asarray([0, 0, 0], dtype=np.int64),
        )


class TestFragmentTreeDepths(unittest.TestCase):

    def test_build_depths(self) -> None:
        edge_store = _FragmentEdgeStore(
            source_indices=np.asarray([0, 0, 1], dtype=np.int64),
            target_indices=np.asarray([1, 2, 3], dtype=np.int64),
        )

        adjacency = _FragmentTreeAdjacency.from_source_target_indices(
            source_indices=edge_store.source_indices,
            target_indices=edge_store.target_indices,
            num_nodes=4,
        )

        depths = _FragmentTreeDepths.build(
            num_nodes=4,
            edge_store=edge_store,
            adjacency=adjacency,
        )

        np.testing.assert_array_equal(
            depths.node_depths,
            np.asarray([0, 1, 1, 2], dtype=np.int64),
        )

    def test_get_nodes_by_depth(self) -> None:
        depths = _FragmentTreeDepths(
            node_depths=np.asarray([0, 1, 1, 2], dtype=np.int64),
        )

        nodes_by_depth = depths.get_nodes_by_depth()

        np.testing.assert_array_equal(
            nodes_by_depth[0],
            np.asarray([0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[1],
            np.asarray([1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[2],
            np.asarray([3], dtype=np.int64),
        )


class TestFragmentTree(unittest.TestCase):

    def test_from_nodes_and_edges(self) -> None:
        nodes = make_nodes()
        edges = make_edges()

        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=nodes,
            edges=edges,
        )

        self.assertEqual(tree.smiles, "CCO")
        self.assertEqual(tree.num_nodes, 4)
        self.assertEqual(tree.num_edges, 3)
        self.assertEqual(tree.num_transitions, 0)

        np.testing.assert_array_equal(
            tree.source_indices,
            np.asarray([0, 0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            tree.target_indices,
            np.asarray([1, 2, 3], dtype=np.int64),
        )

    def test_get_node(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        node = tree.get_node(2)

        self.assertEqual(node.index, 2)
        self.assertEqual(node.smiles, "O")

    def test_get_edge(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        edge = tree.get_edge(0)

        self.assertEqual(edge.index, 0)
        self.assertEqual(edge.source_index, 0)
        self.assertEqual(edge.target_index, 1)

    def test_get_in_edges(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        in_edges = tree.get_in_edges(3)

        self.assertEqual(len(in_edges), 1)
        self.assertEqual(in_edges[0].index, 2)
        self.assertEqual(in_edges[0].source_index, 1)
        self.assertEqual(in_edges[0].target_index, 3)

    def test_get_out_edges(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        out_edges = tree.get_out_edges(0)

        self.assertEqual(len(out_edges), 2)
        self.assertEqual(out_edges[0].index, 0)
        self.assertEqual(out_edges[1].index, 1)

    def test_get_parent_nodes(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        parent_nodes = tree.get_parent_nodes(3)

        self.assertEqual(len(parent_nodes), 1)
        self.assertEqual(parent_nodes[0].index, 1)
        self.assertEqual(parent_nodes[0].smiles, "CC")

    def test_get_child_nodes(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        child_nodes = tree.get_child_nodes(0)

        self.assertEqual(len(child_nodes), 2)
        self.assertEqual(child_nodes[0].index, 1)
        self.assertEqual(child_nodes[1].index, 2)

    def test_get_root_node_indices(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        np.testing.assert_array_equal(
            tree.get_root_node_indices(),
            np.asarray([0], dtype=np.int64),
        )

    def test_node_depths(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        np.testing.assert_array_equal(
            tree.node_depths,
            np.asarray([0, 1, 1, 2], dtype=np.int64),
        )

    def test_get_nodes_by_depth(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        nodes_by_depth = tree.get_nodes_by_depth()

        np.testing.assert_array_equal(
            nodes_by_depth[0],
            np.asarray([0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[1],
            np.asarray([1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            nodes_by_depth[2],
            np.asarray([3], dtype=np.int64),
        )

    def test_from_nodes_and_edges_rejects_non_continuous_node_indices(self) -> None:
        nodes = (
            FragmentNode(index=0, smiles="CCO"),
            FragmentNode(index=2, smiles="CC"),
        )

        with self.assertRaises(ValueError):
            FragmentTree.from_nodes_and_edges(
                smiles="CCO",
                nodes=nodes,
                edges=(),
            )

    def test_from_nodes_and_edges_rejects_invalid_edge_node_index(self) -> None:
        nodes = make_nodes()

        edge = FragmentEdge(index=0, source_index=0, target_index=99)

        with self.assertRaises(ValueError):
            FragmentTree.from_nodes_and_edges(
                smiles="CCO",
                nodes=nodes,
                edges=(edge,),
            )

    def test_copy(self) -> None:
        tree = FragmentTree.from_nodes_and_edges(
            smiles="CCO",
            nodes=make_nodes(),
            edges=make_edges(),
        )

        copied = tree.copy()

        self.assertEqual(copied.smiles, tree.smiles)
        self.assertEqual(copied.num_nodes, tree.num_nodes)
        self.assertEqual(copied.num_edges, tree.num_edges)
        self.assertIsNot(copied, tree)

        np.testing.assert_array_equal(copied.node_smiles, tree.node_smiles)
        np.testing.assert_array_equal(copied.source_indices, tree.source_indices)


if __name__ == "__main__":
    unittest.main()