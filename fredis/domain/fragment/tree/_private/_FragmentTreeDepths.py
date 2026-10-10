from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ._FragmentEdgeStore import _FragmentEdgeStore
from ._FragmentTreeAdjacency import _FragmentTreeAdjacency


@dataclass(frozen=True)
class _FragmentTreeDepths:
    """Node-depth values for a FragmentTree.

    Depth is computed from local node indices.
    """

    node_depths: np.ndarray

    def __post_init__(self) -> None:
        node_depths = np.asarray(self.node_depths, dtype=np.int64)

        if node_depths.ndim != 1:
            raise ValueError("node_depths must be a 1D array.")

        object.__setattr__(self, "node_depths", node_depths)

    @staticmethod
    def build(
        *,
        num_nodes: int,
        edge_store: _FragmentEdgeStore,
        adjacency: _FragmentTreeAdjacency,
    ) -> "_FragmentTreeDepths":
        """Compute minimal edge distance from root nodes to every node.

        Parameters
        ----------
        num_nodes:
            Number of local nodes in FragmentTree.

        edge_store:
            Edge store containing source_indices and target_indices.

        adjacency:
            CSR adjacency built from source_indices and target_indices.
        """
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")

        if not isinstance(edge_store, _FragmentEdgeStore):
            raise TypeError("edge_store must be a _FragmentEdgeStore.")

        if not isinstance(adjacency, _FragmentTreeAdjacency):
            raise TypeError("adjacency must be a _FragmentTreeAdjacency.")

        depth = np.full(int(num_nodes), -1, dtype=np.int64)

        if num_nodes == 0:
            return _FragmentTreeDepths(depth)

        _, in_indptr = adjacency.in_edge_csr

        roots = np.flatnonzero(
            (in_indptr[1:] - in_indptr[:-1]) == 0
        ).astype(np.int64, copy=False)

        if roots.size == 0:
            roots = np.asarray([0], dtype=np.int64)

        out_edge_indices, out_indptr = adjacency.out_edge_csr

        queue: deque[int] = deque()

        for root_index in roots:
            root_index = int(root_index)
            depth[root_index] = 0
            queue.append(root_index)

        while queue:
            source_index = queue.popleft()
            next_depth = int(depth[source_index]) + 1

            start = int(out_indptr[source_index])
            end = int(out_indptr[source_index + 1])

            for edge_index in out_edge_indices[start:end]:
                edge_index = int(edge_index)
                target_index = int(edge_store.target_indices[edge_index])

                if depth[target_index] == -1:
                    depth[target_index] = next_depth
                    queue.append(target_index)

        return _FragmentTreeDepths(depth)

    def get_nodes_by_depth(self) -> Dict[int, np.ndarray]:
        """Group node indices by minimal depth from root nodes."""
        out: Dict[int, np.ndarray] = {}

        for depth in sorted(set(int(d) for d in self.node_depths if d >= 0)):
            out[depth] = np.flatnonzero(
                self.node_depths == depth
            ).astype(np.int64, copy=False)

        return out

    def copy(self) -> "_FragmentTreeDepths":
        return _FragmentTreeDepths(
            node_depths=self.node_depths.copy(),
        )