from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class _FragmentTreeAdjacency:
    """Incoming and outgoing edge adjacency in CSR form.

    edge index is the array position in FragmentTree / _FragmentEdgeStore.
    """

    in_edge_indices: np.ndarray
    in_edge_indptr: np.ndarray
    out_edge_indices: np.ndarray
    out_edge_indptr: np.ndarray

    def __post_init__(self) -> None:
        in_edge_indices = np.asarray(self.in_edge_indices, dtype=np.int64)
        in_edge_indptr = np.asarray(self.in_edge_indptr, dtype=np.int64)
        out_edge_indices = np.asarray(self.out_edge_indices, dtype=np.int64)
        out_edge_indptr = np.asarray(self.out_edge_indptr, dtype=np.int64)

        if not self._valid_csr_arrays(in_edge_indices, in_edge_indptr):
            raise ValueError("Invalid incoming-edge CSR arrays.")

        if not self._valid_csr_arrays(out_edge_indices, out_edge_indptr):
            raise ValueError("Invalid outgoing-edge CSR arrays.")

        if in_edge_indptr.shape != out_edge_indptr.shape:
            raise ValueError(
                "Incoming and outgoing indptr arrays must have the same length."
            )

        object.__setattr__(self, "in_edge_indices", in_edge_indices)
        object.__setattr__(self, "in_edge_indptr", in_edge_indptr)
        object.__setattr__(self, "out_edge_indices", out_edge_indices)
        object.__setattr__(self, "out_edge_indptr", out_edge_indptr)

    @staticmethod
    def _valid_csr_arrays(
        edge_indices: np.ndarray,
        indptr: np.ndarray,
    ) -> bool:
        return (
            edge_indices.ndim == 1
            and indptr.ndim == 1
            and len(indptr) >= 1
            and indptr[0] == 0
            and indptr[-1] == len(edge_indices)
            and np.all(indptr[1:] >= indptr[:-1])
        )

    @staticmethod
    def from_source_target_indices(
        *,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        num_nodes: int,
    ) -> "_FragmentTreeAdjacency":
        """Build incoming/outgoing adjacency from source and target node indices.

        Parameters
        ----------
        source_indices:
            Source node indices for each edge.

        target_indices:
            Target node indices for each edge.

        num_nodes:
            Number of nodes in the FragmentTree.
        """
        source_indices = np.asarray(source_indices, dtype=np.int64)
        target_indices = np.asarray(target_indices, dtype=np.int64)
        num_nodes = int(num_nodes)

        if source_indices.ndim != 1:
            raise ValueError("source_indices must be a 1D array.")

        if target_indices.ndim != 1:
            raise ValueError("target_indices must be a 1D array.")

        if len(source_indices) != len(target_indices):
            raise ValueError(
                "source_indices and target_indices must have the same length."
            )

        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")

        num_edges = len(source_indices)

        if num_edges == 0:
            empty_indices = np.asarray([], dtype=np.int64)
            empty_indptr = np.zeros(num_nodes + 1, dtype=np.int64)

            return _FragmentTreeAdjacency(
                in_edge_indices=empty_indices,
                in_edge_indptr=empty_indptr,
                out_edge_indices=empty_indices,
                out_edge_indptr=empty_indptr,
            )

        if np.any(source_indices < 0) or np.any(source_indices >= num_nodes):
            raise ValueError("source_indices contain invalid node indices.")

        if np.any(target_indices < 0) or np.any(target_indices >= num_nodes):
            raise ValueError("target_indices contain invalid node indices.")

        edge_indices = np.arange(num_edges, dtype=np.int64)

        order_in = np.argsort(target_indices, kind="mergesort")
        in_edge_indices = edge_indices[order_in]
        in_counts = np.bincount(
            target_indices[order_in],
            minlength=num_nodes,
        ).astype(np.int64)

        in_edge_indptr = np.empty(num_nodes + 1, dtype=np.int64)
        in_edge_indptr[0] = 0
        np.cumsum(in_counts, out=in_edge_indptr[1:])

        order_out = np.argsort(source_indices, kind="mergesort")
        out_edge_indices = edge_indices[order_out]
        out_counts = np.bincount(
            source_indices[order_out],
            minlength=num_nodes,
        ).astype(np.int64)

        out_edge_indptr = np.empty(num_nodes + 1, dtype=np.int64)
        out_edge_indptr[0] = 0
        np.cumsum(out_counts, out=out_edge_indptr[1:])

        return _FragmentTreeAdjacency(
            in_edge_indices=in_edge_indices,
            in_edge_indptr=in_edge_indptr,
            out_edge_indices=out_edge_indices,
            out_edge_indptr=out_edge_indptr,
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes inferred from CSR pointer arrays."""
        return len(self.in_edge_indptr) - 1

    @property
    def in_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Incoming edge adjacency as (edge_indices, indptr)."""
        return self.in_edge_indices, self.in_edge_indptr

    @property
    def out_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Outgoing edge adjacency as (edge_indices, indptr)."""
        return self.out_edge_indices, self.out_edge_indptr

    def get_in_edge_indices(self, node_index: int) -> np.ndarray:
        """Return incoming local edge indices for a node."""
        if not 0 <= node_index < self.num_nodes:
            raise IndexError(f"Invalid node index: {node_index}")

        start = int(self.in_edge_indptr[node_index])
        end = int(self.in_edge_indptr[node_index + 1])

        return self.in_edge_indices[start:end]

    def get_out_edge_indices(self, node_index: int) -> np.ndarray:
        """Return outgoing local edge indices for a node."""
        if not 0 <= node_index < self.num_nodes:
            raise IndexError(f"Invalid node index: {node_index}")

        start = int(self.out_edge_indptr[node_index])
        end = int(self.out_edge_indptr[node_index + 1])

        return self.out_edge_indices[start:end]

    def copy(self) -> "_FragmentTreeAdjacency":
        return _FragmentTreeAdjacency(
            in_edge_indices=self.in_edge_indices.copy(),
            in_edge_indptr=self.in_edge_indptr.copy(),
            out_edge_indices=self.out_edge_indices.copy(),
            out_edge_indptr=self.out_edge_indptr.copy(),
        )