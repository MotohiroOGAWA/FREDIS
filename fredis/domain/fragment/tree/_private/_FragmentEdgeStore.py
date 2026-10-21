from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..FragmentEdge import FragmentEdge
from ..ReactionSequenceTransition import ReactionSequenceTransition


@dataclass(frozen=True)
class _FragmentEdgeStore:
    """Array-backed storage for fragment edges.

    The local edge index is the array position.

    source_indices:
        Local source node indices in FragmentTree.

    target_indices:
        Local target node indices in FragmentTree.
    """

    source_indices: np.ndarray
    target_indices: np.ndarray
    transitions_by_edge: tuple[tuple[ReactionSequenceTransition, ...], ...] = ()

    def __post_init__(self) -> None:
        source_indices = np.asarray(self.source_indices, dtype=np.int64)
        target_indices = np.asarray(self.target_indices, dtype=np.int64)

        if source_indices.ndim != 1:
            raise ValueError("source_indices must be a 1D array.")

        if target_indices.ndim != 1:
            raise ValueError("target_indices must be a 1D array.")

        num_edges = len(source_indices)

        if len(target_indices) != num_edges:
            raise ValueError("target_indices must match source_indices.")

        transitions = self.transitions_by_edge or tuple(() for _ in range(num_edges))
        transitions = tuple(tuple(group) for group in transitions)
        if len(transitions) != num_edges:
            raise ValueError("transitions_by_edge must match source_indices")
        if any(
            not isinstance(t, ReactionSequenceTransition)
            for group in transitions
            for t in group
        ):
            raise TypeError("Invalid reaction transition")
        object.__setattr__(self, "transitions_by_edge", transitions)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "target_indices", target_indices)

    @property
    def num_edges(self) -> int:
        return len(self.source_indices)

    @property
    def edge_index(self) -> np.ndarray:
        """Return source/target node indices as shape [E, 2].

        This is computed from source_indices and target_indices.
        """
        return np.stack(
            [self.source_indices, self.target_indices],
            axis=1,
        ).astype(np.int64, copy=False)

    def get_edge(self, index: int) -> FragmentEdge:
        if not 0 <= index < self.num_edges:
            raise IndexError(f"Invalid edge index: {index}")

        return FragmentEdge(
            index=int(index),
            source_index=int(self.source_indices[index]),
            target_index=int(self.target_indices[index]),
            transitions=self.transitions_by_edge[index],
        )

    def copy(self) -> "_FragmentEdgeStore":
        return _FragmentEdgeStore(
            source_indices=self.source_indices.copy(),
            target_indices=self.target_indices.copy(),
            transitions_by_edge=self.transitions_by_edge,
        )
