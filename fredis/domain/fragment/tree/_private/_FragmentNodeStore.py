from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..FragmentNode import FragmentNode


@dataclass(frozen=True)
class _FragmentNodeStore:
    """Array-backed storage for fragment nodes.

    node_smiles:
        Canonical SMILES strings.

    The array position is the local node index in FragmentTree.
    """

    node_smiles: np.ndarray

    def __post_init__(self) -> None:
        node_smiles = np.asarray(self.node_smiles, dtype=object)

        if node_smiles.ndim != 1:
            raise ValueError("node_smiles must be a 1D array.")

        object.__setattr__(self, "node_smiles", node_smiles)

    @property
    def num_nodes(self) -> int:
        return len(self.node_smiles)

    def get_node(self, index: int) -> FragmentNode:
        if not 0 <= index < self.num_nodes:
            raise IndexError(f"Invalid node index: {index}")

        return FragmentNode(
            index=int(index),
            smiles=str(self.node_smiles[index]),
        )

    def get_node_by_smiles(self, smiles: str) -> FragmentNode | None:
        try:
            index = np.where(self.node_smiles == smiles)[0][0]
            return self.get_node(index)
        except IndexError:
            return None

    def copy(self) -> "_FragmentNodeStore":
        return _FragmentNodeStore(
            node_smiles=self.node_smiles.copy(),
        )
