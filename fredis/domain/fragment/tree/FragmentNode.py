from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FragmentNode:
    index: int
    smiles: str

    def __post_init__(self) -> None:

        if not self.smiles:
            raise ValueError("smiles must not be empty.")

    def __repr__(self) -> str:
        return "FragmentNode" + self.__str__()

    def __str__(self) -> str:
        return f"(index={self.index}; {self.smiles})"

    def copy(self) -> "FragmentNode":
        return FragmentNode(
            index=self.index,
            smiles=self.smiles,
        )
