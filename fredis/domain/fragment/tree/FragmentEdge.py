from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from .ReactionSequenceTransition import ReactionSequenceTransition


@dataclass(frozen=True)
class FragmentEdge:
    """Edge in a FragmentTree.

    Transitions describe a parent reaction state plus one reaction (or a
    complete seed sequence), normalized into a child reaction state. They do
    not mean that a chemical reaction was applied to the parent fragment;
    every combined reaction is applied to Original Source.

    Parameters
    ----------
    index:
        Local edge index in the FragmentTree.

    source_index:
        Local index of the source FragmentNode in the FragmentTree.

    target_index:
        Local index of the target FragmentNode in the FragmentTree.

    transitions:
        Reaction sequence transitions associated with this edge.
    """

    index: int
    source_index: int
    target_index: int
    transitions: Tuple[ReactionSequenceTransition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if not all(isinstance(t, ReactionSequenceTransition) for t in self.transitions):
            raise TypeError("transitions must contain ReactionSequenceTransition instances")

    def with_transition(self, transition: ReactionSequenceTransition) -> FragmentEdge:
        if not isinstance(transition, ReactionSequenceTransition):
            raise TypeError("transition must be a ReactionSequenceTransition")
        if transition in self.transitions:
            return self.copy()
        return replace(self, transitions=self.transitions + (transition,))

    def __repr__(self) -> str:
        return "FragmentEdge" + self.__str__()

    def __str__(self) -> str:
        return (
            f"(index={self.index}; "
            f"source_index={self.source_index}; "
            f"target_index={self.target_index}; "
            f"transitions={len(self.transitions)})"
        )

    def copy(self) -> "FragmentEdge":
        """Return a copy of this fragment edge."""
        return FragmentEdge(
            index=self.index,
            source_index=self.source_index,
            target_index=self.target_index,
            transitions=self.transitions,
        )
