"""Order-independent, validated combinations of concrete reactions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from dataclasses import dataclass
from rdkit import Chem
from fredis.libs.mmkit.mmkit import Compound
from itertools import combinations
from .ReactionInstance import ReactionInstance
from .ReactionRelations import reaction_dominates

if TYPE_CHECKING:
    from .CompositeReaction import CompositeReaction


@dataclass(frozen=True, init=False)
class ReactionSequence:
    reactions: tuple[ReactionInstance, ...]

    def __init__(self, reactions: Iterable[ReactionInstance]) -> None:
        unique = tuple(sorted(set(reactions), key=lambda a: a.key))
        if not unique:
            raise ValueError("At least one ReactionInstance is required")
        universe = unique[0].retained_atom_maps | unique[0].discarded_atom_maps
        for a in unique:
            if a.retained_atom_maps | a.discarded_atom_maps != universe:
                raise ValueError(
                    "Reactions must describe the same Source atom universe"
                )
        for a, b in combinations(unique, 2):
            shared = a.changed_bond_maps & b.changed_bond_maps
            if shared:
                raise ValueError(
                    f"Reactions {a.key} and {b.key} share changed Source bonds: {sorted(shared)}"
                )
            if (
                not set(a.source_atom_maps) <= b.retained_atom_maps
                or not set(b.source_atom_maps) <= a.retained_atom_maps
            ):
                raise ValueError(
                    "Every reaction must retain every other reaction's Source match"
                )
        # Retained-set inclusion alone cannot remove additional cuts between surviving atoms.
        kept = [a for a in unique if not any(reaction_dominates(b, a) for b in unique)]
        object.__setattr__(self, "reactions", tuple(kept))

    @classmethod
    def from_reactions(cls, reactions: Iterable[ReactionInstance]) -> ReactionSequence:
        return cls(reactions)

    @property
    def retained_atom_maps(self) -> frozenset[int]:
        return frozenset.intersection(*(a.retained_atom_maps for a in self.reactions))

    @property
    def cut_bond_maps(self) -> frozenset[tuple[int, int]]:
        return frozenset().union(*(a.cut_bond_maps for a in self.reactions))

    @property
    def bond_updates(self) -> frozenset[tuple[int, int, Chem.BondType]]:
        return frozenset().union(*(a.bond_updates for a in self.reactions))

    @property
    def key(self) -> tuple[tuple[object, ...], ...]:
        """Canonical reaction history, distinct from the final effect."""
        return tuple(reaction.key for reaction in self.reactions)

    @property
    def effective_cut_bond_maps(self) -> frozenset[tuple[int, int]]:
        retained = self.retained_atom_maps
        return frozenset(edge for edge in self.cut_bond_maps if set(edge) <= retained)

    @property
    def effective_bond_updates(self) -> frozenset[tuple[int, int, Chem.BondType]]:
        retained = self.retained_atom_maps
        return frozenset(
            (u, v, order) for u, v, order in self.bond_updates if {u, v} <= retained
        )

    @property
    def effect_key(self) -> tuple[
        tuple[int, ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int, Chem.BondType], ...],
    ]:
        return (
            tuple(sorted(self.retained_atom_maps)),
            tuple(sorted(self.effective_cut_bond_maps)),
            tuple(sorted(self.effective_bond_updates)),
        )

    def compile(self, source: Compound | Chem.Mol) -> CompositeReaction:
        from .CompositeReaction import CompositeReaction

        return CompositeReaction.from_sequence(source=source, reaction_sequence=self)
