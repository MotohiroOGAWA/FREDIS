"""Source chemistry relations; computed once, before neural execution."""

from __future__ import annotations
from dataclasses import dataclass
from .ReactionInstance import ReactionInstance


def reaction_dominates(b: ReactionInstance, a: ReactionInstance) -> bool:
    """Exact redundancy rule, shared by domain normalization and ML tables."""
    if a == b or not b.retained_atom_maps <= a.retained_atom_maps:
        return False
    if any({u, v} <= b.retained_atom_maps for u, v, _ in a.bond_updates):
        return False
    surviving = frozenset(
        edge for edge in a.cut_bond_maps if set(edge) <= b.retained_atom_maps
    )
    if not surviving <= b.cut_bond_maps:
        return False
    if b.retained_atom_maps < a.retained_atom_maps:
        return True
    b_cuts = frozenset(
        edge for edge in b.cut_bond_maps if set(edge) <= b.retained_atom_maps
    )
    return surviving < b_cuts or (surviving == b_cuts and b.key < a.key)


@dataclass(frozen=True)
class ReactionRelations:
    conflict_mask: tuple[int, ...]
    invalidation_mask: tuple[int, ...]
    conflict_pairs: tuple[tuple[int, int], ...]
    invalidation_pairs: tuple[tuple[int, int], ...]
    dominance_pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_reactions(
        cls, reactions: tuple[ReactionInstance, ...]
    ) -> ReactionRelations:
        conflicts = [0] * len(reactions)
        invalidation = [0] * len(reactions)
        for i, a in enumerate(reactions):
            for j, b in enumerate(reactions):
                if i == j:
                    continue
                if a.changed_bond_maps & b.changed_bond_maps:
                    conflicts[i] |= 1 << j
                # Composite reactions are simultaneous. A collection is invalid
                # when any reaction removes atoms required by another reaction's
                # concrete Source match; there is no ordering that can rescue it.
                if not set(b.source_atom_maps) <= a.retained_atom_maps:
                    invalidation[i] |= 1 << j
        return cls(
            tuple(conflicts),
            tuple(invalidation),
            tuple(
                (i, j)
                for i, mask in enumerate(conflicts)
                for j in range(len(reactions))
                if mask & (1 << j)
            ),
            tuple(
                (i, j)
                for i, mask in enumerate(invalidation)
                for j in range(len(reactions))
                if mask & (1 << j)
            ),
            tuple(
                (j, i)
                for i, a in enumerate(reactions)
                for j, b in enumerate(reactions)
                if i != j and reaction_dominates(b, a)
            ),
        )

    def rejection(self, indices: tuple[int, ...]) -> str | None:
        selected = sum(1 << i for i in indices)
        if any(self.conflict_mask[i] & selected for i in indices):
            return "hard_conflict"
        if any(self.invalidation_mask[i] & selected for i in indices):
            return "invalidated_reaction"
        return None
