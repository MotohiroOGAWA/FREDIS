"""Peak annotations as unordered reaction combinations on Original Source."""

from __future__ import annotations

from dataclasses import dataclass

from ...libs.mmkit.mmkit import Adduct, Formula
from .reaction.ReactionSequence import ReactionSequence


def reaction_count(sequence: ReactionSequence | None) -> int:
    """Number of reactions in a combination; Source itself has none."""
    return 0 if sequence is None else len(sequence.reactions)


def contains(
    sequence: ReactionSequence | None, subset: ReactionSequence | None
) -> bool:
    """Whether every reaction of ``subset`` is also in ``sequence``."""
    if subset is None:
        return True
    if sequence is None:
        return False
    return set(subset.reactions) <= set(sequence.reactions)


@dataclass(frozen=True)
class PrecursorReaction:
    """One reaction combination that turns Original Source into the precursor.

    reaction_sequence is None exactly when Source itself already is the
    precursor (node_index == 0). Ambiguous chemistry (several combinations, or
    several candidate precursor nodes with the same mass) is recorded as
    several PrecursorReaction instances rather than resolved arbitrarily.
    """

    node_index: int
    reaction_sequence: ReactionSequence | None

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node_index must be a non-negative integer.")
        if self.reaction_sequence is not None and not isinstance(
            self.reaction_sequence, ReactionSequence
        ):
            raise TypeError("reaction_sequence must be a ReactionSequence or None.")
        if (self.node_index == 0) != (self.reaction_sequence is None):
            raise ValueError(
                "node_index 0 (Source) must pair with reaction_sequence=None, and vice versa."
            )


@dataclass(frozen=True)
class ReactionAnnotation:
    """One explanation of a peak by a reaction combination on Original Source.

    ``reaction_sequence`` is applied simultaneously to Source (None for Source
    itself) and contains ``precursor.reaction_sequence``. The ion is fragment
    ``node_index`` with ``fragment_ion_adduct`` and ``hydrogen_shift``;
    ``adduct`` is their combined adduct and ``formula`` the ion formula.
    """

    node_index: int
    smiles: str
    reaction_sequence: ReactionSequence | None
    precursor: PrecursorReaction
    adduct: Adduct
    fragment_ion_adduct: Adduct
    hydrogen_shift: int
    formula: Formula

    def __post_init__(self) -> None:
        if not contains(self.reaction_sequence, self.precursor.reaction_sequence):
            raise ValueError(
                "reaction_sequence must contain the precursor reaction combination."
            )

    @property
    def reaction_count(self) -> int:
        return reaction_count(self.reaction_sequence)
