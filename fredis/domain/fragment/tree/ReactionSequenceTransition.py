"""In-memory exploration transitions expressed in Original Source atom maps."""

from __future__ import annotations
from dataclasses import dataclass
from ..reaction.ReactionInstance import ReactionInstance
from ..reaction.ReactionSequence import ReactionSequence


@dataclass(frozen=True)
class ReactionSequenceTransition:
    """Add a reaction to a parent history; the reaction runs on Original Source.

    Seed links attach an entire starting sequence to node 0. For those links
    added_reaction is None and is_seed is True. Local intermediate atom indices
    are deliberately absent.
    """

    added_reaction: ReactionInstance | None
    reaction_sequence: ReactionSequence
    parent_reaction_sequence: ReactionSequence | None = None
    is_seed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.reaction_sequence, ReactionSequence):
            raise TypeError("reaction_sequence must be a ReactionSequence")
        if self.parent_reaction_sequence is not None and not isinstance(
            self.parent_reaction_sequence, ReactionSequence
        ):
            raise TypeError(
                "parent_reaction_sequence must be a ReactionSequence or None"
            )
        if self.added_reaction is not None and not isinstance(
            self.added_reaction, ReactionInstance
        ):
            raise TypeError("added_reaction must be a ReactionInstance or None")
        if self.is_seed:
            if (
                self.added_reaction is not None
                or self.parent_reaction_sequence is not None
            ):
                raise ValueError(
                    "Seed transitions attach a complete sequence to Original Source"
                )
        elif self.added_reaction is None:
            raise ValueError("Exploration transitions require an added_reaction")
