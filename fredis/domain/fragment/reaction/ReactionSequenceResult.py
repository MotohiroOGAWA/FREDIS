"""A Source-anchored reaction sequence and its generated target."""
from __future__ import annotations
from dataclasses import dataclass
from fredis.libs.mmkit.mmkit import Compound
from .ReactionSequence import ReactionSequence


@dataclass(frozen=True)
class ReactionSequenceResult:
    reaction_sequence: ReactionSequence
    compound: Compound
    smirks: str
