from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ....libs.mmkit.mmkit import Formula


@dataclass(frozen=True)
class IonExplanation:
    """One ``(fragment, fragment-ion adduct, ΔH)`` explanation."""

    fragment_index: int
    adduct_index: int
    hydrogen_shift: int

    def __post_init__(self) -> None:
        if self.fragment_index < 0 or self.adduct_index < 0:
            raise ValueError("fragment_index and adduct_index must be non-negative")


@dataclass(frozen=True)
class PhysicalIonCandidate:
    """A unique final composition/charge retaining every explanation."""

    fragment_index: int
    formula: Formula
    charge: int
    explanations: Tuple[IonExplanation, ...]

    def __post_init__(self) -> None:
        if self.fragment_index < 0:
            raise ValueError("fragment_index must be non-negative")
        if not self.explanations:
            raise ValueError("a physical ion needs at least one explanation")
        if any(
            value.fragment_index != self.fragment_index for value in self.explanations
        ):
            raise ValueError(
                "all explanations must belong to the physical ion fragment"
            )
