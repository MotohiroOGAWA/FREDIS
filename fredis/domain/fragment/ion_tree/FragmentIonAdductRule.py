from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ....libs.mmkit.mmkit import Adduct, Compound


@dataclass(frozen=True)
class FragmentIonAdductRule:
    """Fragment-ion adducts allowed for one precursor adduct.

    Hydrogen movement is deliberately owned by ``FragmentIonAdductRuleSet``.
    """

    name: str
    adduct_type: Adduct
    fragment_ion_adducts: Tuple[Adduct, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.adduct_type, Adduct):
            raise TypeError("adduct_type must be an Adduct")
        if not isinstance(self.fragment_ion_adducts, tuple):
            raise TypeError("fragment_ion_adducts must be a tuple")
        if not all(isinstance(value, Adduct) for value in self.fragment_ion_adducts):
            raise TypeError("fragment_ion_adducts must contain only Adduct values")
        keys = tuple(str(value) for value in self.fragment_ion_adducts)
        if len(keys) != len(set(keys)):
            raise ValueError("fragment_ion_adducts must not contain duplicates")
        if not self.fragment_ion_adducts:
            raise ValueError("fragment_ion_adducts must not be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragmentIonAdductRule":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(
            name=str(data["name"]),
            adduct_type=Adduct.parse(str(data["adduct_type"])),
            fragment_ion_adducts=tuple(
                Adduct.parse(str(value)) for value in data["fragment_ion_adducts"]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "adduct_type": str(self.adduct_type),
            "fragment_ion_adducts": [str(value) for value in self.fragment_ion_adducts],
        }

    def copy(self) -> "FragmentIonAdductRule":
        return FragmentIonAdductRule(
            name=self.name,
            adduct_type=self.adduct_type.copy(),
            fragment_ion_adducts=tuple(
                value.copy() for value in self.fragment_ion_adducts
            ),
        )
