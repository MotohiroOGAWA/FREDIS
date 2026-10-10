from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from ....libs.mmkit.mmkit import Adduct
from .FragmentIonAdductRule import FragmentIonAdductRule


@dataclass(frozen=True)
class FragmentIonAdductRuleSet:
    """Adduct rules plus one shared, inclusive hydrogen-shift range."""

    name: str
    hydrogen_shift_min: int
    hydrogen_shift_max: int
    adduct_rules: Tuple[FragmentIonAdductRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if isinstance(self.hydrogen_shift_min, bool) or not isinstance(self.hydrogen_shift_min, int):
            raise TypeError("hydrogen_shift_min must be an int")
        if isinstance(self.hydrogen_shift_max, bool) or not isinstance(self.hydrogen_shift_max, int):
            raise TypeError("hydrogen_shift_max must be an int")
        if self.hydrogen_shift_min > self.hydrogen_shift_max:
            raise ValueError("hydrogen_shift_min must not exceed hydrogen_shift_max")
        if not isinstance(self.adduct_rules, tuple):
            raise TypeError("adduct_rules must be a tuple")
        if not all(isinstance(rule, FragmentIonAdductRule) for rule in self.adduct_rules):
            raise TypeError("adduct_rules must contain only FragmentIonAdductRule values")
        keys = [str(rule.adduct_type) for rule in self.adduct_rules]
        if len(keys) != len(set(keys)):
            raise ValueError("adduct_rules contains duplicated adduct_type values")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragmentIonAdductRuleSet":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        payload = data.get("fragment_ion_adduct_rule_set", data)
        if not isinstance(payload, dict):
            raise TypeError("fragment_ion_adduct_rule_set must be a dict")
        return cls(
            name=str(payload.get("name", "")),
            hydrogen_shift_min=int(payload["hydrogen_shift_min"]),
            hydrogen_shift_max=int(payload["hydrogen_shift_max"]),
            adduct_rules=tuple(
                FragmentIonAdductRule.from_dict(value)
                for value in payload.get("adduct_rules", ())
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hydrogen_shift_min": self.hydrogen_shift_min,
            "hydrogen_shift_max": self.hydrogen_shift_max,
            "adduct_rules": [rule.to_dict() for rule in self.adduct_rules],
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "FragmentIonAdductRuleSet":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)

    @property
    def hydrogen_shifts(self) -> Tuple[int, ...]:
        return tuple(range(self.hydrogen_shift_min, self.hydrogen_shift_max + 1))

    @property
    def rules(self) -> Tuple[FragmentIonAdductRule, ...]:
        return self.adduct_rules

    @property
    def num_classes(self) -> int:
        return len(self.adduct_rules)

    @property
    def adduct_types(self) -> Tuple[Adduct, ...]:
        return tuple(rule.adduct_type for rule in self.adduct_rules)

    @property
    def adduct_type_to_index(self) -> Dict[Adduct, int]:
        return {rule.adduct_type: index for index, rule in enumerate(self.adduct_rules)}

    @property
    def index_to_adduct_type(self) -> Dict[int, Adduct]:
        return {index: rule.adduct_type for index, rule in enumerate(self.adduct_rules)}

    def get_rule_by_adduct_type(self, adduct_type: Adduct | str) -> FragmentIonAdductRule:
        return self.get_rule_by_index(self.encode_adduct_type(adduct_type))

    def get_rule_by_index(self, index: int) -> FragmentIonAdductRule:
        if not isinstance(index, int):
            raise TypeError("index must be an int")
        if not 0 <= index < self.num_classes:
            raise IndexError(f"index must be in [0, {self.num_classes}); got {index}")
        return self.adduct_rules[index]

    def get_fragment_ion_adducts_by_adduct_type(self, adduct_type: Adduct | str) -> Tuple[Adduct, ...]:
        return self.get_rule_by_adduct_type(adduct_type).fragment_ion_adducts

    def get_fragment_ion_adducts_by_index(self, index: int) -> Tuple[Adduct, ...]:
        return self.get_rule_by_index(index).fragment_ion_adducts

    def encode_adduct_type(self, adduct_type: Adduct | str) -> int:
        key = adduct_type if isinstance(adduct_type, Adduct) else Adduct.parse(adduct_type)
        try:
            return self.adduct_type_to_index[key]
        except KeyError as exc:
            raise KeyError(f"Unknown adduct_type: {key}") from exc

    def decode_adduct_index(self, index: int) -> Adduct:
        return self.get_rule_by_index(index).adduct_type

    def copy(self) -> "FragmentIonAdductRuleSet":
        return FragmentIonAdductRuleSet(
            name=self.name,
            hydrogen_shift_min=self.hydrogen_shift_min,
            hydrogen_shift_max=self.hydrogen_shift_max,
            adduct_rules=tuple(rule.copy() for rule in self.adduct_rules),
        )
