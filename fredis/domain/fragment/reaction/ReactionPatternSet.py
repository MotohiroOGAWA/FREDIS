from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple, Any

from fredis.libs.mmkit.mmkit import Compound

from ._ReactionPattern import _ReactionPattern, _ReactionResult, ProductRule


@dataclass(frozen=True)
class ReactionPattern(_ReactionPattern):
    """Public reaction pattern with stable ID."""

    pattern_id: int

    @classmethod
    def from_base(
        cls,
        pattern_id: int,
        base: _ReactionPattern,
    ) -> ReactionPattern:
        """Create public ReactionPattern from internal _ReactionPattern."""
        return cls(
            pattern_id=pattern_id,
            name=base.name,
            reactant_smarts=base.reactant_smarts,
            reactant_query=base.reactant_query,
            reaction_rules=base.reaction_rules,
        )

    def fragment(
        self,
        compound: Compound,
    ) -> ReactionResult | None:
        """Apply this pattern and return result with pattern_id."""
        result = super().fragment(compound)

        if result is None:
            return None

        if len(result.products) == 0:
            return None

        return ReactionResult.from_base(
            pattern_id=self.pattern_id,
            pattern=self,
            base=result,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "reactant_smarts": self.reactant_smarts,
            "products": [rule.source_rule.to_dict() for rule in self.reaction_rules],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> ReactionPattern:
        return cls.from_base(
            pattern_id=int(data["pattern_id"]),
            base=_ReactionPattern.from_rules(
                name=data["name"],
                reactant_smarts=data["reactant_smarts"],
                products=tuple(
                    ProductRule.from_dict(product_data)
                    for product_data in data["products"]
                ),
            ),
        )

    def copy(self) -> ReactionPattern:
        """Return a copy of this pattern."""
        return ReactionPattern.from_base(
            pattern_id=self.pattern_id,
            base=super().copy(),
        )


@dataclass(frozen=True)
class ReactionResult(_ReactionResult):
    """Public reaction result with pattern ID."""

    pattern_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_id, int):
            raise TypeError("pattern_id must be an integer")

        if self.pattern_id < 0:
            raise ValueError("pattern_id must be non-negative")

        if not isinstance(self.pattern, ReactionPattern):
            raise TypeError(
                "pattern must be a ReactionPattern. "
                f"Got {type(self.pattern).__name__}."
            )

    @classmethod
    def from_base(
        cls,
        pattern_id: int,
        pattern: object,
        base: _ReactionResult,
    ) -> ReactionResult:
        return cls(
            pattern_id=pattern_id,
            pattern=pattern,
            reactant_compound=base.reactant_compound,
            products=base.products,
        )

    def copy(self) -> ReactionResult:
        """Return a copy of this result."""
        return ReactionResult.from_base(
            pattern_id=self.pattern_id,
            pattern=self.pattern,
            base=super().copy(),
        )


@dataclass(frozen=True)
class ReactionPatternSet:
    """A stable set of reaction patterns.

    This class is intentionally a plain dataclass.
    It does not normalize patterns in __init__.

    Use from_patterns(...) to:
    - remove duplicate patterns
    - sort patterns by identity
    - assign stable pattern IDs
    """

    name: str
    patterns: Tuple[ReactionPattern, ...]

    @classmethod
    def from_patterns(
        cls,
        patterns: Iterable[_ReactionPattern],
        name: str = "",
    ) -> ReactionPatternSet:
        """Create a normalized ReactionPatternSet.

        Parameters
        ----------
        patterns:
            Internal reaction patterns without IDs.
        name:
            Human-readable name of this pattern set.

        Returns
        -------
        ReactionPatternSet
            Pattern set with stable pattern IDs.
        """
        unique_patterns: dict[
            str,
            _ReactionPattern,
        ] = {}

        for pattern in patterns:
            if not isinstance(pattern, _ReactionPattern):
                raise TypeError(
                    "patterns must contain only _ReactionPattern instances."
                )

            unique_patterns.setdefault(
                pattern.key,
                pattern,
            )

        sorted_patterns = tuple(unique_patterns[key] for key in sorted(unique_patterns))

        public_patterns = tuple(
            ReactionPattern.from_base(
                pattern_id=pattern_id,
                base=pattern,
            )
            for pattern_id, pattern in enumerate(sorted_patterns)
        )

        return cls(
            name=name,
            patterns=public_patterns,
        )

    def __iter__(self) -> Iterator[ReactionPattern]:
        return iter(self.patterns)

    def __len__(self) -> int:
        return len(self.patterns)

    def __contains__(self, pattern: object) -> bool:
        if not isinstance(pattern, _ReactionPattern):
            return False

        return pattern.key in self._pattern_id_by_identity

    @property
    def pattern_ids(self) -> Tuple[int, ...]:
        """Pattern IDs in this set."""
        return tuple(pattern.pattern_id for pattern in self.patterns)

    @property
    def _pattern_by_id(self) -> dict[int, ReactionPattern]:
        return {pattern.pattern_id: pattern for pattern in self.patterns}

    @property
    def _pattern_id_by_identity(self) -> dict[tuple[str, tuple[str, ...]], int]:
        return {pattern.key: pattern.pattern_id for pattern in self.patterns}

    def identity(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the identity of this pattern set.

        The set name is intentionally not included.
        """
        return tuple(pattern.key for pattern in self.patterns)

    def get_pattern(
        self,
        pattern_id: int,
    ) -> ReactionPattern:
        """Get a reaction pattern by stable pattern ID."""
        try:
            return self._pattern_by_id[pattern_id]
        except KeyError as exc:
            raise KeyError(f"Invalid reaction pattern ID: {pattern_id}") from exc

    def get_pattern_id(
        self,
        pattern: _ReactionPattern,
    ) -> int:
        """Get the stable ID of an equivalent reaction pattern."""
        try:
            return self._pattern_id_by_identity[pattern.key]
        except KeyError as exc:
            raise KeyError(f"ReactionPattern is not in this set: {pattern}") from exc

    def fragment_by_id(
        self,
        pattern_id: int,
        compound: Compound,
    ) -> ReactionResult | None:
        """Apply one reaction pattern to a compound."""
        pattern = self.get_pattern(pattern_id)
        return pattern.fragment(compound)

    def fragment_all(
        self,
        compound: Compound,
    ) -> Tuple[ReactionResult, ...]:
        """Apply all reaction patterns to a compound."""
        results: list[ReactionResult] = []

        for pattern in self.patterns:
            result = pattern.fragment(compound)

            if result is None:
                continue

            if len(result.products) == 0:
                continue

            results.append(result)

        return tuple(results)

    def to_dict(self) -> dict[str, object]:
        """Serialize this pattern set.

        RDKit compiled objects are not serialized here.
        Use this only for metadata-level export.
        """
        return {
            "name": self.name,
            "patterns": [pattern.to_dict() for pattern in self.patterns],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> ReactionPatternSet:
        """Deserialize a pattern set from a dictionary."""
        patterns = tuple(
            (
                ReactionPattern.from_dict(pattern_data)
                if ("pattern_id" in pattern_data)
                else _ReactionPattern.from_dict(pattern_data)
            )
            for pattern_data in data["patterns"]
        )

        if any(not isinstance(pattern, ReactionPattern) for pattern in patterns):
            return cls.from_patterns(
                patterns=patterns,
                name=data.get("name", ""),
            )
        else:
            return cls(
                name=data.get("name", ""),
                patterns=patterns,
            )

    def copy(self) -> ReactionPatternSet:
        """Return a copy of this pattern set."""
        return ReactionPatternSet(
            name=self.name,
            patterns=tuple(pattern.copy() for pattern in self.patterns),
        )
