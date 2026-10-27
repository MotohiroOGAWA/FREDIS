from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Tuple, Union, overload
import re

from rdkit import Chem  # noqa: F401

from .Formula import Formula
from ._parsing import charge_from_str


FormulaKey = Union[str, Formula]


@dataclass(frozen=True, slots=True)
class Adduct:
    """
    Representation of an adduct ion.

    An ``Adduct`` describes how a neutral molecule is transformed into an
    observed ion by tracking:

    - ion type, such as ``"M"`` or ``"F"``
    - number of molecules
    - signed formula counts
    - net charge

    Examples
    --------
    ``[M+H]+``
        Protonated molecule.

    ``[2M+Na]+``
        Sodium-adducted dimer.

    ``[M-H]-``
        Deprotonated molecule.

    ``Adduct.from_dict({"H": -5}, charge=-1)``
        Molecule losing five hydrogens: ``[M-5H]-``.
    """

    ion_type: Literal["M", "F"] = "M"
    n_molecules: int = 1
    formula_counts: Mapping[FormulaKey, int] = field(default_factory=dict)
    charge: int = 0

    def __post_init__(self) -> None:
        if self.ion_type not in ("M", "F"):
            raise ValueError(
                f"ion_type must be one of 'M' or 'F', but got {self.ion_type!r}."
            )

        if self.n_molecules < 1:
            raise ValueError("n_molecules must be at least 1.")

        normalized_counts: dict[Formula, int] = defaultdict(int)

        for formula, count in self.formula_counts.items():
            if count == 0:
                continue

            parsed_formula = self._normalize_formula_key(formula)
            normalized_counts[parsed_formula.plain] += int(count)

        normalized_counts = {
            formula.copy(): count
            for formula, count in normalized_counts.items()
            if count != 0
        }

        object.__setattr__(self, "formula_counts", normalized_counts)

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def parse(cls, adduct_str: str) -> Adduct:
        """
        Parse an adduct string.

        Parameters
        ----------
        adduct_str:
            Adduct string such as ``"[M+H]+"`` or ``"[2M-H]-"``.

        Returns
        -------
        Adduct
            Parsed adduct object.
        """
        if not adduct_str.startswith("[") or "]" not in adduct_str:
            raise ValueError(f"Invalid adduct format: {adduct_str}")

        main, charge_part = adduct_str[1:].split("]", maxsplit=1)
        charge = charge_from_str(charge_part.strip())

        n_match = re.match(r"(\d*)([A-Za-z]+)", main)
        if not n_match:
            raise ValueError(
                f"Invalid adduct format: missing molecule identifier in {adduct_str}"
            )

        n_molecules = int(n_match.group(1)) if n_match.group(1) else 1
        ion_type = n_match.group(2)

        if ion_type not in ("M", "F"):
            raise ValueError(
                f"ion_type must be one of 'M' or 'F', but got {ion_type!r}."
            )

        remainder = main[n_match.end():]

        pattern = re.compile(r"([+-])(\d*)([A-Z][A-Za-z0-9]*)|([+-])(\d*)(i)")
        formula_counts: dict[str, int] = defaultdict(int)

        for sign, num, formula_str, i_sign, i_num, i_formula_str in pattern.findall(
            remainder
        ):
            if i_formula_str == "i":
                formula_str = Formula.neutron().symbol
                sign = i_sign
                num = i_num

            count = int(num) if num else 1

            if sign == "+":
                formula_counts[formula_str] += count
            elif sign == "-":
                formula_counts[formula_str] -= count

        return cls(
            ion_type=ion_type,
            n_molecules=n_molecules,
            formula_counts=formula_counts,
            charge=charge,
        )

    @classmethod
    def from_dict(
        cls,
        formula_counts: Mapping[FormulaKey, int],
        *,
        ion_type: Literal["M", "F"] = "M",
        n_molecules: int = 1,
        charge: int = 0,
    ) -> Adduct:
        """
        Create an adduct from signed formula counts.

        Parameters
        ----------
        formula_counts:
            Mapping from formula to signed count.

            Positive count means the formula is added.
            Negative count means the formula is removed.

            Examples:
            ``{"H": 1}`` -> ``[M+H]+`` if charge is ``+1``.
            ``{"H": -1}`` -> ``[M-H]-`` if charge is ``-1``.
            ``{"H": -5}`` -> ``[M-5H]-`` if charge is ``-1``.

        ion_type:
            Ion type identifier.

        n_molecules:
            Number of neutral molecules.

        charge:
            Net charge of the ion.

        Returns
        -------
        Adduct
            Created adduct object.
        """
        return cls(
            ion_type=ion_type,
            n_molecules=n_molecules,
            formula_counts=formula_counts,
            charge=charge,
        )

    @classmethod
    def from_adducts(
        cls,
        *,
        ion_type: Literal["M", "F"] = "M",
        n_molecules: int = 1,
        adducts_in: Tuple[FormulaKey, ...] = (),
        adducts_out: Tuple[FormulaKey, ...] = (),
        charge: int = 0,
    ) -> Adduct:
        """
        Create an adduct from added and removed formulas.

        This is a compatibility constructor for the old ``adducts_in`` /
        ``adducts_out`` style.
        """
        formula_counts: dict[Formula, int] = defaultdict(int)

        for formula in adducts_in:
            parsed_formula = cls._normalize_formula_key(formula)
            formula_counts[parsed_formula.plain] += 1

        for formula in adducts_out:
            parsed_formula = cls._normalize_formula_key(formula)
            formula_counts[parsed_formula.plain] -= 1

        return cls(
            ion_type=ion_type,
            n_molecules=n_molecules,
            formula_counts=formula_counts,
            charge=charge,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def adduct_formulas(self) -> Dict[Formula, int]:
        """
        Signed counts of adduct formulas.

        Returns
        -------
        dict[Formula, int]
            Mapping from formula to signed count.
        """
        return {
            formula.copy(): count
            for formula, count in self.formula_counts.items()
        }

    @property
    def formula_diff(self) -> Formula:
        """
        Net formula difference introduced by the adduct.
        """
        total = Formula.empty()

        for formula, count in self.formula_counts.items():
            total = total + (formula * count)

        return total

    @property
    def mass_shift(self) -> float:
        """
        Exact mass shift introduced by the adduct.
        """
        return sum(
            formula.exact_mass * count
            for formula, count in self.formula_counts.items()
        )

    @property
    def element_diff(self) -> dict[str, int]:
        """
        Total element count difference introduced by the adduct.
        """
        total: dict[str, int] = defaultdict(int)

        for formula, count in self.formula_counts.items():
            for elem, elem_count in formula.elements.items():
                total[elem] += elem_count * count

        return dict(total)

    # -------------------------------------------------------------------------
    # Basic methods
    # -------------------------------------------------------------------------

    def copy(self) -> Adduct:
        """
        Create a copy of the adduct.
        """
        return Adduct(
            ion_type=self.ion_type,
            n_molecules=self.n_molecules,
            formula_counts=self.formula_counts,
            charge=self.charge,
        )

    def with_charge(self, charge: int) -> Adduct:
        """
        Return a new adduct with a different charge.

        Since this dataclass is frozen, this replaces the old mutable
        ``set_charge`` behavior.
        """
        return Adduct(
            ion_type=self.ion_type,
            n_molecules=self.n_molecules,
            formula_counts=self.formula_counts,
            charge=charge,
        )

    @overload
    def get_formula_count(self, formula: Formula) -> int: ...

    @overload
    def get_formula_count(self, formula: str) -> int: ...

    def get_formula_count(self, formula: FormulaKey) -> int:
        """
        Get the signed count of a specific formula.
        """
        parsed_formula = self._normalize_formula_key(formula)
        return self.formula_counts.get(parsed_formula.plain, 0)

    def to_dict(self) -> dict[str, int]:
        """
        Convert formula counts to a simple string-keyed dictionary.

        Examples
        --------
        ``[M-5H]-`` -> ``{"H": -5}``
        """
        return {
            formula.raw_formula.replace("+", "").replace("-", ""): count
            for formula, count in self.formula_counts.items()
        }

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Convert the adduct to canonical string form.
        """
        parts: list[str] = []

        for formula, count in sorted(
            self.formula_counts.items(),
            key=lambda x: (-x[0].exact_mass, x[0].raw_formula),
        ):
            formula_str = formula.raw_formula.replace("+", "").replace("-", "")

            if count > 0:
                prefix = f"+{count}" if count > 1 else "+"
                parts.append(f"{prefix}{formula_str}")
            elif count < 0:
                prefix = f"-{abs(count)}" if count < -1 else "-"
                parts.append(f"{prefix}{formula_str}")

        body = "".join(parts)
        molecule_part = (
            f"{self.n_molecules if self.n_molecules > 1 else ''}{self.ion_type}"
        )

        if self.charge > 0:
            charge_part = f"+{self.charge}" if self.charge > 1 else "+"
        elif self.charge < 0:
            charge_part = f"{self.charge}" if self.charge < -1 else "-"
        else:
            charge_part = ""

        return f"[{molecule_part}{body}]{charge_part}"

    def __repr__(self) -> str:
        """
        Convert the adduct to debug representation.
        """
        return f"Adduct({self})"

    def __eq__(self, other: object) -> bool:
        """
        Compare two adducts by canonical string representation.
        """
        if not isinstance(other, Adduct):
            return False

        return str(self) == str(other)


    def __hash__(self) -> int:
        """
        Compute hash by canonical string representation.

        This avoids hashing formula_counts directly, because it is stored as a dict.
        """
        return hash(str(self))

    # -------------------------------------------------------------------------
    # Adduct combination
    # -------------------------------------------------------------------------

    def add(
        self,
        other: Adduct,
        *,
        prefer_ion_type: bool = False,
        prefer_n_molecules: bool = False,
        prefer_charge: bool = False,
    ) -> Adduct:
        """
        Combine this adduct with another adduct.
        """
        if not isinstance(other, Adduct):
            raise TypeError(f"Can only add Adduct to Adduct, but got {type(other)}")

        if not prefer_ion_type and self.ion_type != other.ion_type:
            raise ValueError(
                f"Cannot add Adducts with different ion_type: "
                f"{self.ion_type} + {other.ion_type}"
            )

        if not prefer_n_molecules and self.n_molecules != other.n_molecules:
            raise ValueError(
                f"Cannot add Adducts with different n_molecules: "
                f"{self.n_molecules} + {other.n_molecules}"
            )

        if not prefer_charge and self.charge != other.charge:
            raise ValueError(
                f"Cannot add Adducts with different charge: "
                f"{self.charge} + {other.charge}"
            )

        merged: dict[Formula, int] = defaultdict(int)

        for formula, count in self.formula_counts.items():
            merged[formula.plain] += count

        for formula, count in other.formula_counts.items():
            merged[formula.plain] += count

        return Adduct(
            ion_type=self.ion_type,
            n_molecules=self.n_molecules,
            formula_counts=merged,
            charge=self.charge,
        )

    def add_prefer_self(self, other: Adduct) -> Adduct:
        """
        Combine two adducts while keeping this adduct's metadata.
        """
        return self.add(
            other,
            prefer_ion_type=True,
            prefer_n_molecules=True,
            prefer_charge=True,
        )

    # -------------------------------------------------------------------------
    # Formula and mass calculations
    # -------------------------------------------------------------------------

    def apply_to_formula(self, neutral_formula: Formula) -> Formula:
        """
        Apply the adduct to a neutral formula.

        Parameters
        ----------
        neutral_formula:
            Neutral molecular formula.

        Returns
        -------
        Formula
            Resulting ion formula.
        """
        total_formula = neutral_formula * self.n_molecules + self.formula_diff
        return total_formula.with_charge(self.charge)

    def apply_to_mass(self, neutral_mass: float) -> float:
        """
        Apply the adduct to a neutral mass.

        Returns the ion mass before charge division.
        """
        return neutral_mass * self.n_molecules + self.mass_shift

    def apply_to_mz(self, neutral_mass: float) -> float:
        """
        Apply the adduct to a neutral mass and compute m/z.
        """
        if self.charge == 0:
            raise ValueError(f"Cannot calculate m/z for uncharged adduct: {self}")

        return self.apply_to_mass(neutral_mass) / abs(self.charge)

    # -------------------------------------------------------------------------
    # Splitting helpers
    # -------------------------------------------------------------------------

    def _split_adduct_formulas(self) -> Tuple[Tuple[Formula, ...], Tuple[Formula, ...]]:
        """
        Split signed formula counts into input and output tuples.
        """
        adducts_in: list[Formula] = []
        adducts_out: list[Formula] = []

        for formula, count in self.formula_counts.items():
            if count > 0:
                adducts_in.extend([formula.copy()] * count)
            elif count < 0:
                adducts_out.extend([formula.copy()] * (-count))

        return tuple(adducts_in), tuple(adducts_out)

    def split(self, split_each: bool = False) -> Tuple[Adduct, ...]:
        """
        Split the adduct into independent components.
        """
        result: list[Adduct] = []
        adducts_in, adducts_out = self._split_adduct_formulas()

        if adducts_in:
            if split_each:
                for formula in adducts_in:
                    result.append(
                        Adduct.from_adducts(
                            ion_type=self.ion_type,
                            n_molecules=self.n_molecules,
                            adducts_in=(formula,),
                            charge=0,
                        )
                    )
            else:
                result.append(
                    Adduct.from_adducts(
                        ion_type=self.ion_type,
                        n_molecules=self.n_molecules,
                        adducts_in=adducts_in,
                        charge=0,
                    )
                )

        if adducts_out:
            if split_each:
                for formula in adducts_out:
                    result.append(
                        Adduct.from_adducts(
                            ion_type=self.ion_type,
                            n_molecules=self.n_molecules,
                            adducts_out=(formula,),
                            charge=0,
                        )
                    )
            else:
                result.append(
                    Adduct.from_adducts(
                        ion_type=self.ion_type,
                        n_molecules=self.n_molecules,
                        adducts_out=adducts_out,
                        charge=0,
                    )
                )

        return tuple(result)

    @classmethod
    def split_by_reference_adducts(
        cls,
        adduct: Adduct,
        reference_adducts: Tuple[Adduct, ...],
    ) -> Tuple[Tuple[bool, ...], Adduct]:
        """
        Split an adduct into reference-matched flags and a residual component.

        A reference adduct matches only when:

        - ion_type is the same
        - n_molecules is the same
        - charge direction is the same
        - all formula counts in the reference adduct exactly match the target adduct
        """
        if adduct.charge == 0:
            raise ValueError(f"Adduct must have non-zero charge: {adduct}")

        remaining_formula_counts: dict[Formula, int] = defaultdict(int)

        for formula, count in adduct.formula_counts.items():
            remaining_formula_counts[formula.plain] += count

        matched_flags: list[bool] = []

        for reference_adduct in reference_adducts:
            is_same_base_adduct = (
                reference_adduct.ion_type == adduct.ion_type
                and reference_adduct.n_molecules == adduct.n_molecules
                and reference_adduct.charge * adduct.charge > 0
            )

            if not is_same_base_adduct:
                matched_flags.append(False)
                continue

            reference_formula_counts = {
                formula: count
                for formula, count in reference_adduct.formula_counts.items()
                if count != 0
            }

            is_matched = True

            for formula, reference_count in reference_formula_counts.items():
                target_count = remaining_formula_counts.get(formula.plain, 0)

                if target_count != reference_count:
                    is_matched = False
                    break

            if not is_matched:
                matched_flags.append(False)
                continue

            matched_flags.append(True)

            for formula, reference_count in reference_formula_counts.items():
                remaining_formula_counts[formula.plain] -= reference_count

        residual_component = cls(
            ion_type=adduct.ion_type,
            n_molecules=1,
            formula_counts=remaining_formula_counts,
            charge=0,
        )

        return tuple(matched_flags), residual_component

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_formula_key(formula: FormulaKey) -> Formula:
        """
        Normalize a formula key to a plain Formula object.
        """
        if isinstance(formula, str):
            return Formula.parse(formula).plain

        if isinstance(formula, Formula):
            return formula.plain

        raise TypeError(f"formula must be str or Formula, got {type(formula)}")