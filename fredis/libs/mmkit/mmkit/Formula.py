from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Union

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Formula:
    """
    Represent a chemical formula with element counts and net charge.

    A ``Formula`` stores an elemental composition together with a formal charge.
    It supports parsing from strings, arithmetic operations between formulas,
    exact-mass calculation, and conversion back to canonical string form.

    Notes
    -----
    - Elements are stored in Hill order.
    - Negative element counts are allowed for formula arithmetic.
    - ``raw_formula`` can preserve the original input string when needed.
    """

    _elements: OrderedDict[str, int] = field(init=False, repr=False)
    _charge: int = field(init=False, repr=False)
    _raw_formula: str = field(init=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        elements: Mapping[str, int],
        charge: int = 0,
        raw_formula: str = "",
    ) -> None:
        """
        Initialize a Formula.

        Parameters
        ----------
        elements:
            Mapping from element symbol to count.

        charge:
            Net charge of the formula.

        raw_formula:
            Original formula string stored for reference.

        Notes
        -----
        Element keys are reordered into Hill order and zero-count elements
        are removed during initialization.
        """
        ordered_elements = self._build_ordered_elements(dict(elements))

        object.__setattr__(self, "_elements", ordered_elements)
        object.__setattr__(self, "_charge", int(charge))
        object.__setattr__(self, "_raw_formula", raw_formula)

    @classmethod
    def parse(cls, formula_str: str, store_raw: bool = False) -> Formula:
        """
        Parse a formula string into a Formula object.

        Parameters
        ----------
        formula_str:
            Formula string such as ``C6H12O6`` or ``C6H12O6+``.

        store_raw:
            Whether to store the original input string in ``raw_formula``.

        Returns
        -------
        Formula
            Parsed Formula instance.
        """
        formula_body, charge = cls._split_formula_and_charge(formula_str)
        elements = cls._parse_element_counts(formula_body)

        return cls(
            elements=elements,
            charge=charge,
            raw_formula=formula_str if store_raw else "",
        )

    @classmethod
    def from_dict(
        cls,
        elements: Mapping[str, int],
        *,
        charge: int = 0,
        raw_formula: str = "",
    ) -> Formula:
        """
        Create a Formula from an element-count dictionary.

        Examples
        --------
        ``Formula.from_dict({"C": 6, "H": 12, "O": 6})``
        -> ``C6H12O6``

        ``Formula.from_dict({"H": -5})``
        -> ``-H5``
        """
        return cls(
            elements=elements,
            charge=charge,
            raw_formula=raw_formula,
        )

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> Formula:
        """
        Create a Formula from an RDKit molecule.

        Parameters
        ----------
        mol:
            RDKit molecule object.

        Returns
        -------
        Formula
            Formula derived from the molecule.

        Notes
        -----
        Implicit hydrogens are converted to explicit hydrogens before counting
        atoms, so hydrogen counts are included in the result.
        """
        mol_with_h = Chem.AddHs(mol)

        elements: dict[str, int] = {}
        total_charge = 0

        for atom in mol_with_h.GetAtoms():
            symbol = atom.GetSymbol()
            elements[symbol] = elements.get(symbol, 0) + 1
            total_charge += atom.GetFormalCharge()

        return cls(elements=elements, charge=total_charge)

    @classmethod
    def empty(cls) -> Formula:
        """
        Return an empty formula.
        """
        return cls(elements={}, charge=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def elements(self) -> OrderedDict[str, int]:
        """
        Element-count mapping in Hill order.

        Returns
        -------
        OrderedDict[str, int]
            Copy of the internal mapping from element symbol to count.
        """
        return OrderedDict(self._elements)

    @property
    def charge(self) -> int:
        """
        Net charge of the formula.
        """
        return self._charge

    @property
    def raw_formula(self) -> str:
        """
        Original formula string, if available.

        If no raw string was stored, the canonical formula string is returned.
        """
        if self._raw_formula == "":
            return self.to_string(no_charge=False)

        return self._raw_formula

    @property
    def exact_mass(self) -> float:
        """
        Monoisotopic exact mass of the formula.
        """
        periodic_table = Chem.GetPeriodicTable()
        mass = 0.0

        for elem, count in self._elements.items():
            atomic_number = periodic_table.GetAtomicNumber(elem)
            mass += periodic_table.GetMostCommonIsotopeMass(atomic_number) * count

        return mass

    @property
    def is_nonnegative(self) -> bool:
        """
        Whether all element counts are non-negative.
        """
        return all(count >= 0 for count in self._elements.values())

    @property
    def value(self) -> str:
        """
        Canonical formula string including charge.
        """
        return self.to_string(no_charge=False)

    @property
    def plain_value(self) -> str:
        """
        Canonical formula string without charge.
        """
        return self.to_string(no_charge=True)

    @property
    def plain(self) -> Formula:
        """
        Copy of the formula with zero charge.
        """
        return Formula(
            elements=self._elements,
            charge=0,
            raw_formula=self._raw_formula,
        )

    @property
    def normalized(self) -> Formula:
        """
        Copy with ``raw_formula`` cleared.
        """
        return Formula(
            elements=self._elements,
            charge=self._charge,
            raw_formula="",
        )

    @property
    def normalized_plain(self) -> Formula:
        """
        Copy with zero charge and empty ``raw_formula``.
        """
        return Formula(
            elements=self._elements,
            charge=0,
            raw_formula="",
        )

    # ------------------------------------------------------------------
    # Mutation-like utilities for frozen dataclass
    # ------------------------------------------------------------------
    def with_charge(self, charge: int) -> Formula:
        """
        Return a new Formula with a different charge.

        Since Formula is frozen, this should be used instead of directly
        modifying ``_charge``.
        """
        return Formula(
            elements=self._elements,
            charge=charge,
            raw_formula=self._raw_formula,
        )

    def with_raw_formula(self, raw_formula: str) -> Formula:
        """
        Return a new Formula with a different raw formula string.
        """
        return Formula(
            elements=self._elements,
            charge=self._charge,
            raw_formula=raw_formula,
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """
        Return the debug representation.
        """
        return f"Formula({self})"

    def __str__(self) -> str:
        """
        Return the canonical formula string.
        """
        return self.to_string(no_charge=False)

    def to_string(self, no_charge: bool = False) -> str:
        """
        Convert the formula to a string.

        Parameters
        ----------
        no_charge:
            If True, omit the charge suffix.

        Returns
        -------
        str
            Canonical formula string.

        Notes
        -----
        Elements are emitted in Hill order. Negative element counts are written
        with a leading ``-``.
        """
        parts: list[str] = []

        for elem, count in self._elements.items():
            if count > 0:
                parts.append(f"{elem}{count if count != 1 else ''}")
            elif count < 0:
                parts.append(f"-{elem}{-count if count != -1 else ''}")

        formula = "".join(parts)

        if not no_charge:
            if self._charge > 0:
                formula += "+" if self._charge == 1 else f"+{self._charge}"
            elif self._charge < 0:
                formula += "-" if self._charge == -1 else f"-{-self._charge}"

        return formula

    # ------------------------------------------------------------------
    # Comparison / hashing
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        """
        Return whether two Formula objects are equal.

        ``raw_formula`` is ignored for normal equality.
        """
        if not isinstance(other, Formula):
            return False

        return (
            self._charge == other._charge
            and self._elements == other._elements
        )

    def __hash__(self) -> int:
        """
        Return the hash value.

        ``raw_formula`` is intentionally ignored because ``__eq__`` also ignores it.
        This keeps the hash contract consistent.
        """
        return hash((tuple(self._elements.items()), self._charge))

    def eq(self, other: Formula, ignore_raw: bool = True) -> bool:
        """
        Compare two Formula objects with optional raw-string comparison.

        Parameters
        ----------
        other:
            Formula to compare with.

        ignore_raw:
            If False, also compare ``raw_formula``.

        Returns
        -------
        bool
            True if the two Formula objects are considered equal.
        """
        if not isinstance(other, Formula):
            return False

        base_equal = self == other

        if ignore_raw:
            return base_equal

        return base_equal and self._raw_formula == other._raw_formula

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: Formula) -> Formula:
        """
        Return the sum of two formulas.
        """
        if not isinstance(other, Formula):
            return NotImplemented

        combined = dict(self._elements)

        for elem, count in other._elements.items():
            combined[elem] = combined.get(elem, 0) + count

        return Formula(
            elements=combined,
            charge=self._charge + other._charge,
        )

    def __sub__(self, other: Formula) -> Formula:
        """
        Return the difference of two formulas.
        """
        if not isinstance(other, Formula):
            return NotImplemented

        combined = dict(self._elements)

        for elem, count in other._elements.items():
            combined[elem] = combined.get(elem, 0) - count

        return Formula(
            elements=combined,
            charge=self._charge - other._charge,
        )

    def __mul__(self, factor: int) -> Formula:
        """
        Multiply the formula by an integer.

        Examples
        --------
        ``H2O * 2 -> H4O2``
        """
        if not isinstance(factor, int):
            raise TypeError(f"Formula can only be multiplied by int, not {type(factor)}")

        new_elements = {
            elem: count * factor
            for elem, count in self._elements.items()
        }

        return Formula(
            elements=new_elements,
            charge=self._charge * factor,
            raw_formula=self._raw_formula,
        )

    def __rmul__(self, factor: int) -> Formula:
        """
        Support reversed multiplication.
        """
        return self.__mul__(factor)

    # ------------------------------------------------------------------
    # Predicates / membership
    # ------------------------------------------------------------------
    def __contains__(self, item: Union[Formula, str]) -> bool:
        """
        Return whether this formula contains another formula.

        Containment checks are only supported for non-negative formulas.
        """
        if not self.is_nonnegative:
            raise ValueError("Containment check is only supported for non-negative formulas.")

        if isinstance(item, str):
            item = Formula.parse(item)
        elif not isinstance(item, Formula):
            raise TypeError(
                f"Containment check only supports Formula or str, not {type(item)}"
            )

        return all(
            self._elements.get(elem, 0) >= count
            for elem, count in item._elements.items()
        )

    # ------------------------------------------------------------------
    # Copy utilities
    # ------------------------------------------------------------------
    def copy(self) -> Formula:
        """
        Create a copy of the formula.
        """
        return Formula(
            elements=self._elements,
            charge=self._charge,
            raw_formula=self._raw_formula,
        )

    def to_dict(self) -> dict[str, int]:
        """
        Convert the formula to a plain element-count dictionary.
        """
        return dict(self._elements)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @classmethod
    def _build_ordered_elements(
        cls,
        element_counts: Mapping[str, int],
    ) -> OrderedDict[str, int]:
        """
        Reorder elements according to the Hill system and remove zero counts.
        """
        nonzero_counts = {
            elem: int(count)
            for elem, count in element_counts.items()
            if int(count) != 0
        }

        ordered_keys = cls._reorder_element_keys(nonzero_counts.keys())

        return OrderedDict(
            (key, nonzero_counts[key])
            for key in ordered_keys
            if nonzero_counts[key] != 0
        )

    @staticmethod
    def _split_formula_and_charge(formula: str) -> tuple[str, int]:
        """
        Split a formula string into formula body and charge.

        Examples
        --------
        ``C6H12O6+`` -> (``C6H12O6``, 1)
        ``Cl-`` -> (``Cl``, -1)
        ``Fe+2`` -> (``Fe``, 2)
        """
        charge = 0
        charge_match = re.search(r"([+-]+|[+-]\d+)$", formula)

        if charge_match:
            charge_str = charge_match.group(1)
            formula = formula[: -len(charge_str)]

            charge = int(charge_str[1:]) if charge_str[1:] else 1

            if charge_str[0] == "-":
                charge *= -1

        return formula, charge

    @staticmethod
    def _parse_element_counts(formula: str) -> dict[str, int]:
        """
        Parse element counts from a neutral formula body.

        Negative terms such as ``-H2`` are supported.
        """
        matches = re.findall(
            r"([+-]?)([A-Z][a-z]?)(\d*)",
            formula,
        )

        temp: dict[str, int] = {}

        for sign, elem, count in matches:
            value = int(count) if count else 1

            if sign == "-":
                value = -value

            temp[elem] = temp.get(elem, 0) + value

        return {
            elem: count
            for elem, count in temp.items()
            if count != 0
        }

    @staticmethod
    def _reorder_element_keys(elements: Iterable[str]) -> tuple[str, ...]:
        """
        Reorder element symbols according to the Hill system.
        """
        element_tuple = tuple(elements)

        if not element_tuple:
            return tuple()

        mol = Chem.RWMol()

        for elem in element_tuple:
            atom = Chem.Atom(elem)
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)

        formula_str = rdMolDescriptors.CalcMolFormula(mol.GetMol())
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)

        return tuple(match[0] for match in matches)