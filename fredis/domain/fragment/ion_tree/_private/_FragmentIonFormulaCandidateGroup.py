from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from .....libs.mmkit.mmkit import Adduct, Formula

if TYPE_CHECKING:
    from ..FragmentIonAdductRule import FragmentIonAdductRule
    from ..FragmentIonTree import FragmentIonTree


@dataclass(frozen=True)
class _FragmentIonFormulaCandidateGroup:
    """Formula-indexed node/adduct candidates for one main adduct type.

    For a formula_index:

        start = indptr[formula_index]
        end = indptr[formula_index + 1]

    candidate ``i`` in ``start:end`` explains ``formulas[formula_index]`` as
    node ``node_indices[i]`` with fragment-ion adduct
    ``fragment_ion_adducts[fragment_ion_adduct_positions[i]]`` and hydrogen
    shift ``hydrogen_shifts[i]``; ``candidate_adducts[candidate_adduct_indices[i]]``
    is their combined adduct.
    """

    adduct_type: Adduct
    fragment_ion_adducts: Tuple[Adduct, ...]
    formulas: np.ndarray

    indptr: np.ndarray
    node_indices: np.ndarray
    candidate_adduct_indices: np.ndarray
    fragment_ion_adduct_positions: np.ndarray
    hydrogen_shifts: np.ndarray

    candidate_adducts: Tuple[Adduct, ...]

    def __post_init__(self) -> None:
        formulas = np.asarray(self.formulas, dtype=object)
        indptr = np.asarray(self.indptr, dtype=np.int64)
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.int64)
            for name in ("node_indices", "candidate_adduct_indices",
                         "fragment_ion_adduct_positions", "hydrogen_shifts")
        }

        if formulas.ndim != 1:
            raise ValueError("formulas must be a 1D array.")
        if indptr.ndim != 1 or len(indptr) != len(formulas) + 1:
            raise ValueError("indptr length must be len(formulas) + 1.")
        if indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("indptr must start at 0 and be non-decreasing.")
        for name, array in arrays.items():
            if len(array) != int(indptr[-1]):
                raise ValueError(f"{name} length must match indptr[-1].")
        if np.any(arrays["node_indices"] < 0):
            raise ValueError("node_indices must be non-negative.")
        indices = arrays["candidate_adduct_indices"]
        if np.any(indices < 0) or np.any(indices >= len(self.candidate_adducts)):
            raise ValueError("candidate_adduct_indices contains out-of-range values.")
        positions = arrays["fragment_ion_adduct_positions"]
        if np.any(positions < 0) or np.any(positions >= len(self.fragment_ion_adducts)):
            raise ValueError("fragment_ion_adduct_positions contains out-of-range values.")

        object.__setattr__(self, "fragment_ion_adducts", tuple(self.fragment_ion_adducts))
        object.__setattr__(self, "candidate_adducts", tuple(self.candidate_adducts))
        object.__setattr__(self, "formulas", formulas)
        object.__setattr__(self, "indptr", indptr)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @classmethod
    def from_fragment_ion_tree(
        cls,
        *,
        fragment_ion_tree: FragmentIonTree,
        adduct_rule: FragmentIonAdductRule,
    ) -> _FragmentIonFormulaCandidateGroup:
        """Build a formula candidate group for one main adduct rule."""

        hydrogen_shifts = fragment_ion_tree.hydrogen_shifts
        hydrogen_shift_adducts = fragment_ion_tree.hydrogen_shift_adducts
        fragment_ion_adducts = adduct_rule.fragment_ion_adducts

        formula_to_candidates: dict[Formula, list[tuple[int, int, int, int]]] = {}
        candidate_adducts: list[Adduct] = []
        candidate_adduct_key_to_index: dict[str, int] = {}

        for node_index, compound in enumerate(fragment_ion_tree.fragment_compounds):
            # Charged fragments take no fragment-ion adduct.
            if compound.charge != 0:
                continue
            base_formula = compound.formula.normalized

            for hydrogen_shift, hydrogen_shift_adduct in zip(hydrogen_shifts, hydrogen_shift_adducts):
                for position, fragment_ion_adduct in enumerate(fragment_ion_adducts):
                    formula = fragment_ion_adduct.apply_to_formula(
                        hydrogen_shift_adduct.apply_to_formula(base_formula)
                    ).normalized

                    if not formula.is_nonnegative:
                        continue

                    candidate_adduct_index = cls._register_adduct(
                        adduct=fragment_ion_adduct.add_prefer_self(hydrogen_shift_adduct),
                        adducts=candidate_adducts,
                        adduct_key_to_index=candidate_adduct_key_to_index,
                    )

                    formula_to_candidates.setdefault(formula, []).append(
                        (node_index, candidate_adduct_index, position, int(hydrogen_shift))
                    )

        formulas: list[Formula] = []
        indptr: list[int] = [0]
        rows: list[tuple[int, int, int, int]] = []

        for formula in sorted(formula_to_candidates, key=lambda value: value.exact_mass):
            formulas.append(formula)
            rows.extend(formula_to_candidates[formula])
            indptr.append(len(rows))

        columns = np.asarray(rows, dtype=np.int64).reshape(-1, 4)

        return cls(
            adduct_type=adduct_rule.adduct_type,
            fragment_ion_adducts=fragment_ion_adducts,
            formulas=np.asarray(formulas, dtype=object),
            indptr=np.asarray(indptr, dtype=np.int64),
            node_indices=columns[:, 0],
            candidate_adduct_indices=columns[:, 1],
            fragment_ion_adduct_positions=columns[:, 2],
            hydrogen_shifts=columns[:, 3],
            candidate_adducts=tuple(candidate_adducts),
        )

    @staticmethod
    def _formula_key(formula: Formula) -> Formula:
        return formula.normalized

    @staticmethod
    def _register_adduct(
        *,
        adduct: Adduct,
        adducts: list[Adduct],
        adduct_key_to_index: dict[str, int],
    ) -> int:
        key = str(adduct)

        existing_index = adduct_key_to_index.get(key)

        if existing_index is not None:
            return existing_index

        adduct_index = len(adducts)
        adducts.append(adduct)
        adduct_key_to_index[key] = adduct_index

        return adduct_index

    @property
    def num_formulas(self) -> int:
        return int(self.formulas.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.node_indices.shape[0])

    def get_formula_index(
        self,
        formula: Formula,
    ) -> int:
        formula_key = self._formula_key(formula)

        for formula_index, current_formula in enumerate(self.formulas):
            if self._formula_key(current_formula) == formula_key:
                return formula_index

        raise KeyError(f"Unknown formula: {formula_key}")

    def get_candidate_range(
        self,
        formula_index: int,
    ) -> slice:
        self._validate_formula_index(formula_index)

        start = int(self.indptr[formula_index])
        end = int(self.indptr[formula_index + 1])

        return slice(start, end)

    def get_node_indices_by_formula_index(
        self,
        formula_index: int,
    ) -> np.ndarray:
        candidate_range = self.get_candidate_range(formula_index)
        return self.node_indices[candidate_range]

    def get_candidate_adducts_by_formula_index(
        self,
        formula_index: int,
    ) -> Tuple[Adduct, ...]:
        candidate_range = self.get_candidate_range(formula_index)

        return tuple(
            self.candidate_adducts[int(candidate_adduct_index)]
            for candidate_adduct_index in (
                self.candidate_adduct_indices[candidate_range]
            )
        )

    def get_candidates_by_formula_index(
        self,
        formula_index: int,
    ) -> tuple[tuple[int, Adduct], ...]:
        candidate_range = self.get_candidate_range(formula_index)

        return tuple(
            (
                int(node_index),
                self.candidate_adducts[int(candidate_adduct_index)],
            )
            for node_index, candidate_adduct_index in zip(
                self.node_indices[candidate_range],
                self.candidate_adduct_indices[candidate_range],
            )
        )

    def get_candidates_by_formula(
        self,
        formula: object,
    ) -> tuple[tuple[int, Adduct], ...]:
        formula_index = self.get_formula_index(formula)

        return self.get_candidates_by_formula_index(formula_index)

    def copy(self) -> _FragmentIonFormulaCandidateGroup:
        return _FragmentIonFormulaCandidateGroup(
            adduct_type=self.adduct_type,
            fragment_ion_adducts=self.fragment_ion_adducts,
            formulas=self.formulas.copy(),
            indptr=self.indptr.copy(),
            node_indices=self.node_indices.copy(),
            candidate_adduct_indices=self.candidate_adduct_indices.copy(),
            fragment_ion_adduct_positions=self.fragment_ion_adduct_positions.copy(),
            hydrogen_shifts=self.hydrogen_shifts.copy(),
            candidate_adducts=tuple(self.candidate_adducts),
        )

    def _validate_formula_index(
        self,
        formula_index: int,
    ) -> None:
        if not isinstance(formula_index, int):
            raise TypeError("formula_index must be an int.")

        if not 0 <= formula_index < self.num_formulas:
            raise IndexError(
                "formula_index must be in "
                f"[0, {self.num_formulas}). "
                f"Got {formula_index}."
            )
