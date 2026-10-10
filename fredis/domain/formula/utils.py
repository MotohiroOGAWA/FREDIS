from __future__ import annotations

import time
from itertools import product
from dataclasses import dataclass
from typing import Any, Generator, Dict, List, Tuple, Sequence, Union

from ...libs.mmkit.mmkit import Formula
from ..mass import MassTolerance


@dataclass(frozen=True)
class FormulaAssignmentCandidate:
    formula: Formula
    label: str


def _validate_formula_element_signs(
    formula: Formula,
    *,
    allow_zero: bool,
    sign: str,
    label: str,
) -> None:
    counts = list(formula.elements.values())
    if sign == "positive":
        valid = all(count >= 0 if allow_zero else count > 0 for count in counts)
        message = "ordinary formula terms must be positive"
    elif sign == "negative":
        valid = all(count <= 0 if allow_zero else count < 0 for count in counts)
        message = "neutral-loss terms must be negative"
    else:
        raise ValueError(f"Unsupported sign validation: {sign}")

    if not valid:
        raise ValueError(f"Invalid formula label {label!r}: {message}.")


def parse_neutral_loss_formula(loss_text: str) -> Formula:
    text = str(loss_text).strip()
    if not text:
        raise ValueError("Neutral-loss formula must not be empty.")

    if text.startswith("-") and len(text) > 1 and text[1].isalpha():
        positive_loss = Formula.parse(text[1:])
        return positive_loss * -1

    return Formula.parse(text)


def format_neutral_loss_formula(loss_formula: Formula) -> str:
    elements = loss_formula.elements
    if elements and all(count <= 0 for count in elements.values()):
        positive_elements = {element: -int(count) for element, count in elements.items()}
        return "-" + str(Formula(elements=positive_elements, charge=0))

    return str(loss_formula)


def parse_formula_assignment_label(label: str) -> FormulaAssignmentCandidate:
    """Parse ordinary or neutral-loss assignment label.

    Examples
    --------
    C6H6 -> formula C6H6, label C6H6
    C5H3(-CH3) -> formula C5H3, label C5H3(-CH3)
    """

    text = str(label).strip()
    if not text:
        raise ValueError("Formula label must not be empty.")

    if "(" not in text and ")" not in text:
        formula = Formula.parse(text)
        _validate_formula_element_signs(
            formula,
            allow_zero=True,
            sign="positive",
            label=text,
        )
        return FormulaAssignmentCandidate(formula=formula, label=str(formula))

    if not text.endswith(")") or text.count("(") != 1 or text.count(")") != 1:
        raise ValueError(
            "Neutral-loss formula labels must look like C5H3(-CH3). "
            f"Got {text!r}."
        )

    formula_text, loss_text = text[:-1].split("(", maxsplit=1)
    formula = Formula.parse(formula_text)
    loss_formula = parse_neutral_loss_formula(loss_text)
    _validate_formula_element_signs(
        formula,
        allow_zero=True,
        sign="positive",
        label=text,
    )
    _validate_formula_element_signs(
        loss_formula,
        allow_zero=True,
        sign="negative",
        label=text,
    )
    return FormulaAssignmentCandidate(
        formula=formula,
        label=f"{formula}({format_neutral_loss_formula(loss_formula)})",
    )


def _neutral_loss_label_for_formula(
    formula: Formula,
    original_formula: Formula | None,
) -> str:
    if original_formula is None:
        return str(formula)

    loss_formula = formula - original_formula.plain
    loss_counts = list(loss_formula.elements.values())

    if loss_counts and all(count <= 0 for count in loss_counts) and any(
        count < 0 for count in loss_counts
    ):
        return f"{formula}({format_neutral_loss_formula(loss_formula)})"

    return str(formula)


def make_formula_assignment_candidate(
    candidate: Union[Formula, str, FormulaAssignmentCandidate],
    *,
    original_formula: Formula | None = None,
) -> FormulaAssignmentCandidate:
    if isinstance(candidate, FormulaAssignmentCandidate):
        return candidate

    if isinstance(candidate, str):
        return parse_formula_assignment_label(candidate)

    if isinstance(candidate, Formula):
        _validate_formula_element_signs(
            candidate,
            allow_zero=True,
            sign="positive",
            label=str(candidate),
        )
        return FormulaAssignmentCandidate(
            formula=candidate,
            label=_neutral_loss_label_for_formula(candidate, original_formula),
        )

    raise TypeError(f"Unsupported formula candidate type: {type(candidate).__name__}")


def calculate_dbe(elements: dict[str, int]) -> float:
    """
    Calculate the double bond equivalent (DBE) from an element count dictionary.
    Only common organic elements are supported (C, H, N, O, P, S, F, Cl, Br, I).

    Args:
        elements (dict): A dictionary of element counts.

    Returns:
        float: The calculated DBE.

    Raises:
        AssertionError: If unsupported elements are present.
    """
    allowed_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Na'}
    unsupported = set(elements.keys()) - allowed_elements
    assert not unsupported, f"Unsupported elements in formula: {unsupported}"

    C = elements.get('C', 0)
    H = elements.get('H', 0)
    O = elements.get('O', 0)
    N = elements.get('N', 0)
    P = elements.get('P', 0)
    S = elements.get('S', 0)
    X = sum([elements.get(x, 0) for x in ['F', 'Cl', 'Br', 'I']])  # Halogens
    
    dbe = (2*C + N - H - X + 2) / 2.0
    return dbe

def enumerate_possible_sub_formulas(elements: dict[str, int], timeout:float=float('inf')) -> Generator[tuple[dict[str, int], float], None, None]:
    base_elements = [(elem, count) for elem, count in elements.items() if elem != "H"]
    max_h = elements.get("H", 0)
    
    # Create all combinations from 0 to the original count for each element (excluding H)
    ranges = [range(c + 1) for _, c in base_elements]

    start_time = time.time()
    for counts in product(*ranges):
        if (time.time() - start_time) > timeout:
            raise TimeoutError("Timeout exceeded during formula enumeration.")
        # Generate element count combinations
        temp_counts = {elem: count for (elem, _), count in zip(base_elements, counts)}
        
        # Try hydrogen counts from 0 to max_h
        for h in range(max_h + 1):
            dbe = calculate_dbe(temp_counts | {"H": h})
            if dbe < 0:
                continue  # Skip if degree of unsaturation is negative

            temp_counts["H"] = h
            if sum(temp_counts.values()) == 0:
                continue
            
            yield temp_counts.copy(), dbe

def get_possible_sub_formulas(formula: Formula, hydrogen_delta: int = 0, timeout: float=float('inf')) -> Dict[str, float]:
    """
    Generate possible sub-formulas with their degree of unsaturation.
    Returns a dictionary of sub-formula strings and their DBE values.
    """
    # Add hydrogen shift to original count
    elements = formula.elements.copy()
    elements["H"] = elements.get("H", 0) + hydrogen_delta
    elements["H"] = max(elements["H"], 0)  # prevent negative H count

    sub_formulas = [
        Formula(candidate_elements, charge=0)
        for candidate_elements, dbe in enumerate_possible_sub_formulas(elements, timeout=timeout)
    ]

    sub_formulas = sorted(sub_formulas , key=lambda f: (f.exact_mass, str(f.plain_value)))

    return sub_formulas
    
def assign_formulas_to_peaks(
    peaks_mz: List[float],
    formula_candidates: Sequence[Union[Formula, str, FormulaAssignmentCandidate]],
    mass_tolerance: MassTolerance,
    *,
    original_formula: Formula | None = None,
) -> List[Dict[str, Any]]:
    """Assign candidate formulas to peaks using mass tolerance."""

    sorted_peaks = sorted(
        ((float(mz), index) for index, mz in enumerate(peaks_mz)),
        key=lambda item: item[0],
    )

    assignment_candidates = [
        make_formula_assignment_candidate(
            formula_candidate,
            original_formula=original_formula,
        )
        for formula_candidate in formula_candidates
    ]

    sorted_formulas = sorted(
        (
            (
                formula_index,
                formula_candidate.label,
                float(formula_candidate.formula.exact_mass),
            )
            for formula_index, formula_candidate in enumerate(assignment_candidates)
        ),
        key=lambda item: item[2],
    )

    results: list[dict[str, Any]] = [{} for _ in peaks_mz]

    left = 0
    n_formulas = len(sorted_formulas)

    for mz, original_index in sorted_peaks:
        lower, upper = mass_tolerance.to_da_range(mz)

        while left < n_formulas and sorted_formulas[left][2] < lower:
            left += 1

        matches: list[tuple[int, str, float]] = []

        cursor = left
        while cursor < n_formulas and sorted_formulas[cursor][2] <= upper:
            formula_index, formula_name, exact_mass = sorted_formulas[cursor]

            if mass_tolerance.within(mz, exact_mass):
                matches.append(
                    (
                        formula_index,
                        formula_name,
                        mass_tolerance.error(mz, exact_mass),
                    )
                )

            cursor += 1

        matches.sort(key=lambda item: abs(item[2]))

        results[original_index] = {
            "mz": mz,
            "n_matches": len(matches),
            "matched_formulas": [
                formula_name
                for _formula_index, formula_name, _error in matches
            ],
            "matched_formula_indices": [
                formula_index
                for formula_index, _formula_name, _error in matches
            ],
            "mass_errors": [
                error
                for _formula_index, _formula_name, error in matches
            ],
        }

    return results

def get_isotopic_masses(formula: Formula) -> List[Tuple[float, int, int]]:
    """
    Calculate all possible exact masses for a given molecular formula
    considering isotopic variations of chlorine (Cl) and bromine (Br).

    For each isotopic combination, this function returns:
        - The resulting isotopic mass (Da)
        - The number of 37Cl atoms (heavy chlorine)
        - The number of 81Br atoms (heavy bromine)

    Assumptions:
        - 37Cl - 35Cl = +1.99705 Da
        - 81Br - 79Br = +1.99795 Da

    Args:
        formula (Formula): The molecular formula.

    Returns:
        List[Tuple[float, int, int]]:
            A list of tuples:
                (mass, n_heavy_Cl, n_heavy_Br)
            Each entry represents one isotopic combination.
            If no Cl or Br is present, returns [(exact_mass, 0, 0)].

    Example:
        >>> f = Formula.parse("C6H5Cl")
        >>> get_isotopic_masses(f)
        [(112.00085, 0, 0), (113.9979, 1, 0)]
    """
    # --- Base exact mass ---
    base_mass = formula.exact_mass

    # --- Count chlorine and bromine atoms ---
    n_cl = formula._elements.get("Cl", 0)
    n_br = formula._elements.get("Br", 0)

    # --- Isotopic mass differences (Da) ---
    delta_cl = 1.99705   # 37Cl - 35Cl
    delta_br = 1.99795   # 81Br - 79Br

    # --- No isotopic elements ---
    if n_cl == 0 and n_br == 0:
        return [(base_mass, 0, 0)]

    # --- Enumerate all isotopic combinations ---
    cl_indices = range(n_cl + 1) if n_cl > 0 else [0]
    br_indices = range(n_br + 1) if n_br > 0 else [0]

    results = []
    for n_heavy_cl, n_heavy_br in product(cl_indices, br_indices):
        delta_mass = n_heavy_cl * delta_cl + n_heavy_br * delta_br
        mass = round(base_mass + delta_mass, 6)
        results.append((mass, n_heavy_cl, n_heavy_br))

    # --- Remove duplicates and sort by mass ---
    results = sorted(set(results), key=lambda x: x[0])
    return results