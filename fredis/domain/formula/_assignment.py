from __future__ import annotations

import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from fredis.libs.msentity.msentity import MSDataset
from fredis.libs.mmkit.mmkit import Adduct, Compound, Formula
from fredis.domain.mass.tolerance import MassTolerance

from .utils import assign_formulas_to_peaks


def max_neutral_precursor_formula(
    original_formula: Formula,
    precursor_formula: Formula,
) -> Formula:
    """Element-wise max of the original molecule and precursor formula."""

    original_elements = original_formula.plain.elements
    precursor_elements = precursor_formula.plain.elements
    elements = {
        element: max(
            int(original_elements.get(element, 0)),
            int(precursor_elements.get(element, 0)),
        )
        for element in set(original_elements) | set(precursor_elements)
    }
    return Formula(elements=elements, charge=0)


def formula_to_label(formula: Formula) -> str:
    """Return stable formula label."""

    if hasattr(formula, "plain_value"):
        return str(formula.plain_value)

    if hasattr(formula, "plain"):
        return str(formula.plain)

    if hasattr(formula, "value"):
        return str(formula.value)

    return str(formula)


def group_record_indexes_by_precursor_formula(
    dataset: MSDataset,
    *,
    smiles_column: str,
    adduct_type_column: str,
    precursor_mz_column: str,
    mass_tolerance: MassTolerance,
    error_column: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, List[int]]:
    """Group records by calculated precursor formula.

    Warning/error messages are not written to output columns here.
    If error_column is provided, it is used only for temporary in-memory marking.
    """

    groups: Dict[str, List[int]] = defaultdict(list)

    success_count = 0
    progress_count = 0

    pbar = (
        tqdm(
            total=len(dataset),
            desc="Grouping by precursor formula",
            mininterval=1.0,
        )
        if show_progress
        else None
    )

    for record_index, record in enumerate(dataset):
        try:
            smiles = str(record[smiles_column])
            adduct_type_str = str(record[adduct_type_column])
            precursor_mz = float(record[precursor_mz_column])

            compound = Compound.from_smiles(smiles)
            adduct_type = Adduct.parse(adduct_type_str)

            original_formula = compound.formula.plain
            precursor_formula = adduct_type.apply_to_formula(compound.formula)
            grouping_formula = max_neutral_precursor_formula(
                original_formula=original_formula,
                precursor_formula=precursor_formula,
            )
            calculated_precursor_mz = adduct_type.apply_to_mz(
                compound.formula.exact_mass
            )

            if mass_tolerance.within(precursor_mz, calculated_precursor_mz):
                groups[formula_to_label(grouping_formula)].append(record_index)
                success_count += 1
            else:
                message = (
                    "PrecursorMZMismatch: "
                    f"record_index={record_index}, "
                    f"observed={precursor_mz}, "
                    f"calculated={calculated_precursor_mz}, "
                    f"smiles={smiles}, "
                    f"adduct={adduct_type_str}"
                )
                print(f"[WARN] {message}", file=sys.stderr)

        except Exception as e:
            message = (
                "ErrorCalculatingPrecursorFormula: "
                f"record_index={record_index}, error={e}"
            )
            print(f"[WARN] {message}", file=sys.stderr)

        finally:
            progress_count += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "success": (
                            f"{success_count}/{progress_count}"
                            f"({success_count / progress_count * 100:.1f}%)"
                        )
                    }
                )

    if pbar is not None:
        pbar.close()

    return groups


def assign_record_formulas(
    *,
    record,
    formula_candidates: Sequence[Formula],
    mass_tolerance: MassTolerance,
    calc_formula_column: str,
    calc_formula_coverage_column: str,
    original_formula: Optional[Formula] = None,
) -> float:
    """Assign formulas to peaks in one spectrum record.

    This function writes:
        - peak-level matched formulas to calc_formula_column
        - record-level coverage to calc_formula_coverage_column

    Returns
    -------
    float
        Matched-intensity coverage.
    """

    peaks_mz = [float(peak.mz) for peak in record.peaks]
    peak_intensities = [float(peak.intensity) for peak in record.peaks]

    total_intensity = sum(peak_intensities)

    assignment_results = assign_formulas_to_peaks(
        peaks_mz=peaks_mz,
        formula_candidates=list(formula_candidates),
        mass_tolerance=mass_tolerance,
        original_formula=original_formula,
    )

    assigned_formula_texts = [
        ",".join(result["matched_formulas"]) for result in assignment_results
    ]

    matched_intensity = sum(
        peak_intensities[i]
        for i, result in enumerate(assignment_results)
        if len(result["matched_formulas"]) > 0
    )

    coverage = matched_intensity / total_intensity if total_intensity > 0 else 0.0

    record.peaks[calc_formula_column] = assigned_formula_texts
    record[calc_formula_coverage_column] = str(coverage)

    return coverage
