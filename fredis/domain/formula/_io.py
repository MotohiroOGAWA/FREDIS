from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import msentity as ms


def default_output_file(input_file: str) -> str:
    """Return default output path."""

    input_path = Path(input_file)
    return str(
        input_path.with_name(input_path.stem + "_assigned_formula" + input_path.suffix)
    )


def save_dataset(
    dataset: ms.Dataset,
    output_file: str,
    *,
    add_finished_tag: bool = False,
) -> None:
    """Save MSDataset to file."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if add_finished_tag:
        dataset.add_tag('finished')
    dataset.save(str(output_path))


def prepare_dataset_io(
    *,
    input_file: str,
    output_file: Optional[str],
    overwrite: bool,
) -> Tuple[ms.Dataset, str]:
    """Load dataset and determine output file."""

    dataset = ms.load(input_file)

    if output_file is None:
        output_file = default_output_file(input_file)

    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            "Use --overwrite to overwrite it."
        )

    return dataset, output_file


def is_finished_dataset(path: str) -> bool:
    """Return whether a dataset file has a finished tag."""

    if not os.path.exists(path):
        return False

    try:
        with ms.load(path) as dataset:
            return 'finished' in dataset.tags
    except (OSError, ms.MsEntityError):
        return False


def validate_columns(
    dataset: ms.Dataset,
    *,
    smiles_column: str,
    adduct_type_column: str,
    precursor_mz_column: str,
) -> None:
    """Validate required metadata columns."""

    required_columns = [
        smiles_column,
        adduct_type_column,
        precursor_mz_column,
    ]

    for column in required_columns:
        if column not in dataset.columns:
            raise ValueError(f"Column '{column}' was not found in dataset.")


def infer_calc_formula_column(
    calc_formula_coverage_column: str,
) -> str:
    """Infer peak-level formula column name from coverage column name.

    Examples
    --------
    CalcFormulaCov -> CalcFormula
    """

    if calc_formula_coverage_column.endswith("Cov"):
        return calc_formula_coverage_column[:-3]

    return calc_formula_coverage_column + "Formula"


def initialize_output_columns(
    dataset: ms.Dataset,
    *,
    calc_formula_coverage_column: str,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """Initialize output columns.

    Returns
    -------
    Tuple[str, str]
        calc_formula_column, calc_formula_coverage_column
    """

    calc_formula_coverage_column = calc_formula_coverage_column.strip()
    calc_formula_column = infer_calc_formula_column(calc_formula_coverage_column)

    if overwrite:
        dataset[calc_formula_coverage_column] = ""
        dataset.set_peak_column(calc_formula_column, [""] * dataset.n_peaks_total)
        return calc_formula_column, calc_formula_coverage_column

    if calc_formula_coverage_column not in dataset.columns:
        dataset[calc_formula_coverage_column] = ""

    if calc_formula_column not in dataset.peak_columns:
        dataset.set_peak_column(calc_formula_column, [''] * dataset.n_peaks_total)

    return calc_formula_column, calc_formula_coverage_column


def try_move_peak_column_to_front(
    dataset: ms.Dataset,
    column: str,
) -> None:
    """Move one peak metadata column to the front if possible."""

    # Native peak columns retain their insertion order. The formula column is
    # inserted before annotation work begins; no private pandas state is used.
    return
