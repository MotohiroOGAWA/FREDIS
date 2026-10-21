from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from fredis.libs.msentity.msentity import MSDataset, load_ms_dataset


def default_output_file(input_file: str) -> str:
    """Return default output path."""

    input_path = Path(input_file)
    return str(
        input_path.with_name(input_path.stem + "_assigned_formula" + input_path.suffix)
    )


def save_dataset(
    dataset: MSDataset,
    output_file: str,
    *,
    add_finished_tag: bool = False,
) -> None:
    """Save MSDataset to file."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if add_finished_tag and hasattr(dataset, "add_tag"):
        dataset.add_tag("finished")

    if hasattr(dataset, "save"):
        dataset.save(str(output_path))
        return

    if hasattr(dataset, "to_hdf5"):
        dataset.to_hdf5(str(output_path), mode="w")
        return

    raise AttributeError("MSDataset must provide either save() or to_hdf5().")


def prepare_dataset_io(
    *,
    input_file: str,
    output_file: Optional[str],
    overwrite: bool,
) -> Tuple[MSDataset, str]:
    """Load dataset and determine output file."""

    dataset = load_ms_dataset(input_file)

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

    if not hasattr(MSDataset, "read_dataset_meta"):
        return False

    try:
        meta = MSDataset.read_dataset_meta(path)
        tags = getattr(meta, "tags", [])
        return "finished" in tags
    except Exception:
        return False


def validate_columns(
    dataset: MSDataset,
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
    dataset: MSDataset,
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
        dataset.peaks[calc_formula_column] = ""
        return calc_formula_column, calc_formula_coverage_column

    if calc_formula_coverage_column not in dataset.columns:
        dataset[calc_formula_coverage_column] = ""

    peak_meta_columns = []
    if getattr(dataset.peaks, "_metadata_ref", None) is not None:
        peak_meta_columns = list(dataset.peaks._metadata_ref.columns)

    if calc_formula_column not in peak_meta_columns:
        dataset.peaks[calc_formula_column] = ""

    return calc_formula_column, calc_formula_coverage_column


def try_move_peak_column_to_front(
    dataset: MSDataset,
    column: str,
) -> None:
    """Move one peak metadata column to the front if possible."""

    try:
        peak_meta = dataset.peaks._metadata_ref
        if peak_meta is None:
            return

        columns = [column] + [c for c in peak_meta.columns if c != column]
        dataset.peaks.meta_columns = columns
    except Exception:
        return
