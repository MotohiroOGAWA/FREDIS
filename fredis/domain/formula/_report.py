from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class AssignFormulaReportRow:
    """One skipped/warning/error record during formula assignment."""

    index: int
    spec_id: str
    status: str
    message: str

    formula: str = ""
    smiles: str = ""
    adduct_type: str = ""
    precursor_mz: str = ""

    estimated_candidates: str = ""
    max_candidates: str = ""
    generated_candidates: str = ""


def default_report_file(output_file: str) -> str:
    """Return default TSV report path."""

    output_path = Path(output_file)
    return str(
        output_path.with_name(
            output_path.stem + "_assign_formula_report.tsv"
        )
    )


def get_record_spec_id(record, *, default: str = "") -> str:
    """Get SpecID from record if available."""

    for key in ("SpecID", "spec_id", "SpectrumID", "spectrum_id", "ID", "id"):
        try:
            value = record[key]
            if value is not None and str(value) != "":
                return str(value)
        except Exception:
            pass

    return default


def write_report_tsv(
    rows: Iterable[AssignFormulaReportRow],
    report_file: Optional[str],
) -> None:
    """Write report rows as TSV."""

    if report_file is None:
        return

    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [field.name for field in fields(AssignFormulaReportRow)]

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field.name: getattr(row, field.name)
                    for field in fields(AssignFormulaReportRow)
                }
            )