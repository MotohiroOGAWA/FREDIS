from __future__ import annotations

import csv
import os
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def replace_arg(
    cmd_args: List[str],
    flag_name: str,
    new_value: str,
) -> None:
    """Replace value after a CLI flag if the flag exists."""

    if flag_name in cmd_args:
        idx = cmd_args.index(flag_name)
        if idx + 1 < len(cmd_args):
            cmd_args[idx + 1] = new_value


def delete_arg(
    cmd_args: List[str],
    flag_name: str,
) -> None:
    """Delete a CLI flag and its value if present."""

    if flag_name not in cmd_args:
        return

    idx = cmd_args.index(flag_name)
    cmd_args.pop(idx)

    if idx < len(cmd_args) and not cmd_args[idx].startswith("-"):
        cmd_args.pop(idx)


def try_add_arg(
    cmd_args: List[str],
    flag_name: str,
    value: Optional[str] = None,
) -> None:
    """Add CLI flag if it does not already exist."""

    if flag_name in cmd_args:
        return

    cmd_args.append(flag_name)

    if value is not None:
        cmd_args.append(value)


def make_chunks_by_precursor_formula(
    precursor_formula_groups: Dict[str, List[int]],
    *,
    chunk_size: int,
) -> List[List[int]]:
    """Make chunks by precursor-formula groups."""

    iterator = iter(precursor_formula_groups.items())
    chunks: List[List[int]] = []

    while True:
        part = list(islice(iterator, chunk_size))

        if not part:
            break

        chunk_indexes = [
            record_index
            for _, record_indexes in part
            for record_index in record_indexes
        ]
        chunks.append(chunk_indexes)

    return chunks


def merge_report_tsvs(
    input_files: Sequence[str],
    output_file: Optional[str],
) -> None:
    """Merge subprocess report TSV files.

    Missing files are ignored because some chunks may have no errors.
    If no report file exists, an empty output report is not created.
    """

    if output_file is None:
        return

    existing_files = [
        input_file
        for input_file in input_files
        if os.path.exists(input_file)
    ]

    if len(existing_files) == 0:
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: Optional[List[str]] = None
    rows: List[dict[str, str]] = []

    for input_file in existing_files:
        with open(input_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            if reader.fieldnames is None:
                continue

            if fieldnames is None:
                fieldnames = list(reader.fieldnames)

            for row in reader:
                rows.append(dict(row))

    if fieldnames is None:
        return

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)