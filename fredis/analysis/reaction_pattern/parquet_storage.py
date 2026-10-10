"""B: one ``compounds/<compound_id>.peaks.parquet`` per compound.

One row is one peak of one spectrum evaluated by at least one root
condition.  The file holds only binary peak annotations, never fragment
structures or paths (those are in C).  ``assigned::<condition_id>`` and
``count::<condition_id>`` exist for every condition resolved for this
compound; ``compound_condition`` in A tells which samples each condition
evaluated.  An edge's newly assigned peaks are ``child and not parent``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .comparison import CompoundComparison
from .constants import assigned_column, count_column

IDENTITY_COLUMNS = {
    "compound_id": object, "smiles": object, "sample_id": "int64", "dataset_id": "int64",
    "record_index": "int64", "sample_index": "int64", "peak_index": "int64", "adduct": object,
    "collision_energy": "float64", "precursor_mz": "float64", "mz": "float64", "intensity": "float64",
    "relative_intensity": "float64", "is_precursor": bool,
}


def peak_frame(comparison: CompoundComparison) -> pd.DataFrame:
    group = comparison.group
    conditions = sorted(comparison.outcomes)
    parts = []
    for index in comparison.population:
        sample = group.samples[index]
        count = len(sample.mz)
        part = {
            "compound_id": np.full(count, group.compound_id, dtype=object),
            "smiles": np.full(count, group.canonical_smiles, dtype=object),
            "sample_id": np.full(count, sample.sample_id, dtype=np.int64),
            "dataset_id": np.full(count, sample.dataset_id, dtype=np.int64),
            "record_index": np.full(count, sample.record_index, dtype=np.int64),
            "sample_index": np.full(count, sample.sample_index, dtype=np.int64),
            "peak_index": np.arange(count, dtype=np.int64),
            "adduct": np.full(count, sample.adduct, dtype=object),
            "collision_energy": np.full(count, sample.collision_energy, dtype=np.float64),
            "precursor_mz": np.full(count, sample.precursor_mz, dtype=np.float64),
            "mz": sample.mz.astype(np.float64),
            "intensity": sample.intensity.astype(np.float64),
            "relative_intensity": sample.relative_intensity.astype(np.float64),
            "is_precursor": sample.is_precursor.astype(bool),
        }
        for condition_id in conditions:
            outcome = comparison.outcomes[condition_id]
            part[assigned_column(condition_id)] = np.asarray(outcome.assigned[index], dtype=bool)
            part[count_column(condition_id)] = np.asarray(outcome.counts[index], dtype=np.int32)
        parts.append(pd.DataFrame(part))
    if parts:
        return pd.concat(parts, ignore_index=True)
    frame = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in IDENTITY_COLUMNS.items()})
    for condition_id in conditions:
        frame[assigned_column(condition_id)] = pd.Series(dtype=bool)
        frame[count_column(condition_id)] = pd.Series(dtype="int32")
    return frame


def write_peaks(comparison: CompoundComparison, output: Path) -> Path:
    """Write atomically, so an interrupted extension keeps the previous file."""
    path = output / comparison.group.peak_summary_file
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    peak_frame(comparison).to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)
    return path


def read_condition_masks(path: Path, columns: dict[int, tuple[str, str | None]]) -> dict:
    """``{(sample_index, condition_id): (assigned, counts)}`` from a peak table.

    ``columns`` maps condition ids to their (assigned, count) column names in
    that file; a missing count column means one path per assigned peak.
    """
    frame = pd.read_parquet(path)
    result = {}
    if not len(frame):
        return result
    for sample_index, rows in frame.groupby("sample_index", sort=False):
        rows = rows.sort_values("peak_index")
        for condition_id, (assigned_name, count_name) in columns.items():
            if assigned_name not in rows.columns:
                continue
            assigned = rows[assigned_name].to_numpy(dtype=bool)
            counts = (rows[count_name].to_numpy(dtype=np.int32) if count_name and count_name in rows.columns
                      else assigned.astype(np.int32))
            result[(int(sample_index), int(condition_id))] = (assigned, counts)
    return result
