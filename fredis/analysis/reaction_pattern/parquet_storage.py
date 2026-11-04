"""B: one ``compounds/<compound_id>.peaks.parquet`` per compound.

One row is one peak of one spectrum evaluated by at least one root
condition.  The file holds only binary peak annotations, never fragment
structures or paths (those are in C).  ``assigned::<condition_id>`` and
``count::<condition_id>`` exist for every condition resolved for this
compound; ``compound_condition`` in A tells which samples each condition
evaluated.  ``patterns::<condition_id>`` lists (comma-joined) the ids of the
patterns in any history that explains the peak under that condition, and
``rules::<condition_id>`` the ``pattern:rule`` (product) pairs.  An edge's newly assigned peaks are ``child and not parent``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .comparison import CompoundComparison
from .constants import FORMULA_COLUMN, assigned_column, count_column, patterns_column, rules_column

IDENTITY_COLUMNS = {
    "compound_id": object,
    "smiles": object,
    "sample_id": "int64",
    "dataset_id": "int64",
    "record_index": "int64",
    "sample_index": "int64",
    "peak_index": "int64",
    "adduct": object,
    "collision_energy": "float64",
    "precursor_mz": "float64",
    "mz": "float64",
    "intensity": "float64",
    "relative_intensity": "float64",
    "is_precursor": bool,
}


def pattern_masks(per_peak, count: int) -> dict[int, np.ndarray]:
    """``pattern id -> peak mask`` from per-peak id lists (or comma-joined strings)."""
    masks: dict[int, np.ndarray] = {}
    for peak, ids in enumerate(per_peak):
        if isinstance(ids, str):
            ids = [int(value) for value in ids.split(",") if value != ""]
        for pattern_id in ids or ():
            mask = masks.get(int(pattern_id))
            if mask is None:
                mask = masks[int(pattern_id)] = np.zeros(count, dtype=bool)
            mask[peak] = True
    return masks


def rule_masks(per_peak, count: int) -> dict[tuple[int, int], np.ndarray]:
    """``(pattern id, rule id) -> peak mask`` from per-peak ``pattern:rule`` lists or strings."""
    masks: dict[tuple[int, int], np.ndarray] = {}
    for peak, ids in enumerate(per_peak):
        if isinstance(ids, str):
            ids = [value for value in ids.split(",") if value != ""]
        for value in ids or ():
            pattern, _, rule = str(value).partition(":")
            key = (int(pattern), int(rule))
            mask = masks.get(key)
            if mask is None:
                mask = masks[key] = np.zeros(count, dtype=bool)
            mask[peak] = True
    return masks


def pattern_strings(masks: dict, count: int) -> np.ndarray:
    ids = [[] for _ in range(count)]
    for key in sorted(masks):
        label = f"{key[0]}:{key[1]}" if isinstance(key, tuple) else str(key)
        for peak in np.flatnonzero(masks[key]):
            ids[int(peak)].append(label)
    return np.asarray([",".join(values) for values in ids], dtype=object)


def peak_frame(comparison: CompoundComparison) -> pd.DataFrame:
    group = comparison.group
    conditions = sorted(comparison.outcomes)
    formula_enabled = any(sample.formula_status for sample in group.samples)
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
            "collision_energy": np.full(
                count, sample.collision_energy, dtype=np.float64
            ),
            "precursor_mz": np.full(count, sample.precursor_mz, dtype=np.float64),
            "mz": sample.mz.astype(np.float64),
            "intensity": sample.intensity.astype(np.float64),
            "relative_intensity": sample.relative_intensity.astype(np.float64),
            "is_precursor": sample.is_precursor.astype(bool),
        }
        if formula_enabled:
            part["formula_status"] = np.full(count, sample.formula_status, dtype=object)
            part["precursor_formula"] = np.full(
                count, sample.precursor_formula, dtype=object
            )
            part[FORMULA_COLUMN] = np.asarray(
                sample.formulas if sample.formulas is not None else [""] * count,
                dtype=object,
            )
        for condition_id in conditions:
            outcome = comparison.outcomes[condition_id]
            part[assigned_column(condition_id)] = np.asarray(
                outcome.assigned[index], dtype=bool
            )
            part[count_column(condition_id)] = np.asarray(
                outcome.counts[index], dtype=np.int32
            )
            if outcome.patterns is not None:
                part[patterns_column(condition_id)] = pattern_strings(
                    outcome.patterns[index], count
                )
            if outcome.rules is not None:
                part[rules_column(condition_id)] = pattern_strings(
                    outcome.rules[index], count
                )
        parts.append(pd.DataFrame(part))
    if parts:
        return pd.concat(parts, ignore_index=True)
    frame = pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in IDENTITY_COLUMNS.items()}
    )
    if formula_enabled:
        for name in ("formula_status", "precursor_formula", FORMULA_COLUMN):
            frame[name] = pd.Series(dtype=object)
    for condition_id in conditions:
        frame[assigned_column(condition_id)] = pd.Series(dtype=bool)
        frame[count_column(condition_id)] = pd.Series(dtype="int32")
        if comparison.outcomes[condition_id].patterns is not None:
            frame[patterns_column(condition_id)] = pd.Series(dtype=object)
        if comparison.outcomes[condition_id].rules is not None:
            frame[rules_column(condition_id)] = pd.Series(dtype=object)
    return frame


def write_peaks(comparison: CompoundComparison, output: Path) -> Path:
    """Write atomically, so an interrupted extension keeps the previous file."""
    path = output / comparison.group.peak_summary_file
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    peak_frame(comparison).to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)
    return path


def read_condition_masks(path: Path, columns: dict[int, tuple]) -> dict:
    """``{(sample_index, condition_id): (assigned, counts, patterns, rules)}`` from a peak table.

    ``columns`` maps condition ids to their (assigned, count[, patterns[, rules]])
    column names in that file; a missing count column means one path per
    assigned peak, and a missing patterns/rules column gives ``None``.
    """
    frame = pd.read_parquet(path)
    result = {}
    if not len(frame):
        return result
    for sample_index, rows in frame.groupby("sample_index", sort=False):
        rows = rows.sort_values("peak_index")
        for condition_id, names in columns.items():
            assigned_name, count_name = names[0], names[1]
            patterns_name = names[2] if len(names) > 2 else None
            rules_name = names[3] if len(names) > 3 else None
            if assigned_name not in rows.columns:
                continue
            assigned = rows[assigned_name].to_numpy(dtype=bool)
            counts = (
                rows[count_name].to_numpy(dtype=np.int32)
                if count_name and count_name in rows.columns
                else assigned.astype(np.int32)
            )
            patterns = (
                pattern_masks(rows[patterns_name].fillna("").tolist(), len(rows))
                if patterns_name and patterns_name in rows.columns
                else None
            )
            rules = (
                rule_masks(rows[rules_name].fillna("").tolist(), len(rows))
                if rules_name and rules_name in rows.columns
                else None
            )
            result[(int(sample_index), int(condition_id))] = (
                assigned,
                counts,
                patterns,
                rules,
            )
    return result
