"""A: ``result.reactionpatternanalysis.pft`` as a plain HDF5 file (read with ``h5py``).

Each table is an HDF5 group holding one 1-D dataset per column (numeric,
bool, or UTF-8 variable-length string).  Columnar storage lets readers load
only the columns and row ranges they need.  Large tables are chunked and
gzip-compressed.  ``/indexes`` stores row ranges and row-index arrays so a
single compound or pattern never needs a full-table Python scan.

Layout::

    /                         root attrs: schema, schema_version, result_format, analysis_status, ...
    /metadata                 same attrs + overall_summary (UTF-8 JSON)
    /configuration/json       run configuration (UTF-8 JSON)
    /datasets /patterns /compounds /compound_pattern_summary
    /compound_pattern_distribution /pattern_summary /pattern_distribution
    /sample_summary /sample_pattern_summary /detail_files
    /indexes/<name>/{start,count[,key,rows]}
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .constants import RESULT_FORMAT, SCHEMA, SCHEMA_VERSION

S, I, F, B = "str", "int", "float", "bool"
_DISTRIBUTION = [("metric", S), ("without_precursor", I), ("n", I), ("mean", F), ("std", F), ("min", F),
                 ("q25", F), ("median", F), ("q75", F), ("max", F), ("histogram_json", S)]
_SUFFIXES = ("", "_without_precursor")
_MEANS = ("peak_coverage", "intensity_coverage", "parent_peak_coverage", "parent_intensity_coverage",
          "delta_peak_coverage", "delta_intensity_coverage")


def _both(columns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(name + suffix, kind) for name, kind in columns for suffix in _SUFFIXES]


TABLE_SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "datasets": [("dataset_id", I), ("label", S), ("original_path", S), ("source_path", S), ("source_kind", S),
                 ("columns_json", S), ("record_count", I), ("valid_record_count", I), ("skipped_record_count", I),
                 ("compound_count", I), ("sample_count", I), ("peak_count", I)],
    "pattern_sets": [("pattern_set_id", S), ("name", S), ("pattern_count", I), ("pattern_ids", S),
                     ("node_count", I), ("source", S)],
    "patterns": [("pattern_id", I), ("name", S), ("reactant_smarts", S), ("definition_json", S),
                 ("pattern_set_ids", S)],
    "conditions": [("condition_id", I), ("condition_key", S), ("pattern_ids", S), ("label", S),
                   ("detail_directory", S)],
    "nodes": [("node_row", I), ("node_id", S), ("parent_node_row", I), ("parent_node_id", S),
              ("pattern_set_id", S), ("pattern_set_name", S), ("depth", I), ("is_root", B), ("condition_id", I),
              ("parent_condition_id", I), ("set_pattern_ids", S), ("added_pattern_ids", S), ("path_label", S),
              ("condition_label", S)],
    "compounds": [("compound_id", S), ("canonical_smiles", S), ("original_smiles", S), ("dataset_count", I),
                  ("dataset_ids", S), ("sample_count", I), ("peak_count", I), ("total_intensity", F),
                  ("peak_count_without_precursor", I), ("total_intensity_without_precursor", F),
                  ("evaluated_node_count", I), ("applicable_node_rows", S), ("improving_edge_count", I),
                  ("new_peak_union_count", I), ("new_intensity_union", F), ("peak_summary_file", S),
                  ("status", S), ("reason", S)],
    "compound_node_summary": [
        ("compound_id", S), ("compound_row", I), ("node_row", I), ("is_root", B), ("applicable", B),
        ("status", S), ("skip_category", S), ("reason", S), ("error", S), ("sample_count", I),
        ("improved_sample_count", I),
        *_both([("peak_count", I), ("total_intensity", F), ("assigned_peak_count_total", I),
                ("assigned_intensity_total", F), ("weighted_peak_coverage", F), ("weighted_intensity_coverage", F),
                ("parent_assigned_peak_count_total", I), ("parent_assigned_intensity_total", F),
                ("newly_assigned_peak_count_total", I), ("newly_assigned_intensity_total", F),
                ("parent_weighted_peak_coverage", F), ("parent_weighted_intensity_coverage", F),
                ("weighted_delta_peak_coverage", F), ("weighted_delta_intensity_coverage", F)]),
        *_both([("mean_" + name, F) for name in _MEANS])],
    "compound_node_distribution": [("compound_id", S), ("compound_row", I), ("node_row", I), *_DISTRIBUTION],
    "node_summary": [
        ("node_row", I), ("node_id", S), ("is_root", B), ("applicable_compound_count", I),
        ("calculated_compound_count", I), ("skipped_compound_count", I), ("failed_compound_count", I),
        ("evaluated_compound_count", I), ("evaluated_sample_count", I), ("improved_compound_count", I),
        ("improved_sample_count", I), ("overall_compound_count", I), ("overall_sample_count", I),
        *_both([("weighted_peak_coverage", F), ("weighted_intensity_coverage", F),
                ("parent_weighted_peak_coverage", F), ("parent_weighted_intensity_coverage", F),
                ("weighted_delta_peak_coverage", F), ("weighted_delta_intensity_coverage", F),
                ("newly_assigned_peak_count", I), ("newly_assigned_intensity", F),
                ("overall_peak_coverage", F), ("overall_intensity_coverage", F)]),
        *_both([("mean_" + name, F) for name in _MEANS])],
    "node_distribution": [("node_row", I), *_DISTRIBUTION],
    "sample_summary": [
        ("sample_id", I), ("compound_id", S), ("compound_row", I), ("dataset_id", I), ("record_index", I),
        ("sample_index", I), ("adduct", S), ("collision_energy", F), ("precursor_mz", F), ("peak_count", I),
        ("total_intensity", F), ("peak_count_without_precursor", I), ("total_intensity_without_precursor", F)],
    # Parent/newly/delta values are NaN for root nodes.
    "sample_node_summary": [
        ("sample_id", I), ("compound_row", I), ("node_row", I),
        *_both([(name, F) for name in (
            "assigned_peak_count", "assigned_intensity", "peak_coverage", "intensity_coverage",
            "parent_assigned_peak_count", "parent_assigned_intensity", "parent_peak_coverage",
            "parent_intensity_coverage", "newly_assigned_peak_count", "newly_assigned_intensity",
            "delta_peak_coverage", "delta_intensity_coverage")])],
    # One row per (compound, condition) resolved for that compound.  ``kept``
    # lists sample_index values; ``alias_of`` marks a condition identical to
    # another one for this compound; ``seed_path`` seeds child conditions.
    "compound_condition": [("compound_id", S), ("compound_row", I), ("condition_id", I), ("status", S),
                           ("skip_category", S), ("reason", S), ("error", S), ("kept", S), ("detail_path", S),
                           ("seed_path", S), ("alias_of", I)],
}
LARGE_TABLES = frozenset({"compounds", "compound_node_summary", "compound_node_distribution", "sample_summary",
                          "sample_node_summary", "compound_condition"})
# Tables sorted by compound_row, with a per-compound row range index.
COMPOUND_RANGE_INDEXES = {
    "sample_summary": "sample_by_compound",
    "sample_node_summary": "sample_node_by_compound",
    "compound_node_summary": "compound_node_by_compound",
    "compound_node_distribution": "compound_node_distribution_by_compound",
    "compound_condition": "compound_condition_by_compound",
}
NODE_INDEXES = {
    "compound_node_summary": "compound_node_by_node",
    "sample_node_summary": "sample_node_by_node",
    "node_distribution": "node_distribution_by_node",
}
_STRING = None


def string_dtype():
    import h5py

    global _STRING
    if _STRING is None:
        _STRING = h5py.string_dtype(encoding="utf-8")
    return _STRING


def _array(values: list, kind: str) -> np.ndarray:
    if kind == S:
        return np.asarray(["" if value is None else str(value) for value in values], dtype=object)
    if kind == I:
        return np.asarray(values, dtype=np.int64)
    if kind == B:
        return np.asarray(values, dtype=bool)
    return np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)


def _dataset(group, name: str, array: np.ndarray, *, compress: bool, dtype=None):
    options = {}
    if compress and len(array):
        options = dict(chunks=(min(len(array), 65536),), compression="gzip", compression_opts=4, shuffle=True,
                       maxshape=(None,))
    return group.create_dataset(name, data=array, dtype=dtype, **options)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _fredis_version() -> str:
    try:
        from importlib.metadata import version

        return version("fredis")
    except Exception:
        return ""


class ResultWriter:
    """Write A to ``<final>.tmp`` and atomically rename it when complete."""

    def __init__(self, final_path: str | Path):
        import h5py

        self.final_path = Path(final_path)
        self.temporary_path = self.final_path.with_name(self.final_path.name + ".tmp")
        self.temporary_path.unlink(missing_ok=True)
        self.file = h5py.File(self.temporary_path, "w")
        self.metadata = {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "result_format": RESULT_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(), "analysis_status": "running",
            "fredis_version": _fredis_version(), "git_commit_sha": _git_commit(),
        }
        self._set_status("running")
        self.row_counts: dict[str, int] = {}

    def _set_status(self, status: str) -> None:
        self.metadata["analysis_status"] = status
        group = self.file.require_group("metadata")
        for target in (self.file, group):
            for key, value in self.metadata.items():
                target.attrs[key] = value
        self.file.flush()

    def write_json(self, group_name: str, value) -> None:
        group = self.file.require_group(group_name)
        if "json" in group:
            del group["json"]
        group.create_dataset("json", data=json.dumps(value, ensure_ascii=False, default=str), dtype=string_dtype())

    def write_overall_summary(self, value: dict) -> None:
        group = self.file.require_group("metadata")
        if "overall_summary" in group:
            del group["overall_summary"]
        group.create_dataset("overall_summary", data=json.dumps(value, ensure_ascii=False), dtype=string_dtype())

    def write_table(self, name: str, columns: dict[str, list]) -> None:
        schema = TABLE_SCHEMAS[name]
        lengths = {len(values) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"Columns of {name} have different lengths.")
        count = lengths.pop() if lengths else 0
        unknown = set(columns) - {column for column, _ in schema}
        if unknown:
            raise ValueError(f"Unknown {name} columns: {', '.join(sorted(unknown))}")
        group = self.file.create_group(name)
        compress = name in LARGE_TABLES
        for column, kind in schema:
            values = columns.get(column)
            if values is None:
                if count:
                    raise ValueError(f"Missing column {name}.{column}")
                values = []
            _dataset(group, column, _array(values, kind), compress=compress,
                     dtype=string_dtype() if kind == S else None)
        group.attrs["columns"] = [column for column, _ in schema]
        group.attrs["row_count"] = count
        self.row_counts[name] = count

    def write_indexes(self, compound_count: int, tables: dict[str, dict[str, list]]) -> None:
        indexes = self.file.require_group("indexes")
        for table, index_name in COMPOUND_RANGE_INDEXES.items():
            rows = np.asarray(tables.get(table, {}).get("compound_row", []), dtype=np.int64)
            if rows.size and np.any(np.diff(rows) < 0):
                raise ValueError(f"{table} must be sorted by compound_row.")
            counts = np.bincount(rows, minlength=compound_count) if rows.size else np.zeros(compound_count, np.int64)
            group = indexes.create_group(index_name)
            group.attrs.update(table=table, key="compound_row")
            _dataset(group, "start", (np.cumsum(counts) - counts).astype(np.int64), compress=True)
            _dataset(group, "count", counts.astype(np.int64), compress=True)
        for table, index_name in NODE_INDEXES.items():
            self._key_index(indexes, index_name, table,
                            np.asarray(tables.get(table, {}).get("node_row", []), dtype=np.int64), "node_row")
        compound_rows, dataset_keys = [], []
        for row, value in enumerate(tables.get("compounds", {}).get("dataset_ids", [])):
            for item in str(value).split(","):
                if item:
                    compound_rows.append(row)
                    dataset_keys.append(int(item))
        self._key_index(indexes, "compound_by_dataset", "compounds", np.asarray(dataset_keys, dtype=np.int64),
                        "dataset_id", np.asarray(compound_rows, dtype=np.int64))

    @staticmethod
    def _key_index(indexes, name: str, table: str, keys: np.ndarray, key_name: str,
                   values: np.ndarray | None = None) -> None:
        """Sorted unique keys with [start, start+count) slices of ``rows``."""
        order = np.argsort(keys, kind="stable")
        rows = (values if values is not None else np.arange(len(keys), dtype=np.int64))[order]
        unique, start, count = np.unique(keys[order], return_index=True, return_counts=True)
        group = indexes.create_group(name)
        group.attrs.update(table=table, key=key_name)
        for column, data in (("key", unique), ("start", start), ("count", count), ("rows", rows)):
            _dataset(group, column, np.asarray(data, dtype=np.int64), compress=True)

    def finalize(self) -> Path:
        self._set_status("completed")
        self.file.close()
        os.replace(self.temporary_path, self.final_path)
        return self.final_path

    def abort(self) -> None:
        try:
            self._set_status("failed")
        finally:
            self.file.close()


def is_hdf5_result(path: str | Path) -> bool:
    """True for an HDF5 file whose root ``schema`` is a Reaction Pattern Analysis result.

    Other HDF5 files (for example ``.msds`` spectrum datasets) are rejected.
    """
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89HDF\r\n\x1a\n":
            return False
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            return handle.attrs.get("schema") == SCHEMA
    except OSError:
        return False


class Table:
    """Column/row-range reader for one table group."""

    def __init__(self, group):
        self.group = group
        self.columns = [str(value) for value in group.attrs["columns"]]
        self.row_count = int(group.attrs["row_count"])

    def __len__(self) -> int:
        return self.row_count

    def column(self, name: str, rows=None) -> np.ndarray:
        dataset = self.group[name]
        is_string = dataset.dtype.kind == "O"
        source = dataset.asstr() if is_string else dataset
        if rows is None:
            values = source[()]
        elif isinstance(rows, slice):
            values = source[rows]
        else:
            rows = np.asarray(rows, dtype=np.int64)
            if not rows.size:
                return np.asarray([], dtype=object if is_string else dataset.dtype)
            if rows.size > 4096:
                values = source[()][rows]
            else:
                order = np.argsort(rows)
                unique, inverse = np.unique(rows[order], return_inverse=True)
                picked = source[unique.tolist()]
                result = np.empty(rows.size, dtype=picked.dtype)
                result[order] = picked[inverse]
                values = result
        return np.asarray(values, dtype=object) if is_string else np.asarray(values)

    def read(self, columns=None, rows=None) -> dict[str, np.ndarray]:
        return {name: self.column(name, rows) for name in (columns or self.columns)}

    def records(self, columns=None, rows=None) -> list[dict]:
        data = self.read(columns, rows)
        names = list(data)
        count = len(data[names[0]]) if names else 0
        return [{name: _plain(data[name][index]) for name in names} for index in range(count)]


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value
