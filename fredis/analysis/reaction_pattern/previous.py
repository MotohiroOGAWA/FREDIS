"""Reuse the conditions an existing result already calculated.

A run with ``extend`` keeps the result in its output directory and only
calculates conditions it does not have.  Conditions are identified by the
content of their patterns (``tree.condition_key``), so renaming pattern sets
or moving nodes never forces a recalculation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from .constants import RESULT_NAME, SCHEMA, SCHEMA_VERSION
from .tree import PatternRegistry, TreeConfiguration

@dataclass
class PreviousOutcome:
    condition_key: str
    status: str
    skip_category: str
    reason: str
    error: str
    kept: tuple[int, ...]          # sample_index values
    detail_path: str               # absolute path or ""
    seed_path: str                 # absolute path or ""
    assigned_column: str
    count_column: str | None


def _split(value) -> list[str]:
    return [item for item in str(value or "").split(",") if item != ""]


class PreviousResult:
    def __init__(self, path: str | Path):
        import h5py

        from .hdf5_storage import Table

        path = Path(path)
        if path.is_dir():
            path = path / RESULT_NAME
        self.path = path.resolve()
        self.root = self.path.parent
        with h5py.File(self.path, "r") as handle:
            attrs = dict(handle["metadata"].attrs)
            if attrs.get("schema") != SCHEMA:
                raise ValueError(f"Not a Reaction Pattern Analysis result: {self.path}")
            if attrs.get("analysis_status") not in (None, "completed"):
                raise ValueError(f"The existing result is incomplete ({attrs.get('analysis_status')}): {self.path}")
            self.schema_version = int(attrs.get("schema_version", 0))
            if self.schema_version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported Reaction Pattern Analysis schema version {self.schema_version}: {self.path}")
            self.configuration = json.loads(handle["configuration/json"].asstr()[()])
            read = lambda name, columns=None: Table(handle[name]).records(columns)
            self.datasets = read("datasets")
            self._load_tree_result(read)

    # -- loading ------------------------------------------------------------------
    def _identity(self, rows: list[dict]) -> None:
        self.samples: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
        for row in rows:
            self.samples[row["compound_id"]][int(row["sample_index"])] = (int(row["dataset_id"]), int(row["record_index"]))

    def _load_tree_result(self, read) -> None:
        self.patterns = [{"pattern_id": row["pattern_id"], "definition": json.loads(row["definition_json"])}
                         for row in read("patterns")]
        self.registry = PatternRegistry(self.patterns)
        self.tree = TreeConfiguration.from_dict(self.configuration.get("tree") or {})
        keys = {int(row["condition_id"]): row["condition_key"] for row in read("conditions")}
        self.outcomes: dict[str, dict[str, PreviousOutcome]] = defaultdict(dict)
        for row in read("compound_condition"):
            condition_id = int(row["condition_id"])
            self.outcomes[row["compound_id"]][keys[condition_id]] = PreviousOutcome(
                keys[condition_id], row["status"], row["skip_category"] or "", row["reason"] or "",
                row["error"] or "", tuple(int(value) for value in _split(row["kept"])),
                str(self.root / row["detail_path"]) if row["detail_path"] else "",
                str(self.root / row["seed_path"]) if row["seed_path"] else "",
                f"assigned::{condition_id}", f"count::{condition_id}")
        self._identity(read("sample_summary", ["compound_id", "sample_index", "dataset_id", "record_index"]))
        self.peak_files = {row["compound_id"]: row["peak_summary_file"]
                           for row in read("compounds", ["compound_id", "peak_summary_file"])}

    # -- queries --------------------------------------------------------------------
    def compound_outcomes(self, compound_id: str) -> dict[str, PreviousOutcome]:
        return self.outcomes.get(compound_id, {})

    def peak_file(self, compound_id: str) -> Path | None:
        name = self.peak_files.get(compound_id)
        return self.root / name if name else None

    @property
    def run(self) -> dict:
        return self.configuration.get("run") or {}

    @property
    def chemistry_config(self) -> dict:
        return self.configuration.get("chemistryConfig") or {}


def chemistry_signature(chemistry_config: dict, settings: dict) -> dict:
    """Everything except the reaction patterns that changes a calculated tree."""
    import copy

    chemistry = copy.deepcopy(chemistry_config or {})
    builder = chemistry.get("fragmenter_params", {}).get("fragment_ion_tree_builder", {})
    builder.pop("reaction_pattern_set", None)
    keys = ("maxUniqueFragmentSmiles", "maxReactionCombinations", "minimumRelativeIntensity")
    return {"fragmenter_params": chemistry.get("fragmenter_params"),
            "symbols": chemistry.get("symbols"),
            **{key: settings.get(key) for key in keys}}


def compare_signatures(previous: dict, current: dict) -> list[str]:
    differences = []
    for key in sorted(set(previous) | set(current)):
        if json.dumps(previous.get(key), sort_keys=True, default=str) != json.dumps(current.get(key), sort_keys=True, default=str):
            differences.append(key)
    return differences
