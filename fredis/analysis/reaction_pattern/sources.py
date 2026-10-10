"""Phase 1: resolve analysis sources and load their valid spectra.

A source is a raw spectrum dataset, a fragment-tree preparation (whose
original dataset and record subset are reused), or an earlier Reaction Pattern
Analysis result (whose datasets and column mappings are reused).  Sources only
supply spectra; reaction patterns always come from the Tree Configuration.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .compounds import SampleRecord
from .constants import DEFAULT_COLUMNS, RESULT_SUFFIX
from .hdf5_storage import is_hdf5_result


def is_result_file(path: str | Path) -> bool:
    path = Path(path)
    return is_hdf5_result(path)


def inspect_analysis_source(path: str | Path, kind: str = "auto") -> dict:
    """Identify one source without running chemistry (also used by Workbench)."""
    source = Path(path).resolve()
    if source.is_dir():
        candidate = next(iter(sorted(source.glob("*" + RESULT_SUFFIX))), None)
        if candidate is not None and kind in ("auto", "analysis"):
            source = candidate
    if kind in ("auto", "analysis") and source.is_file() and is_hdf5_result(source):
        import h5py

        from .hdf5_storage import Table

        with h5py.File(source, "r") as handle:
            datasets = Table(handle["datasets"]).records()
        return {"path": str(source), "kind": "analysis", "format": "hdf5", "label": source.parent.name,
                "records": sum(int(row["valid_record_count"]) for row in datasets), "datasets": len(datasets),
                "columns": [], "datasetRows": datasets}
    document = None
    if source.is_file() and source.suffix.lower() in (".json", ".pft"):
        try:
            document = _read_json(source)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            document = None
    detected = kind
    if detected == "auto":
        if source.name.endswith(".preft.pt") or source.is_dir() or (
                document and document.get("application") == "fragment-tree-data-preparation"):
            detected = "prepared"
        else:
            detected = "dataset"
    if detected == "dataset":
        from fredis.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset

        dataset = load_spectrum_dataset(source)
        return {"path": str(source), "kind": detected, "label": source.name,
                "records": len(dataset), "columns": [str(value) for value in dataset.columns],
                "conditions": [], "patterns": []}
    if detected == "analysis":
        raise ValueError(f"Not a Reaction Pattern Analysis result: {source}")
    if detected != "prepared":
        raise ValueError(f"Unknown source kind: {detected}")
    files = _prepared_files(source, document)
    config = _prepared_config(source, document)
    patterns = (config.get("model_config", {}).get("fragmenter_params", {})
                .get("fragment_ion_tree_builder", {}).get("reaction_pattern_set", {}).get("patterns", []))
    manifest = next(iter((source if source.is_dir() else source.parent).glob("manifest.tsv")), None)
    records = None
    if manifest:
        try:
            with manifest.open(newline="") as stream:
                records = sum(int(row.get("num_valid_samples", 0)) for row in csv.DictReader(stream, delimiter="\t"))
        except Exception:
            records = None
    return {"path": str(source), "kind": detected, "label": source.name,
            "records": records, "molecules": len(files), "columns": [], "conditions": [],
            "patterns": [{"name": item.get("name"), "reactant_smarts": item.get("reactant_smarts", ""),
                          "definition": item} for item in patterns]}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _prepared_config(source: Path, document: dict | None = None) -> dict:
    """Preparation settings of a prepared source: its Workbench project file or ``preparation_config.json``."""
    if document and document.get("application") == "fragment-tree-data-preparation":
        return {"input": document.get("input"), "model_config": {
            "fragmenter_params": document.get("fragmenterParams")},
            "columns": {"smiles": document.get("smilesColumn", "SMILES"),
                        "adduct": document.get("adductTypeColumn", "AdductType"),
                        "collision_energy": document.get("collisionEnergyColumn", "CollisionEnergy"),
                        "precursor_mz": document.get("precursorMzColumn", "PrecursorMZ")}}
    start = source if source.is_dir() else source.parent
    config_path = next((parent / "preparation_config.json" for parent in (start, *start.parents)
                        if (parent / "preparation_config.json").exists()), None)
    if config_path is None:
        raise ValueError(f"Cannot find preparation_config.json for {source}")
    value = _read_json(config_path)
    value["_config_path"] = str(config_path)
    return value


def _prepared_files(source: Path, document: dict | None = None) -> list[Path]:
    if source.name.endswith(".preft.pt"):
        return [source]
    root = source
    if source.is_file():
        configured = (document or {}).get("outputDir")
        root = Path(configured).resolve() if configured else source.parent
    files = sorted(root.rglob("*.preft.pt"))
    if not files:
        raise ValueError(f"No .preft.pt structures found in {root}")
    return files


def normalize_analysis_sources(*, sources: Iterable[dict] = (), columns: dict | None = None,
                               show_progress: bool = False) -> list[dict]:
    """Return source rows with resolved kind/path and per-source column mappings."""
    defaults = {**DEFAULT_COLUMNS, **(columns or {})}
    rows = []
    values = list(sources)
    for value in tqdm(values, desc="Identifying sources", unit="source", file=sys.stderr,
                      disable=not show_progress or not values):
        if not isinstance(value, dict) or not value.get("path"):
            raise ValueError("Every analysis source needs a path.")
        row = copy.deepcopy(value)
        inspected = inspect_analysis_source(row["path"], row.get("kind", "auto"))
        row.update(path=inspected["path"], kind=inspected["kind"], inspected=inspected,
                   columns={**defaults, **(row.get("columns") or {})})
        rows.append(row)
    if not rows:
        raise ValueError("At least one dataset, prepared mapping, or analysis result is required.")
    return rows


@dataclass
class DatasetInput:
    dataset_id: int
    path: Path
    label: str
    columns: dict
    source_path: str
    source_kind: str
    record_indexes: list[int] | None = None  # original-record subset (prepared sources)


def expand_dataset_inputs(source_rows: list[dict]) -> list[DatasetInput]:
    """Turn every source into the raw datasets (and record subsets) it stands for."""
    from fredis.analysis.prepared_result import load_prepared_structure

    result: list[DatasetInput] = []

    def add(path, columns, source, record_indexes=None, label=None):
        path = Path(path)
        result.append(DatasetInput(len(result) + 1, path, label or path.name, dict(columns), str(source["path"]),
                                   source["kind"], record_indexes))

    for source in source_rows:
        kind, path = source["kind"], Path(source["path"])
        if kind == "dataset":
            add(path, source["columns"], source)
        elif kind == "prepared":
            document = _read_json(path) if path.is_file() and path.suffix in (".json", ".pft") else None
            prepared = _prepared_config(path, document)
            original = prepared.get("input") or prepared.get("input_path")
            if not original or not Path(original).exists():
                raise ValueError(f"The original dataset of prepared source {path} is unavailable: {original}")
            records = sorted({int(value) for file in _prepared_files(path, document)
                              for value in load_prepared_structure(file)[1].get("record_indexes", ())})
            add(original, {**source["columns"], **(prepared.get("columns") or {})}, source, records)
        elif kind == "analysis":
            inspected = source.get("inspected") or inspect_analysis_source(path, kind)
            for row in inspected["datasetRows"]:
                columns = json.loads(row.get("columns_json") or "{}") or source["columns"]
                if not Path(row["original_path"]).exists():
                    raise ValueError(f"Dataset of reused analysis is unavailable: {row['original_path']}")
                add(row["original_path"], {**DEFAULT_COLUMNS, **columns}, source, label=row["label"])
        else:
            raise ValueError(f"Unknown source kind: {kind}")
    return result


def load_samples(dataset: DatasetInput, *, fragmenter, symbols,
                 minimum_relative_intensity: float) -> tuple[list[SampleRecord], dict]:
    """Load valid spectra of one dataset with Training Data's validation and peak filtering."""
    from fredis.domain.mass.parse_ce import parse_ce_to_ev
    from fredis.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset
    from fredis.ml.data_preparation.fragment_tree.record_validation import inspect_records

    raw = load_spectrum_dataset(dataset.path)
    record_count = len(raw)
    record_map = dataset.record_indexes
    data = raw[record_map] if record_map is not None else raw
    columns = dataset.columns
    mapping = dict(smilesColumn=columns["smiles"], adductTypeColumn=columns["adduct"],
                   collisionEnergyColumn=columns["collision_energy"], precursorMzColumn=columns["precursor_mz"])
    inspection = inspect_records(data, fragmenter, mapping, symbols=symbols)
    smiles = data[columns["smiles"]].tolist()
    adducts = data[columns["adduct"]].tolist()
    energies = data[columns["collision_energy"]].tolist()
    precursors = data[columns["precursor_mz"]].tolist()
    samples: list[SampleRecord] = []
    invalid_peaks = 0
    for local in tqdm(inspection["validIndexes"], desc=f"Loading {dataset.path.name[:32]}", unit="spectrum",
                      file=sys.stderr, leave=False):
        precursor_mz = float(precursors[local])
        energy = parse_ce_to_ev(energies[local], precursor_mz)
        values = [(float(peak.mz), float(peak.intensity)) for peak in data[local].peaks]
        if energy is None or not math.isfinite(float(energy)) or any(
                not math.isfinite(mz) or mz <= 0 or not math.isfinite(value) or value < 0 for mz, value in values):
            invalid_peaks += 1
            continue
        maximum = max((value for _, value in values), default=0.0)
        retained = [(mz, value) for mz, value in values if value >= maximum * minimum_relative_intensity]
        mz = np.asarray([item[0] for item in retained], dtype=np.float64)
        raw_intensity = np.asarray([item[1] for item in retained], dtype=np.float64)
        intensity = raw_intensity / maximum if maximum > 0 else raw_intensity
        top = float(intensity.max()) if intensity.size else 0.0
        samples.append(SampleRecord(
            dataset_id=dataset.dataset_id, record_index=record_map[local] if record_map is not None else local,
            smiles=str(smiles[local]), adduct=str(adducts[local]), collision_energy=float(energy),
            precursor_mz=precursor_mz, mz=mz, intensity=intensity,
            relative_intensity=intensity / top if top > 0 else np.zeros_like(intensity)))
    row = dict(dataset_id=dataset.dataset_id, label=dataset.label, original_path=str(dataset.path),
               source_path=dataset.source_path, source_kind=dataset.source_kind,
               columns_json=json.dumps(columns, ensure_ascii=False), record_count=record_count,
               valid_record_count=len(samples), skipped_record_count=len(data) - len(samples))
    return samples, row
