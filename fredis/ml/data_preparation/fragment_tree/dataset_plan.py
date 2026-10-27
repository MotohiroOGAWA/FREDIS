"""Shared dataset-list loading, validation and canonical molecule assignment."""
from pathlib import Path

from fredis.data import MSDataset
from .datasets import canonical_keys, load_spectrum_dataset, split_dataset
from .record_validation import inspect_records

DEFAULT_MAPPING = dict(smilesColumn="SMILES", adductTypeColumn="AdductType",
                       collisionEnergyColumn="CollisionEnergy", precursorMzColumn="PrecursorMZ")
SPLITS = ("train", "validation", "test")


def normalize_entries(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Add at least one dataset.")
    normalized = []
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"].strip():
            raise ValueError(f"Dataset {index + 1} requires a path.")
        identity = str(entry.get("id") or f"dataset-{index + 1}")
        if identity in seen:
            raise ValueError("Dataset IDs must be unique.")
        seen.add(identity)
        mapping = entry.get("mapping") or {}
        if not isinstance(mapping, dict):
            raise ValueError("Dataset column mapping must be an object.")
        mapping = {key: mapping.get(key, default) for key, default in DEFAULT_MAPPING.items()}
        if any(not isinstance(value, str) or not value.strip() for value in mapping.values()):
            raise ValueError("Dataset column names must be non-empty strings.")
        normalized.append(dict(id=identity, path=str(Path(entry["path"].strip()).expanduser().resolve()),
                               split=entry.get("split", "train"), mapping=mapping))
    return normalized


def validate_list_config(entries, mode, validation_ratio=0, test_ratio=0):
    from .datasets import validate_split_ratios

    entries = normalize_entries(entries)
    if mode not in ("automatic", "assigned"):
        raise ValueError("Dataset mode must be automatic or assigned.")
    if mode == "automatic":
        validate_split_ratios(validation_ratio, test_ratio)
    else:
        if any(entry["split"] not in SPLITS for entry in entries):
            raise ValueError("Assign each dataset to train, validation or test.")
        if not any(entry["split"] == "train" for entry in entries):
            raise ValueError("Assign at least one dataset to train.")
    return entries


def plan_datasets(entries, mode, fragmenter, symbols, validation_ratio=0, test_ratio=0, seed=0,
                  invalid_detail_limit=None, progress=None):
    entries = validate_list_config(entries, mode, validation_ratio, test_ratio)
    datasets, reports, sources, valid_indexes = [], {}, [], {}
    def notify(stage, percent, **details):
        if progress:
            progress(dict(stage=stage, percent=percent, **details))

    for source_index, entry in enumerate(entries):
        notify("loading", source_index * 70 / len(entries), datasetIndex=source_index + 1,
               datasetsTotal=len(entries), path=entry["path"], current=0, total=0)
        data = load_spectrum_dataset(entry["path"])
        def inspected(current, total):
            notify("validating", (source_index + current / max(1, total)) * 70 / len(entries),
                   datasetIndex=source_index + 1, datasetsTotal=len(entries),
                   path=entry["path"], current=current, total=total)
        report = inspect_records(data, fragmenter, entry["mapping"], symbols=symbols,
                                 invalid_detail_limit=invalid_detail_limit, progress=inspected if progress else None)
        reports[entry["id"]] = report["invalidRecords"]
        valid_indexes[entry["id"]] = report["validIndexes"]
        if not report["eligibleRecords"]:
            raise ValueError(f"{entry['path']}: no valid records to prepare.")
        # Materialize each filtered view, preserving peaks and unrelated metadata.
        data = data[report["validIndexes"]].copy()
        columns = {DEFAULT_MAPPING[key]: data[column].copy() for key, column in entry["mapping"].items()}
        for column, values in columns.items():
            data[column] = values
        data["_fredis_dataset_path"] = str(Path(entry["path"]).resolve())
        data["_fredis_source_record"] = report["validIndexes"]
        datasets.append(data)
        sources.append(dict(id=entry["id"], path=entry["path"], split=entry["split"],
                            records=len(data), excludedRecords=report["excludedRecords"]))
    notify("combining", 72)
    combined = MSDataset.concat(datasets) if len(datasets) > 1 else datasets[0]
    if mode == "automatic":
        splits, assignments = split_dataset(combined, "SMILES", validation_ratio, test_ratio, seed,
                                              progress=(lambda current, total: notify("splitting", 75 + current / max(1, total) * 24, current=current, total=total)) if progress else None)
    else:
        assignments, grouped = {}, {name: [] for name in SPLITS}
        for source_index, (entry, data) in enumerate(zip(entries, datasets)):
            keys = canonical_keys(data["SMILES"], progress=(lambda current, total: notify("splitting", 75 + (source_index + current / max(1, total)) * 24 / len(entries), current=current, total=total)) if progress else None)
            for key in set(keys):
                if key in assignments and assignments[key] != entry["split"]:
                    raise ValueError(f"Canonical molecule {key} appears in both {assignments[key]} and {entry['split']}. Assign overlapping datasets to the same split or use automatic splitting.")
                assignments[key] = entry["split"]
            grouped[entry["split"]].append(data)
        empty = combined[[]]
        splits = {name: (MSDataset.concat(parts) if len(parts) > 1 else parts[0]) if parts else empty
                  for name, parts in grouped.items()}
    summary = {name: dict(molecules=sum(split == name for split in assignments.values()), records=len(data))
               for name, data in splits.items()}
    notify("complete", 100)
    return dict(entries=entries, combined=combined, splits=splits, assignments=assignments,
                reports=reports, sources=sources, summary=summary, valid_indexes=valid_indexes)
