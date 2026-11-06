"""Shared dataset-list loading, validation and canonical molecule assignment."""
from pathlib import Path
from collections import Counter
import random
import math
from decimal import Decimal, ROUND_FLOOR

import msentity as ms
from .datasets import canonical_keys, load_spectrum_dataset, split_dataset, split_assignments
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


def train_ratio_limit(value):
    if value is None or value == "":
        return None
    try:
        ratio = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Assigned split limits must be finite, non-negative ratios or blank.") from exc
    if isinstance(value, bool) or not math.isfinite(ratio) or ratio < 0:
        raise ValueError("Assigned split limits must be finite, non-negative ratios or blank.")
    return ratio


def validate_list_config(entries, mode, validation_ratio=0, test_ratio=0,
                         validation_max_train_ratio=None, test_max_train_ratio=None):
    from .datasets import validate_split_ratios

    entries = normalize_entries(entries)
    if mode not in ("automatic", "per-dataset", "assigned"):
        raise ValueError("Dataset mode must be automatic, per-dataset or assigned.")
    if mode != "assigned":
        validate_split_ratios(validation_ratio, test_ratio)
    else:
        train_ratio_limit(validation_max_train_ratio)
        train_ratio_limit(test_max_train_ratio)
        if any(entry["split"] not in SPLITS for entry in entries):
            raise ValueError("Assign each dataset to train, validation or test.")
        if not any(entry["split"] == "train" for entry in entries):
            raise ValueError("Assign at least one dataset to train.")
    return entries


def plan_datasets(entries, mode, fragmenter, symbols, validation_ratio=0, test_ratio=0, seed=0,
                  invalid_detail_limit=None, progress=None,
                  validation_max_train_ratio=None, test_max_train_ratio=None, reaction_models=None):
    entries = validate_list_config(entries, mode, validation_ratio, test_ratio,
                                   validation_max_train_ratio, test_max_train_ratio)
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
        selection = None
        if reaction_models is not None:
            from .pattern_filter import select_trainable_pattern_records
            input_smiles = list(data[entry['mapping']['smilesColumn']])
            selected, selection = select_trainable_pattern_records(
                [input_smiles[index] for index in report['validIndexes']],
                fragmenter.reaction_pattern_set, reaction_models,
            )
            report['validIndexes'] = [report['validIndexes'][index] for index in selected]
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
                            records=len(data), excludedRecords=report["excludedRecords"],
                            pattern_selection=selection))
    notify("combining", 72)
    combined = ms.concat(datasets) if len(datasets) > 1 else datasets[0]
    if not len(combined):
        raise ValueError("No records match a trainable reaction pattern before dataset splitting.")
    if mode == "automatic":
        splits, assignments = split_dataset(combined, "SMILES", validation_ratio, test_ratio, seed,
                                              progress=(lambda current, total: notify("splitting", 75 + current / max(1, total) * 24, current=current, total=total)) if progress else None)
    elif mode == "per-dataset":
        import numpy as np

        assignments, all_keys = {}, []
        for source_index, (entry, data) in enumerate(zip(entries, datasets)):
            if not len(data):
                sources[source_index]["splits"] = {name: dict(molecules=0, records=0) for name in SPLITS}
                continue
            try:
                keys, proposed = split_assignments(
                    data["SMILES"], validation_ratio, test_ratio, seed,
                    progress=(lambda current, total: notify(
                        "splitting", 75 + (source_index + current / max(1, total)) * 24 / len(entries),
                        current=current, total=total)) if progress else None,
                )
            except ValueError as exc:
                raise ValueError(f"{entry['path']}: {exc}") from exc
            # Keep earlier assignments of shared molecules to prevent leakage.
            # Allocate new molecules toward this dataset's remaining quotas.
            molecules = set(keys)
            targets = Counter(proposed.values())
            counts = Counter(assignments[key] for key in molecules if key in assignments)
            unseen = sorted(molecules - assignments.keys())
            random.Random(seed).shuffle(unseen)
            for key in unseen:
                split = max(SPLITS, key=lambda name: targets[name] - counts[name])
                assignments[key] = split
                counts[split] += 1
            sources[source_index]["splits"] = {
                name: dict(molecules=counts[name], records=sum(assignments[key] == name for key in keys))
                for name in SPLITS
            }
            all_keys.extend(keys)
        labels = np.array([assignments[key] for key in all_keys])
        splits = {name: combined[np.flatnonzero(labels == name)] for name in SPLITS}
    else:
        import numpy as np

        capped_splits = {
            name for name, value in (("validation", validation_max_train_ratio), ("test", test_max_train_ratio))
            if train_ratio_limit(value) is not None
        }
        if capped_splits:
            # Remove held-out duplicates before counting or sampling. Filter
            # source row indexes too so checked-plan export cannot reintroduce
            # those spectra under their training molecule's global assignment.
            source_keys = [canonical_keys(data["SMILES"]) for data in datasets]
            train_keys = {key for entry, keys in zip(entries, source_keys)
                          if entry["split"] == "train" for key in keys}
            for i, (entry, data, keys) in enumerate(zip(entries, datasets, source_keys)):
                if entry["split"] not in capped_splits:
                    continue
                keep = [row for row, key in enumerate(keys) if key not in train_keys]
                sources[i]["trainOverlapExcludedRecords"] = len(data) - len(keep)
                valid_indexes[entry["id"]] = [valid_indexes[entry["id"]][row] for row in keep]
                datasets[i] = data[keep]
            combined = ms.concat(datasets) if len(datasets) > 1 else datasets[0]
        assignments, grouped = {}, {name: [] for name in SPLITS}
        for source_index, (entry, data) in enumerate(zip(entries, datasets)):
            keys = canonical_keys(data["SMILES"], progress=(lambda current, total: notify("splitting", 75 + (source_index + current / max(1, total)) * 24 / len(entries), current=current, total=total)) if progress else None)
            for key in set(keys):
                if key in assignments and assignments[key] != entry["split"]:
                    raise ValueError(f"Canonical molecule {key} appears in both {assignments[key]} and {entry['split']}. Assign overlapping datasets to the same split or use automatic splitting.")
                assignments[key] = entry["split"]
            grouped[entry["split"]].append(data)
        empty = combined[[]]
        splits = {name: (ms.concat(parts) if len(parts) > 1 else parts[0]) if parts else empty
                  for name, parts in grouped.items()}
        train_count = sum(split == "train" for split in assignments.values())
        rng = random.Random(seed)
        for name, value in (("validation", validation_max_train_ratio), ("test", test_max_train_ratio)):
            ratio = train_ratio_limit(value)
            if ratio is None:
                continue
            limit = int((Decimal(str(ratio)) * train_count).to_integral_value(rounding=ROUND_FLOOR))
            molecules = sorted(key for key, split in assignments.items() if split == name)
            if len(molecules) <= limit:
                continue
            rng.shuffle(molecules)
            for key in molecules[limit:]:
                assignments[key] = "excluded"
            import numpy as np
            keys = canonical_keys(splits[name]["SMILES"])
            splits[name] = splits[name][np.flatnonzero([assignments[key] == name for key in keys])]
    summary = {name: dict(molecules=sum(split == name for split in assignments.values()), records=len(data))
               for name, data in splits.items()}
    notify("complete", 100)
    return dict(entries=entries, combined=combined, splits=splits, assignments=assignments,
                reports=reports, sources=sources, summary=summary, valid_indexes=valid_indexes)
