"""Thin JSON adapter around FREDIS dataset and configuration APIs."""

from __future__ import annotations
import contextlib
import json
import math
import sys
from pathlib import Path

DEFAULT_MAPPING = {
    "smilesColumn": "SMILES",
    "adductTypeColumn": "AdductType",
    "collisionEnergyColumn": "CollisionEnergy",
    "precursorMzColumn": "PrecursorMZ",
}


def columns(payload: dict) -> dict:
    """Column-mapping validity only: no peak stats, adduct list or row preview.

    Costs only a dataset load (dominated by file I/O, not row/column count),
    so the Column Mapping indicators can update immediately after choosing a
    file, before the slower per-record value checks (preview/data/check) run.
    """
    from fredis.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset

    file = Path(payload["path"]).resolve()
    dataset = load_spectrum_dataset(file)
    mapping = {**DEFAULT_MAPPING, **payload.get("mapping", {})}
    dataset_columns = dataset.columns
    validation = {
        key: {
            "column": name,
            "exists": name in dataset_columns,
            "valid": 0,
            "total": len(dataset),
        }
        for key, name in mapping.items()
    }
    errors = []
    for key, item in validation.items():
        if not item["exists"]:
            errors.append("Missing " + key + ": " + item["column"])
    return {
        "path": str(file),
        "format": file.suffix.lstrip(".").upper(),
        "size": file.stat().st_size,
        "records": len(dataset),
        "columns": dataset_columns,
        "validation": validation,
        "errors": errors,
    }


def preview(payload: dict) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from fredis.libs.mmkit.mmkit import Adduct
    from fredis.domain.mass.parse_ce import parse_ce_to_ev
    from fredis.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset

    file = Path(payload["path"]).resolve()
    dataset = load_spectrum_dataset(file)
    mapping = {**DEFAULT_MAPPING, **payload.get("mapping", {})}
    columns = dataset.columns
    validation = {
        key: {
            "column": name,
            "exists": name in columns,
            "valid": 0,
            "total": len(dataset),
        }
        for key, name in mapping.items()
    }
    errors = []
    for key, item in validation.items():
        if not item["exists"]:
            errors.append("Missing " + key + ": " + item["column"])
    check_values = bool(payload.get("validateValues", False))
    fragmenter = None
    if check_values:
        from fredis.domain.fragment.fragmenter import Fragmenter

        if not payload.get("fragmenterParams"):
            raise ValueError("Fragmenter parameters are required to check values.")
        fragmenter = Fragmenter.from_dict(payload["fragmenterParams"])
    inspection = {}
    if check_values and not errors:
        from fredis.ml.data_preparation.fragment_tree.record_validation import (
            inspect_records,
        )

        def report_progress(current, total):
            print(
                "FREDIS_PROGRESS " + json.dumps({"current": current, "total": total}),
                file=sys.stderr,
                flush=True,
            )

        # Every record is checked. Only detailed invalid-record samples are
        # bounded so a very dirty dataset cannot overflow the webview response.
        inspection = inspect_records(
            dataset,
            fragmenter,
            mapping,
            progress=report_progress if payload.get("reportProgress") else None,
            invalid_detail_limit=5000,
            symbols=payload.get("symbols"),
        )
        validation = inspection["validation"]
        if not inspection["eligibleRecords"]:
            errors.append("The dataset contains no valid records to prepare.")
    import numpy as np

    lengths = dataset.peaks.lengths
    eligible = set(inspection.get("validIndexes", range(len(dataset))))
    checked_peaks = (
        dataset[inspection["validIndexes"]].peaks if inspection else dataset.peaks
    )
    if np.any(checked_peaks.lengths == 0):
        errors.append(f"{int((checked_peaks.lengths==0).sum())} spectra have no peaks.")
    if not np.isfinite(checked_peaks.mz).all() or np.any(checked_peaks.mz <= 0):
        errors.append("Peak m/z values must be finite and positive.")
    if not np.isfinite(checked_peaks.intensity).all() or np.any(
        checked_peaks.intensity < 0
    ):
        errors.append("Peak intensities must be finite and non-negative.")
    # MSDataset.metadata rebuilds a full-length view on every dataset[name]
    # access, so each mapped/id column is fetched exactly once here; reading
    # them inside the row loop below would otherwise cost O(limit * columns)
    # full-dataset rebuilds instead of O(columns) for a large dataset.
    mapped_series = {
        key: dataset[name] if name in columns else None for key, name in mapping.items()
    }
    spec_id_series = dataset["SpecID"] if "SpecID" in columns else None
    adducts = (
        mapped_series["adductTypeColumn"].astype(str).value_counts().to_dict()
        if mapped_series["adductTypeColumn"] is not None
        else {}
    )
    normalized_adducts = []
    for value in adducts:
        try:
            normalized_adducts.append(str(Adduct.parse(value)))
        except Exception:
            pass
    smiles = (
        mapped_series["smilesColumn"].astype(str)
        if mapped_series["smilesColumn"] is not None
        else None
    )
    rows = []
    limit = min(max(int(payload.get("limit", 50)), 1), 100)
    for index in range(min(len(dataset), limit)):
        record = dataset[index]
        values = {
            key: str(series.iloc[index])[:4096] if series is not None else ""
            for key, series in mapped_series.items()
        }
        spectrum = list(record.peaks)
        if index in eligible and any(
            not math.isfinite(float(p.mz))
            or float(p.mz) <= 0
            or not math.isfinite(float(p.intensity))
            or float(p.intensity) < 0
            for p in spectrum
        ):
            errors.append(f"Invalid peak values in preview record {index}")
        rows.append(
            {
                "index": index,
                "id": (
                    str(spec_id_series.iloc[index])
                    if spec_id_series is not None
                    else str(index)
                ),
                **values,
                "numPeaks": int(lengths[index]),
                "peaks": [
                    {"mz": float(p.mz), "intensity": float(p.intensity)}
                    for p in spectrum[:2000]
                    if math.isfinite(float(p.mz)) and math.isfinite(float(p.intensity))
                ],
            }
        )
    structure = None
    if rows:
        mol = Chem.MolFromSmiles(rows[0]["smilesColumn"])
        if mol:
            from rdkit.Chem.Draw import rdMolDraw2D

            drawer = rdMolDraw2D.MolDraw2DSVG(320, 230)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            structure = {
                "svg": drawer.GetDrawingText(),
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "exactMass": Descriptors.ExactMolWt(mol),
            }
    if len(dataset) == 0:
        errors.append("The dataset contains no spectra.")
    return {
        "path": str(file),
        "format": file.suffix.lstrip(".").upper(),
        "size": file.stat().st_size,
        "records": len(dataset),
        "columns": columns,
        "summary": {
            "uniqueSmiles": int(smiles.nunique()) if smiles is not None else None,
            "adducts": adducts,
            "normalizedAdducts": normalized_adducts,
            "averagePeaks": float(lengths.mean()) if len(lengths) else 0,
        },
        "valuesChecked": check_values,
        **{key: value for key, value in inspection.items() if key != "validIndexes"},
        "validation": validation,
        "errors": list(dict.fromkeys(errors)),
        "rows": rows,
        "structure": structure,
        **(
            {
                "_smiles": [
                    smiles.iloc[index]
                    for index in inspection.get("validIndexes", range(len(dataset)))
                ]
            }
            if payload.get("_identifiers") and smiles is not None
            else {}
        ),
    }


def validate(payload: dict, *, check_datasets: bool = True) -> dict:
    from fredis.ml.data_preparation.fragment_tree.context import (
        create_preparation_context,
        validate_limits,
    )

    config = {
        "fragmenter_params": payload["fragmenterParams"],
        "symbols": payload.get("symbols"),
    }
    if payload.get("reactionModels"):
        from fredis.ml.training.fragment_tree_training.pattern_models import (
            prepare_pattern_models,
        )

        config = prepare_pattern_models(config, payload["reactionModels"])
    generator = create_preparation_context(config)
    validate_limits(
        payload.get("maxUniqueFragmentSmiles", -1),
        payload.get("maxReactionCombinations", -1),
    )
    for key in ("numWorkers", "chunkSize"):
        value = payload.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(
                "Worker processes and chunk size must be positive integers."
            )
    minimum = float(payload.get("minimumRelativeIntensity", 0))
    if not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("Minimum relative intensity must be between 0 and 1.")
    if "datasets" in payload:
        from fredis.ml.data_preparation.fragment_tree.dataset_plan import (
            plan_datasets,
            validate_list_config,
        )

        mode = payload.get("datasetMode", "automatic")
        ratio = float(payload.get("validationRatio") or 0)
        test_ratio = float(payload.get("testRatio") or 0)
        seed = payload.get("validationSeed", 0)
        if type(seed) is not int:
            raise ValueError("Split seed must be an integer.")
        entries = validate_list_config(payload["datasets"], mode, ratio, test_ratio)
        if not check_datasets:
            return {"valid": True, "architecture": generator.architecture}
        plan = plan_datasets(
            entries,
            mode,
            generator.fragmenter,
            config["symbols"],
            ratio,
            test_ratio,
            seed,
            invalid_detail_limit=5000,
        )
        import numpy as np

        peaks = plan["combined"].peaks
        if np.any(peaks.lengths == 0):
            raise ValueError("Some spectra have no peaks.")
        if not np.isfinite(peaks.mz).all() or np.any(peaks.mz <= 0):
            raise ValueError("Peak m/z values must be finite and positive.")
        if not np.isfinite(peaks.intensity).all() or np.any(peaks.intensity < 0):
            raise ValueError("Peak intensities must be finite and non-negative.")
        return {
            "valid": True,
            "architecture": generator.architecture,
            "splits": plan["summary"],
            "sources": plan["sources"],
        }
    from fredis.ml.data_preparation.fragment_tree.datasets import (
        split_assignments,
        canonical_keys,
        validate_split_ratios,
    )

    ratio = float(payload.get("validationRatio") or 0)
    test_ratio = float(payload.get("testRatio") or 0)
    seed = payload.get("validationSeed", 0)
    if type(seed) is not int:
        raise ValueError("Split seed must be an integer.")
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1.")
    # Validate fractions even when the configuration check skips dataset loading.
    validate_split_ratios(0 if payload.get("validationInput") else ratio, test_ratio)
    split_summary = {}
    if check_datasets and payload.get("input"):
        train_mapping = {
            key: payload.get(key, default) for key, default in DEFAULT_MAPPING.items()
        }
        validation_mapping = {
            key: payload.get("validation" + key[0].upper() + key[1:]) or value
            for key, value in train_mapping.items()
        }
        sources = []
        for file, mapping in (
            (payload["input"], train_mapping),
            (payload.get("validationInput"), validation_mapping),
        ):
            if not file:
                continue
            result = preview(
                {
                    "path": file,
                    "mapping": mapping,
                    "limit": 1,
                    "_identifiers": True,
                    "validateValues": True,
                    "fragmenterParams": generator.fragmenter.to_dict(),
                    "symbols": config["symbols"],
                }
            )
            sources.append(result.pop("_smiles", []))
            if result["errors"]:
                raise ValueError("; ".join(result["errors"]))
        keys, assignments = split_assignments(
            sources[0], 0 if payload.get("validationInput") else ratio, test_ratio, seed
        )
        for name in ("train", "validation", "test"):
            split_summary[name] = {
                "molecules": sum(value == name for value in assignments.values()),
                "records": sum(assignments[key] == name for key in keys),
            }
        if len(sources) == 2:
            remaining = canonical_keys(sources[1])
            if set(remaining) & assignments.keys():
                raise ValueError(
                    "Training and validation datasets share SMILES (RDKit canonical). Choose molecule-disjoint datasets."
                )
            remaining = [key for key in remaining if key not in assignments]
            available = list(dict.fromkeys(remaining))
            if ratio:
                import random

                random.Random(seed).shuffle(available)
                available = available[: max(1, round(len(assignments) * ratio))]
            kept = set(available)
            split_summary["validation"] = {
                "molecules": len(kept),
                "records": sum(key in kept for key in remaining),
            }
    return {
        "valid": True,
        "architecture": generator.architecture,
        "splits": split_summary,
    }


def model(payload: dict) -> dict:
    import torch

    checkpoint = torch.load(payload["path"], map_location="cpu")
    if checkpoint.get("architecture") != "fragment-tree-competitive-ion":
        raise ValueError("Choose a compatible physical-ion fragment-tree checkpoint.")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a model configuration.")
    return {"modelConfig": config, "path": payload["path"]}


def structure_manifest(payload):
    import torch
    from fredis.ml.data_preparation.fragment_tree.manifest_summary import (
        structure_manifest_fields,
    )

    directory = Path(payload["directory"])
    rows = []
    files = sorted(
        set(directory.glob("*.preft.pt")) | set((directory / "data").glob("*.preft.pt"))
    )
    for file in files:
        saved = torch.load(file, map_location="cpu", weights_only=False)
        metadata = saved.get("metadata", {})
        indexes = metadata.get("record_indexes", [])
        rows.append(
            dict(
                file=str(file.relative_to(directory)),
                smiles=metadata.get("smiles", ""),
                record_indexes=json.dumps(indexes),
                num_input_records=len(indexes),
                num_valid_samples=int(saved["structure"].num_samples),
                rejected_sample_count=max(
                    0, len(indexes) - int(saved["structure"].num_samples)
                ),
                rejection_log="",
                **structure_manifest_fields(saved["structure"]),
                status="completed",
            )
        )
    return rows


def training_checkpoint(payload):
    import torch

    checkpoint = torch.load(payload["path"], map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "fragment-tree-competitive-ion":
        raise ValueError("Checkpoint architecture is incompatible with training.")
    config = checkpoint.get("model_config")
    if not config:
        raise ValueError("Checkpoint does not contain model configuration.")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError(
            "This checkpoint cannot resume training. Import its reaction models in New model instead."
        )
    group = checkpoint.get("optimizer_state_dict", {}).get(
        "param_groups", [{"lr": 1e-4, "weight_decay": 0.01}]
    )[0]
    fields = {
        "branch_weight": "branchWeight",
        "negative_weight": "negativeWeight",
        "validation_interval_steps": "validationIntervalSteps",
        "validation_fraction": "validationFraction",
        "intensity_loss_weight": "intensityLossWeight",
        "cosine_loss_weight": "cosineLossWeight",
    }
    restored = {
        field: checkpoint.get("training_settings", {})[key]
        for key, field in fields.items()
        if key in checkpoint.get("training_settings", {})
    }
    params = config
    restored["dropout"] = params.get("reaction_model_params", {}).get("edge_dropout", 0)
    return {
        "modelConfig": config,
        "patternModels": describe_patterns(config),
        "lr": group["lr"],
        "weightDecay": group.get("weight_decay", 0),
        "trainingSettings": restored,
    }


def training_sources(payload):
    import torch
    from fredis.ml.training.fragment_tree_training.sources import (
        inherit_model_config,
        dataset_sources,
    )

    saved = None
    path = payload.get("resume") or payload.get("fineTuneCheckpoint")
    if path:
        saved = torch.load(path, map_location="cpu", weights_only=False)["model_config"]
        if not payload.get("resume"):
            inherited = dataset_sources(payload["trainDir"], payload["valDir"])
            saved = {
                **saved,
                "fragmenter_params": inherited["fragmenter_params"],
                "adduct_type_strs": inherited["adduct_type_strs"],
            }
    config = inherit_model_config(
        saved or payload.get("modelConfig") or {},
        payload["trainDir"],
        payload["valDir"],
        saved_model=saved,
    )
    if not payload.get("resume") and payload.get("pretrainedPatterns"):
        from fredis.ml.training.fragment_tree_training.pattern_models import (
            assemble_config,
        )

        config = assemble_config(config, payload["pretrainedPatterns"])
        config = inherit_model_config(
            config, payload["trainDir"], payload["valDir"], saved_model=config
        )
    return {"modelConfig": config, "patternModels": describe_patterns(config)}


def training_assignment_score_counts(payload):
    from fredis.ml.training.fragment_tree_training.prepared_dataset import (
        prepared_piece_sequence,
    )

    minimum = float(payload.get("minimumAssignmentScore", 0))
    minimum_without_precursor = float(
        payload.get("minimumAssignmentScoreWithoutPrecursor", 0)
    )
    for value, label in (
        (minimum, "Minimum assignment score"),
        (minimum_without_precursor, "Minimum assignment score without precursor"),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{label} must be between 0 and 1.")
    result = {}
    for field, directory in (
        ("train", payload.get("trainDir")),
        ("validation", payload.get("valDir")),
    ):
        if not directory:
            continue
        directory = Path(directory)
        try:
            stats = json.loads((directory / "reaction_statistics.json").read_text())
        except (OSError, json.JSONDecodeError):
            stats = {}
        pieces = prepared_piece_sequence(
            directory,
            max_samples=2**31 - 1,
            minimum_assignment_score=minimum,
            minimum_assignment_score_without_precursor=minimum_without_precursor,
            cache_in_memory=False,
        )
        result[field] = dict(
            rawRecords=stats.get("num_metadata_valid_records"),
            samples=pieces.total_unfiltered_samples,
            filteredSamples=pieces.total_samples,
        )
    return result


def pattern_definitions(payload):
    from fredis.domain.fragment.reaction import ReactionPatternSet

    value = json.loads(Path(payload["path"]).read_text())
    value = value.get("reaction_pattern_set", value.get("reaction_pattern", value))
    if "reactant_smarts" in value:
        value = {"name": "selected", "patterns": [value]}
    patterns = ReactionPatternSet.from_dict(value).patterns
    return {"patterns": [p.to_dict() for p in patterns]}


def describe_patterns(config):
    from fredis.domain.fragment import Fragmenter
    from fredis.ml.training.fragment_tree_training.pattern_models import pattern_config

    imports = {
        item["new_id"]: item["path"]
        for item in config.get("fine_tuning", {}).get("pattern_imports", [])
    }
    return [
        dict(
            pattern=p.to_dict(),
            architecture=pattern_config(config, p.pattern_id),
            source=imports.get(p.pattern_id),
            atom_count=p.reactant_query.GetNumAtoms(),
        )
        for p in Fragmenter.from_dict(
            config["fragmenter_params"]
        ).reaction_pattern_set.patterns
    ]


def model_inspect(payload):
    from fredis.ml.training.fragment_tree_training.pattern_models import inspect_model

    return inspect_model(payload["path"])


def pattern_export(payload):
    from fredis.ml.training.fragment_tree_training.pattern_models import export_pattern

    return export_pattern(payload["path"], payload["pattern_id"], payload["output"])


def training_integrate(payload):
    from fredis.ml.training.fragment_tree_training.integration import (
        integrate_checkpoints,
    )

    return integrate_checkpoints(
        payload["checkpoints"], payload["headCheckpoint"], payload["output"]
    )


def main():
    request = json.loads(sys.stdin.read())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = {
                "pattern-definitions": pattern_definitions,
                "model-inspect": model_inspect,
                "pattern-export": pattern_export,
                "training-integrate": training_integrate,
                "training-sources": training_sources,
                "training-checkpoint": training_checkpoint,
                "training-assignment-score-counts": training_assignment_score_counts,
                "structure-manifest": structure_manifest,
                "columns": columns,
                "preview": preview,
                "validate": validate,
                "validate-config": lambda payload: validate(
                    payload, check_datasets=False
                ),
                "model": model,
            }[request["command"]](request["payload"])
        print(json.dumps({"ok": True, "result": result}, allow_nan=False))
    except Exception as error:
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(error)}))


if __name__ == "__main__":
    main()
