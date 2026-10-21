"""Model inputs inherited from prepared datasets and training checkpoints."""

from copy import deepcopy
import json
from pathlib import Path


def canonical_symbols(symbols):
    """Return the feature order used by AtomFeatureLayer."""
    return tuple(sorted(set(symbols))) if symbols is not None else None


def _metadata_config(saved):
    """Extract normalized model inputs from any supported preparation file."""
    # preparation_config.json / reaction_statistics.json carry chemistry_config;
    # Workbench project files (*.training-data.pft) carry the same values camelCased.
    for config in (saved.get("chemistry_config"), saved.get("workbench_config"), saved):
        if not isinstance(config, dict):
            continue
        fragmenter = config.get("fragmenter_params") or config.get("fragmenterParams")
        if not fragmenter:
            continue
        result = {
            "fragmenter_params": deepcopy(fragmenter),
            "symbols": deepcopy(config.get("symbols")),
            "reaction_models": deepcopy(
                config.get("reaction_models", config.get("reactionModels", []))
            ),
        }
        if config.get("adduct_type_strs") is not None:
            result["adduct_type_strs"] = list(config["adduct_type_strs"])
        return result
    return None


def _metadata_equal(key, left, right):
    if key == "symbols":
        return canonical_symbols(left) == canonical_symbols(right)
    if key == "adduct_type_strs":
        return tuple(left) == tuple(right)
    return left == right


def dataset_chemistry_config(directory):
    directory = Path(directory)
    if directory.name == "data":
        directory = directory.parent
    candidates = []
    files = [(directory / "preparation_config.json", "preparation_config.json")]
    if directory.name in ("train_structures", "validation_structures", "test_structures"):
        files.append(
            (directory.parent / "preparation_config.json", "../preparation_config.json")
        )
    files.extend(
        (directory / filename, filename)
        for filename in ("fragment-tree.training-data.pft", "reaction_statistics.json")
    )
    for file, filename in files:
        if not file.is_file():
            continue
        saved = json.loads(file.read_text())
        config = _metadata_config(saved)
        if config is not None:
            for record in config.get("reaction_models", []):
                if record.get("source"):
                    record["source"] = str((file.parent / record["source"]).resolve())
            candidates.append((filename, config))
    if not candidates:
        raise ValueError(
            f"Dataset preparation configuration was not found in {directory}"
        )
    for key in ("fragmenter_params", "symbols", "adduct_type_strs", "reaction_models"):
        defined = [
            (filename, config.get(key))
            for filename, config in candidates
            if config.get(key) is not None
        ]
        if len(defined) > 1:
            first_file, first_value = defined[0]
            for filename, value in defined[1:]:
                if not _metadata_equal(key, first_value, value):
                    raise ValueError(
                        f"Dataset metadata conflict for {key}: {first_file} differs from {filename} in {directory}. "
                        "Regenerate this dataset before training."
                    )
    result = deepcopy(candidates[0][1])
    for key in ("fragmenter_params", "symbols", "adduct_type_strs", "reaction_models"):
        if result.get(key) is None:
            for _, config in candidates[1:]:
                if config.get(key) is not None:
                    result[key] = deepcopy(config[key])
                    break
    return result


def dataset_sources(train_dir, val_dir):
    train = dataset_chemistry_config(train_dir)
    validation = dataset_chemistry_config(val_dir)
    if train.get("reaction_models", []) != validation.get("reaction_models", []):
        raise ValueError("Training and validation reaction model selections differ")
    if train["fragmenter_params"] != validation["fragmenter_params"]:
        raise ValueError(
            "Training Fragmenter does not match Validation Fragmenter. Regenerate or reselect compatible datasets."
        )
    if train.get("adduct_type_strs") != validation.get("adduct_type_strs"):
        raise ValueError(
            "Training adduct ordering does not match Validation adduct ordering. Regenerate or reselect compatible datasets."
        )
    train_symbols = canonical_symbols(train.get("symbols"))
    validation_symbols = canonical_symbols(validation.get("symbols"))
    if train_symbols != validation_symbols:
        raise ValueError(
            "Training and validation datasets use different elements: Training symbols do not match Validation symbols. Regenerate or reselect compatible datasets."
        )
    adducts = train.get("adduct_type_strs")
    if not adducts:
        from fredis.domain.fragment.fragmenter import Fragmenter

        adducts = [
            str(value)
            for value in Fragmenter.from_dict(train["fragmenter_params"]).adduct_types
        ]
    return {
        "fragmenter_params": deepcopy(train["fragmenter_params"]),
        "adduct_type_strs": list(adducts),
        "symbols": train_symbols,
        "reaction_models": deepcopy(train.get("reaction_models", [])),
    }


def inherit_model_config(model_config, train_dir, val_dir, *, saved_model=None):
    config = deepcopy(model_config)
    sources = dataset_sources(train_dir, val_dir)
    if saved_model is not None:
        saved = saved_model
        for key in ("fragmenter_params", "adduct_type_strs"):
            if saved.get(key) != sources[key]:
                raise ValueError(f"Dataset {key} differs from the training checkpoint")
        encoder_params = saved["mol_encoder_params"]
    else:
        encoder_params = deepcopy(config.get("mol_encoder_params", {}))
        if not encoder_params:
            raise ValueError("Specify molecular encoder parameters for joint training")
        # A fresh encoder owns its architecture, while the prepared datasets
        # own the atom-feature vocabulary.
        if sources["symbols"] is not None:
            encoder_params["symbols"] = list(sources["symbols"])
    if (
        sources["symbols"] is not None
        and canonical_symbols(encoder_params.get("symbols")) != sources["symbols"]
    ):
        raise ValueError(
            "Dataset elements differ from the molecular encoder checkpoint: Dataset symbols do not match the MolEncoder / applicable checkpoint symbols"
        )
    config.update(
        fragmenter_params=sources["fragmenter_params"],
        adduct_type_strs=sources["adduct_type_strs"],
        mol_encoder_params=deepcopy(encoder_params),
    )
    if saved_model is None and sources.get("reaction_models"):
        from .pattern_models import apply_dataset_pattern_models

        config = apply_dataset_pattern_models(config, sources["reaction_models"])
    return config
