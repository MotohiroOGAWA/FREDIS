"""Portable, complete immutable reaction features and architecture assembly."""

from copy import deepcopy
from pathlib import Path
import math
import torch
from fredis.domain.fragment import Fragmenter
from .integration import PATTERN_KEY

FORMAT = "fredis-reaction-model-v1"
SHARED_SETTINGS = ("hidden_dim",)


def load(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("architecture") not in ("fragment-tree-competitive-ion", FORMAT):
        raise ValueError("Expected a prediction checkpoint or .reaction-model.pt file")
    return saved


def pattern_config(params, pid):
    reaction = params.get("reaction_model_params", {})
    hidden = reaction.get("hidden_dim", 128)
    result = dict(
        hidden_dim=reaction.get("reactant_node_dim", hidden),
        graph_dim=reaction.get("reactant_graph_dim", hidden),
        num_layers=reaction.get("reactant_num_layers", 2),
        num_heads=reaction.get("num_heads", 4),
        dropout=params.get("dropout", 0.0),
        descriptors=[],
        input_mode="chemical",
        neighborhood_hops=3,
        neighborhood_hidden_dim=32,
        neighborhood_edge_features=True,
    )
    override = dict(reaction.get("pattern_model_params", {}).get(str(pid), {}))
    result.update(override)
    frozen = set(map(str, params.get("fine_tuning", {}).get("frozen_pattern_ids", [])))
    if str(pid) not in frozen:
        result["dropout"] = params.get("dropout", 0.0)
    return result


def use_shared_dropout(config):
    """Keep one authoritative dropout setting for all trainable networks."""
    config = deepcopy(config)
    reaction = config.setdefault("reaction_model_params", {})
    rate = config.get("dropout", 0.0)
    if not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("Shared dropout must be finite and in [0, 1)")
    config["dropout"] = rate
    frozen = {str(pid) for pid in config.get("fine_tuning", {}).get("frozen_pattern_ids", [])}
    if not config.get("fine_tuning"):
        config.setdefault("mol_encoder_params", {})["dropout"] = rate
    spectrum = config.setdefault("spectrum_model_params", {})
    for key in ("fragment_dropout", "state_dropout"):
        spectrum.pop(key, None)
    for key in ("edge_dropout", "reactant_dropout"):
        reaction.pop(key, None)
    for pid, params in reaction.get("pattern_model_params", {}).items():
        if str(pid) not in frozen:
            params.pop("dropout", None)
    return config


def entries(saved):
    params = saved["model_config"]
    patterns = Fragmenter.from_dict(
        params["fragmenter_params"]
    ).reaction_pattern_set.patterns
    selected = saved.get("pattern_ids")
    return [p for p in patterns if selected is None or p.pattern_id in selected]


def inspect_model(path):
    saved = load(path)
    params = saved["model_config"]
    state = saved["model_state_dict"]
    result = []
    for p in entries(saved):
        config = pattern_config(params, p.pattern_id)
        count = sum(
            t.numel()
            for name, t in state.items()
            if (match := PATTERN_KEY.search(name))
            and int(match.group(1)) == p.pattern_id
        )
        result.append(
            dict(
                pattern=p.to_dict(),
                architecture=config,
                atom_count=p.reactant_query.GetNumAtoms(),
                parameter_count=count,
                source=str(Path(path).resolve()),
            )
        )
    return dict(
        path=str(Path(path).resolve()),
        architecture=saved["architecture"],
        modelConfig=params,
        patterns=result,
        resumable="optimizer_state_dict" in saved,
    )


def export_pattern(checkpoint, pattern_id, output):
    saved = load(checkpoint)
    pattern_id = int(pattern_id)
    if pattern_id not in {p.pattern_id for p in entries(saved)}:
        raise ValueError("Unknown reaction pattern ID")
    destination = Path(output)
    if not str(destination).endswith(".reaction-model.pt"):
        raise ValueError("Single-pattern files must end in .reaction-model.pt")
    if destination.exists():
        raise ValueError("Output already exists")
    # Shared source encoders are dependencies of the complete reaction feature.
    state = {
        name: value
        for name, value in saved["model_state_dict"].items()
        if name.startswith(
            (
                "feature_model.mol_encoder.",
                "spectrum_model.mol_encoder.",
                "feature_model.reaction_instance_encoder.",
            )
        )
        and (
            not (match := PATTERN_KEY.search(name)) or int(match.group(1)) == pattern_id
        )
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            architecture=FORMAT,
            model_config=saved["model_config"],
            pattern_ids=[pattern_id],
            model_state_dict=state,
        ),
        destination,
    )
    return inspect_model(destination)


def assemble_config(config, imports):
    """Inherit shared dependencies, attach each imported feature, train the rest."""
    config = deepcopy(config)
    target = Fragmenter.from_dict(config["fragmenter_params"]).reaction_pattern_set
    by_key = {p.key: p for p in target.patterns}
    owners = {}
    shared = None
    common_params = None
    reaction = config.setdefault("reaction_model_params", {})
    overrides = reaction.setdefault("pattern_model_params", {})
    bindings = []
    for item in imports:
        saved = load(item["path"])
        params = saved["model_config"]
        state = saved["model_state_dict"]
        current = {
            name: value
            for name, value in state.items()
            if name.startswith(
                (
                    "feature_model.mol_encoder.",
                    "spectrum_model.mol_encoder.",
                    "feature_model.reaction_instance_encoder.",
                )
            )
            and not PATTERN_KEY.search(name)
        }
        if shared is None:
            shared = current
            common_params = params
            config["mol_encoder_params"] = deepcopy(params["mol_encoder_params"])
            for name in SHARED_SETTINGS:
                if name in params.get("reaction_model_params", {}):
                    reaction[name] = deepcopy(params["reaction_model_params"][name])
                else:
                    reaction.pop(name, None)
        elif (
            set(shared) != set(current)
            or any(not torch.equal(shared[k], current[k]) for k in shared)
            or params["mol_encoder_params"] != common_params["mol_encoder_params"]
            or any(
                params.get("reaction_model_params", {}).get(k)
                != common_params.get("reaction_model_params", {}).get(k)
                for k in SHARED_SETTINGS
            )
        ):
            raise ValueError("Imported models have conflicting shared source features")
        ids = item.get("pattern_ids")
        selected = [p for p in entries(saved) if ids is None or p.pattern_id in ids]
        if ids is not None and set(ids) != {p.pattern_id for p in selected}:
            raise ValueError("Unknown imported pattern IDs")
        for p in selected:
            if p.key not in by_key:
                raise ValueError(
                    f"Imported pattern {p.name} is absent from prepared datasets; regenerate the data with this pattern"
                )
            pid = by_key[p.key].pattern_id
            if pid in owners:
                raise ValueError(f"Duplicate pretrained pattern: {p.name}")
            owners[pid] = item["path"]
            overrides[str(pid)] = pattern_config(params, p.pattern_id)
            bindings.append(
                dict(
                    path=str(Path(item["path"]).resolve()),
                    old_id=p.pattern_id,
                    new_id=pid,
                )
            )
    if bindings:
        config["fine_tuning"] = dict(
            frozen_pattern_ids=list(owners),
            pattern_imports=bindings,
            pattern_mapping=[],
        )
    else:
        config.pop("fine_tuning", None)
    return config


def initialize_imports(model, config):
    target = model.state_dict()
    fine = config["fine_tuning"]
    sources = {
        item["path"]: load(item["path"])["model_state_dict"]
        for item in fine["pattern_imports"]
    }
    by_id = {item["new_id"]: item for item in fine["pattern_imports"]}
    shared = next(iter(sources.values()))
    for name, tensor in target.items():
        match = PATTERN_KEY.search(name)
        if match:
            item = by_id.get(int(match.group(1)))
            if item is None:
                continue
            old_name = (
                name[: match.start(1)] + str(item["old_id"]) + name[match.end(1) :]
            )
            value = sources[item["path"]].get(old_name)
        elif name.startswith(
            (
                "feature_model.mol_encoder.",
                "spectrum_model.mol_encoder.",
                "feature_model.reaction_instance_encoder.",
            )
        ):
            value = shared.get(name)
        else:
            continue  # Downstream predictors are independent, newly initialized modules.
        if value is None or value.shape != tensor.shape:
            raise ValueError(f"Incompatible imported feature tensor: {name}")
        tensor.copy_(value)
    model.load_state_dict(target, strict=True)


def prepare_pattern_models(chemistry, records):
    """Resolve chemistry and feature ownership before dataset preparation."""
    from fredis.domain.fragment.reaction import ReactionPatternSet

    chemistry = deepcopy(chemistry)
    original = Fragmenter.from_dict(chemistry["fragmenter_params"]).reaction_pattern_set
    definitions = {p.key: p.to_dict() for p in original.patterns}
    resolved = {}
    for record in records:
        record = deepcopy(record)
        record.pop(
            "architecture", None
        )  # New architectures belong to Training, never preparation.
        definition = record["pattern"]
        query = ReactionPatternSet.from_dict(
            {
                "name": "selected",
                "patterns": [
                    {k: v for k, v in definition.items() if k != "pattern_id"}
                ],
            }
        ).patterns[0]
        if record.get("source"):
            saved = load(record["source"])
            source_pattern = next(
                (
                    p
                    for p in entries(saved)
                    if p.pattern_id == record["source_pattern_id"]
                ),
                None,
            )
            if source_pattern is None or source_pattern.key != query.key:
                raise ValueError(
                    "Pretrained model does not match its reaction definition"
                )
            record["source"] = str(Path(record["source"]).resolve())
        if query.key in resolved:
            raise ValueError("Duplicate reaction model definition")
        definitions[query.key] = query.to_dict()
        resolved[query.key] = record
    union = ReactionPatternSet.from_dict(
        {
            "name": original.name,
            "patterns": [
                {k: v for k, v in value.items() if k != "pattern_id"}
                for value in definitions.values()
            ],
        }
    )
    chemistry["fragmenter_params"]["fragment_ion_tree_builder"][
        "reaction_pattern_set"
    ] = union.to_dict()
    chemistry["reaction_models"] = [
        {**resolved[p.key], "pattern": p.to_dict()}
        for p in union.patterns
        if p.key in resolved
    ]
    # Only pretrained dependencies are checked here; new modules are not built.
    imported = [r for r in chemistry["reaction_models"] if r.get("source")]
    if imported:
        first = load(imported[0]["source"])["model_config"]
        first = first
        from .sources import canonical_symbols

        if canonical_symbols(
            first["mol_encoder_params"]["symbols"]
        ) != canonical_symbols(chemistry["symbols"]):
            raise ValueError("Dataset symbols must match the pretrained source encoder")
        apply_dataset_pattern_models(
            {**chemistry, "mol_encoder_params": first["mol_encoder_params"]}, imported
        )
    return chemistry


def apply_dataset_pattern_models(config, records):
    config = deepcopy(config)
    patterns = Fragmenter.from_dict(
        config["fragmenter_params"]
    ).reaction_pattern_set.patterns
    by_key = {p.key: p for p in patterns}
    from fredis.domain.fragment.reaction import ReactionPatternSet

    imports = {}
    for record in records:
        definition = {k: v for k, v in record["pattern"].items() if k != "pattern_id"}
        query = ReactionPatternSet.from_dict(
            {"name": "selected", "patterns": [definition]}
        ).patterns[0]
        if query.key not in by_key:
            raise ValueError(
                "Dataset reaction model definition differs from dataset chemistry"
            )
        pid = by_key[query.key].pattern_id
        if record.get("source"):
            imports.setdefault(record["source"], []).append(record["source_pattern_id"])
    config.pop("reaction_models", None)
    return (
        assemble_config(
            config, [dict(path=path, pattern_ids=ids) for path, ids in imports.items()]
        )
        if imports
        else config
    )
