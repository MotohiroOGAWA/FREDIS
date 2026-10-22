"""Clickable, reproducible training settings shared with the Workbench."""
from copy import deepcopy
import json
from pathlib import Path

CONFIG_NAME = "fragment-tree.training-config.pft"


def training_configuration(output_dir, model_config, settings):
    def camel(key):
        first, *rest = key.split("_")
        return first + "".join(word.title() for word in rest)

    value = {camel(key): item for key, item in settings.items()}
    train = Path(settings["train_dir"]).resolve()
    value.update(
        schema="fredis.training-config",
        schemaVersion=1,
        application="fragment-tree-training",
        datasetDir=str(train.parent),
        outputDir=str(Path(output_dir).resolve()),
        modelConfig=deepcopy(model_config),
        dropout=model_config.get("dropout", 0.0),
        resume=str(settings["resume"]) if settings.get("resume") else "",
        trainingMode="resume" if settings.get("resume") else "new",
        patternArchitecture={
            "mode": "individual",
            "shared": {},
            "frozenPatternIds": list(model_config.get("fine_tuning", {}).get("frozen_pattern_ids", [])),
        },
    )
    return value


def write_training_configuration(output_dir, model_config, settings):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / CONFIG_NAME
    temporary = output / (CONFIG_NAME + ".tmp")
    temporary.write_text(json.dumps(training_configuration(output, model_config, settings), indent=2) + "\n")
    temporary.replace(target)
    return target
