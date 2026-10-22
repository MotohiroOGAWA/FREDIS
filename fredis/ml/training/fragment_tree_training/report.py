"""Small live manifest used by the Workbench training-result viewer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from .configuration import CONFIG_NAME

REPORT_NAME = "result.training.pft"


def write_training_report(
    output_dir,
    *,
    status,
    settings=None,
    datasets=None,
    completed_epochs=0,
    global_step=0,
    latest_validation=None,
    error=None,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    project = (
        output.parent.parent.name
        if output.parent.name == "runs"
        else output.parent.name
    )
    payload = {
        "schema": "fredis.training-report",
        "schemaVersion": 1,
        "kind": "training-result",
        "project": project,
        "run": output.name,
        "status": status,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "completedEpochs": int(completed_epochs),
        "globalStep": int(global_step),
        "latestValidation": latest_validation,
        "settings": settings or {},
        "datasets": datasets or {},
        "artifacts": {
            "trainingConfiguration": CONFIG_NAME,
            "metrics": "metrics.tsv",
            "distributions": "metric_distributions.tsv",
            "intermediateValidation": "intermediate_validation.tsv",
            "validationDirectory": "spectrum_validation",
            "bestCheckpoint": "best.pt",
            "lastCheckpoint": "last.pt",
        },
    }
    if error is not None:
        payload["error"] = str(error)
    target = output / REPORT_NAME
    temporary = output / (REPORT_NAME + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    return target
