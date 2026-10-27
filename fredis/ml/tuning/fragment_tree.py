"""Run normal training CLI processes and retain all validation observations."""

import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

RESULT_NAME = "fragment-tree.tuning.pft"
PLAN_NAME = "fragment-tree.tuning-plan.pft"


def validation_history(directory):
    from ..training.fragment_tree_training.metric_store import (
        INDEX_NAME,
        iter_series_rows,
        read_scalar_events,
    )

    if (Path(directory) / INDEX_NAME).exists():
        history = [
            {key: value for key, value in event.items() if key != "event"}
            for event in read_scalar_events(directory)
            if event.get("event") == "epoch_end"
        ]
        indexed = {(row.get("epoch"), row.get("global_step")): row for row in history}
        for item in iter_series_rows(directory, "distributions/validation/"):
            try:
                row = indexed.get((float(item["epoch"]), float(item["global_step"])))
                value = float(item["value"])
            except (ValueError, KeyError):
                continue
            if row is not None and math.isfinite(value):
                row["validation_" + item["series"].split("/", 2)[2]] = value
        return history
    file = Path(directory) / "metrics.tsv"
    if not file.exists():
        return []
    history = []
    with file.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row.get("event") != "epoch_end":
                continue
            numeric = {}
            for key, value in row.items():
                try:
                    number = float(value)
                    if math.isfinite(number):
                        numeric[key] = number
                except (TypeError, ValueError):
                    pass
            history.append(numeric)
    # Distribution quartiles are measured across validation spectra at each epoch,
    # rather than across epoch means. Merge only observations of completed epochs.
    distributions = Path(directory) / "metric_distributions.tsv"
    if distributions.exists():
        indexed = {(row.get("epoch"), row.get("global_step")): row for row in history}
        with distributions.open() as stream:
            for item in csv.DictReader(stream, delimiter="\t"):
                if item.get("split") != "validation":
                    continue
                try:
                    row = indexed.get(
                        (float(item["epoch"]), float(item["global_step"]))
                    )
                    value = float(item["value"])
                except (ValueError, KeyError):
                    continue
                if row is not None and math.isfinite(value):
                    row["validation_" + item["metric"]] = value
    return history


def summarize_trial(trial):
    """Retain every comparison column, independent of any ranking preference."""
    history = trial.get("validationHistory", [])
    metrics = sorted(
        {key for row in history for key in row if key.startswith("validation")}
    )
    result = {}
    for metric in metrics:
        observations = [
            (row.get("epoch"), row[metric])
            for row in history
            if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])
        ]
        if not observations:
            continue
        maximum = max(observations, key=lambda pair: pair[1])
        summary = {"max": maximum[1], "maxEpoch": maximum[0]}
        # Last means the final completed epoch, never the last non-missing value.
        summary.update(last=history[-1].get(metric), lastEpoch=history[-1].get("epoch"))
        if "loss" in metric:
            minimum = min(observations, key=lambda pair: pair[1])
            summary.update(min=minimum[1], minEpoch=minimum[0])
        result[metric] = summary
    return result


def expand_search(spec):
    """Public CLI search format: normal CLI arguments plus flag candidate lists."""
    command = spec.get("trainingCommand", "fragment-tree")
    if command not in ("fragment-tree", "fragment-tree-detailed"):
        raise ValueError("trainingCommand must be fragment-tree or fragment-tree-detailed")
    base = spec["baseArgs"]
    search = spec["parameters"]
    if not isinstance(base, list) or not all(isinstance(v, str) for v in base):
        raise ValueError("baseArgs must be a list of ordinary training CLI arguments")
    if not search or any(
        not key.startswith("--") or not isinstance(values, list) or not values
        for key, values in search.items()
    ):
        raise ValueError("parameters must map CLI flags to non-empty candidate lists")
    trials = []
    for index, values in enumerate(itertools.product(*search.values()), 1):
        args = list(base)
        parameters = dict(zip(search, values))
        for flag, value in parameters.items():
            if flag in ("--output-dir", "--resume", "--overwrite"):
                raise ValueError("Output, resume and overwrite cannot be tuned")
            if flag in args:
                pos = args.index(flag)
                args[pos + 1] = str(value)
            else:
                args.extend([flag, str(value)])
        directory = str(Path(spec["outputDir"]) / f"trial-{index:03d}")
        if "--output-dir" in args:
            args[args.index("--output-dir") + 1] = directory
        else:
            args.extend(["--output-dir", directory])
        trials.append(
            {
                "config": {"outputDir": directory, "parameters": parameters},
                "args": ["-m", "fredis.cli", "train", command, *args],
            }
        )
    return {**spec, "trials": trials}


def tuning_configuration(plan):
    configuration = plan.get("configuration")
    if configuration is None and "modelConfig" in plan["trials"][0]["config"]:
        configuration = dict(plan["trials"][0]["config"])
        configuration["tuning"] = {"enabled": True, "parameters": {}}
        for trial in plan["trials"]:
            def collect(first, current, prefix=""):
                for key, value in current.items():
                    name = prefix + key
                    if isinstance(value, dict) and isinstance(first.get(key), dict):
                        collect(first[key], value, name + ".")
                    elif name != "outputDir" and value != first.get(key):
                        candidates = configuration["tuning"]["parameters"].setdefault(name, [first.get(key)])
                        if value not in candidates:
                            candidates.append(value)
            collect(plan["trials"][0]["config"], trial["config"])
        if configuration.get("architectureMode") == "shared":
            configuration["tuning"]["parameters"] = {
                key: value for key, value in configuration["tuning"]["parameters"].items()
                if not key.startswith(("modelConfig.", "patternArchitecture."))
            }
        if configuration.get("patternArchitecture", {}).get("mode") == "shared":
            configuration["tuning"]["parameters"] = {
                key: value for key, value in configuration["tuning"]["parameters"].items()
                if not key.startswith("modelConfig.reaction_model_params.pattern_model_params.")
            }
    if configuration is not None:
        configuration = dict(configuration, outputDir=plan["outputDir"], resume="", trainingMode="new")
    return configuration


def write_tuning_plan(output, plan):
    value = dict(plan, schema="fredis.tuning-plan", schemaVersion=1)
    configuration = tuning_configuration(plan)
    if configuration is not None:
        value["configuration"] = configuration
    temporary = output / (PLAN_NAME + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(output / PLAN_NAME)


def run(plan):
    # The lock is released by the OS even after an ungraceful orchestrator exit.
    import fcntl
    output = Path(plan["outputDir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".tuning.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another tuning process is already using this output directory") from error
        return _run(plan)


def _run(plan):
    if "trials" not in plan:
        plan = expand_search(plan)
    output = Path(plan["outputDir"])
    trials = plan["trials"]
    if not trials:
        raise ValueError("Supply at least one trial")
    directories = []
    for trial in trials:
        directory = Path(trial["config"]["outputDir"]).resolve()
        args = trial["args"]
        if (
            not isinstance(args, list)
            or not all(isinstance(a, str) for a in args)
            or len(args) < 4
            or args[:3] != ["-m", "fredis.cli", "train"]
            or args[3] not in ("fragment-tree", "fragment-tree-detailed")
        ):
            raise ValueError("Trials must invoke the normal fragment-tree training CLI")
        if (
            "--output-dir" not in args
            or Path(args[args.index("--output-dir") + 1]).resolve() != directory
        ):
            raise ValueError("Trial output argument does not match its configuration")
        if "--resume" in args or "--overwrite" in args:
            raise ValueError("Plan arguments cannot set resume or overwrite; checkpoint continuation is automatic")
        if (
            not directory.is_relative_to(output.resolve())
            or directory == output.resolve()
        ):
            raise ValueError("Trial outputs must be inside the search output directory")
        if directory in directories:
            raise ValueError(
                "Trial output is duplicated: " + str(directory)
            )
        directories.append(directory)
    output.mkdir(parents=True, exist_ok=True)
    report_file = output / RESULT_NAME
    from .resume import completed, continuation, plan_signature, read_progress, score_summaries
    prior, records, scores = read_progress(output, plan, RESULT_NAME, PLAN_NAME)
    # Preserve TSV-only rows before the first atomic publication of a resumed run.
    for index, row in scores.items():
        trial = trials[index - 1]
        record = records.setdefault(index, {
            "trial": index,
            "outputDir": str(Path(trial["config"]["outputDir"]).resolve().relative_to(output.resolve())),
            "config": trial["config"], "args": trial["args"],
            "status": row["status"],
            "exitCode": int(row["exitCode"]) if row.get("exitCode") else None,
            "validationHistory": [],
        })
        if not record.get("scoreSummaries"):
            record["scoreSummaries"] = score_summaries(row)
    results = [dict(record) for _, record in sorted(records.items())]
    report = {
        **prior,
        "schema": "fredis.tuning-result",
        "status": "running",
        "createdAt": prior.get("createdAt", datetime.now(timezone.utc).isoformat()),
        "planSignature": plan_signature(trials),
        "searchMethod": "grid",
        "totalTrials": len(trials),
        "trials": results,
    }

    def publish():
        for trial in results:
            if trial.get("validationHistory"):
                trial["scoreSummaries"] = summarize_trial(trial)
            else:
                trial.setdefault("scoreSummaries", {})
        report.update(updatedAt=datetime.now(timezone.utc).isoformat())
        columns = sorted(
            {
                f"{metric}.{stat}"
                for trial in results
                for metric, summary in trial["scoreSummaries"].items()
                for stat in summary
                if stat in ("max", "last", "min")
            }
        )
        report["scoreColumns"] = columns
        table = output / "tuning-scores.tsv.tmp"
        with table.open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["trial", "status", "exitCode", "config", *columns])
            for trial in results:
                values = {
                    f"{metric}.{stat}": value
                    for metric, summary in trial["scoreSummaries"].items()
                    for stat, value in summary.items()
                }
                writer.writerow(
                    [
                        trial["trial"],
                        trial["status"],
                        trial.get("exitCode", ""),
                        json.dumps(trial["config"]),
                        *[values.get(c) for c in columns],
                    ]
                )
        table.replace(output / "tuning-scores.tsv")
        temp = report_file.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(report_file)

    write_tuning_plan(output, plan)
    publish()
    active_record = None
    try:
        for index, trial in enumerate(trials, 1):
            directory = Path(trial["config"]["outputDir"])
            previous = records.get(index)
            record = previous if previous is not None else {
                "trial": index,
                "outputDir": str(directory.resolve().relative_to(output.resolve())),
                "config": trial["config"],
                "args": trial["args"],
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "validationHistory": [],
            }
            # Results retain their identity/order when a later trial was already done.
            results[:] = [item for item in results if item["trial"] != index] + [record]
            results.sort(key=lambda item: item["trial"])
            if completed(previous, scores.get(index), directory):
                history = validation_history(directory)
                if not record.get("scoreSummaries"):
                    record["scoreSummaries"] = score_summaries(scores.get(index))
                record.update(status="completed", exitCode=0,
                              validationHistory=history or record.get("validationHistory", []))
                print(f"Tuning trial {index}/{len(trials)}: already completed; skipping", flush=True)
                publish()
                continue
            active_record = record
            record.update(status="running")
            record.pop("error", None)
            record.pop("finishedAt", None)
            publish()
            print(f"Tuning trial {index}/{len(trials)}: {directory}", flush=True)
            try:
                argv, resume, already_finished = continuation(trial, directory)
                if resume:
                    record.setdefault("resumeHistory", []).append({
                        **resume, "resumedAt": datetime.now(timezone.utc).isoformat(),
                    })
                    print(f"Trial {index}: checkpoint epoch {resume['epoch']}; remaining {resume['remainingEpochs']} epochs", flush=True)
                if already_finished:
                    record.update(exitCode=0, status="completed")
                else:
                    record["executionArgs"] = argv
                    publish()
                    record["exitCode"] = subprocess.call([sys.executable, *argv])
                    record["status"] = "completed" if record["exitCode"] == 0 else "failed"
            except (OSError, ValueError) as error:
                record.update(exitCode=1, status="failed", error=str(error))
            record.update(
                validationHistory=validation_history(directory),
                finishedAt=datetime.now(timezone.utc).isoformat(),
            )
            publish()
            active_record = None
            print(f'Trial {index}: {record["status"]}', flush=True)
    except BaseException:
        if active_record is not None and active_record["status"] == "running":
            active_record.update(
                status="interrupted",
                validationHistory=validation_history(
                    trials[active_record["trial"] - 1]["config"]["outputDir"]
                ),
            )
        report["status"] = "interrupted"
        publish()
        raise
    report["status"] = (
        "completed" if all(r["status"] == "completed" and r.get("exitCode") == 0 for r in results) else "failed"
    )
    publish()
    print(f"Tuning results: {report_file}", flush=True)
    return 0 if report["status"] == "completed" else 1
