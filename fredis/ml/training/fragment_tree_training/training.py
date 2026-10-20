"""Prepared-tensor fragment-tree branch and physical-ion training entry point."""

from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
import torch
from tqdm.auto import tqdm
from fredis.ml.common.torch_utils.early_stopping import EarlyStopping
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.fragment_spectrum_model import SpectrumLossSettings
from fredis.cli.base import output_directory
from .model import ReactionFragmentTreeTrainingModel
from .mol_encoder_options import (
    MOL_ENCODER_OPTIONS as SHARED_MOL_ENCODER_OPTIONS,
    add_mol_encoder_arguments,
)


def confirm_output_overwrite(
    output: Path, *, overwrite: bool, resume, protected_paths=()
) -> None:
    """A fresh run into a non-empty output directory used to silently mix in
    a previous run's leftovers: metric_distributions.tsv is appended to (not
    replaced), and spectrum_validation/*.json from a longer or differently
    configured previous run is never cleaned up on its own. So confirming an
    overwrite here also clears the directory (mirroring the data-preparation
    CLI's --overwrite), guarded by the same protected-paths check so it can
    never delete an input dataset or checkpoint accidentally pointed at the
    output directory. Resuming into that same directory is the normal,
    intentional case, is never asked about, and never clears anything."""
    if resume:
        return
    if not output.exists() or not any(output.iterdir()):
        return
    if not overwrite:
        if sys.stdin.isatty():
            answer = input(
                f"Output directory {output} already exists and is not empty. Overwrite it? [y/N] "
            )
            if answer.strip().lower() not in ("y", "yes"):
                raise SystemExit(
                    "Aborted: pass --overwrite to reuse a non-empty output directory without asking."
                )
        else:
            raise SystemExit(
                f"Output directory {output} already exists and is not empty. "
                "Pass --overwrite to reuse it, or choose an empty directory. "
                "(No interactive terminal is attached, so confirmation cannot be prompted for.)"
            )
    resolved = output.resolve()
    for protected in (
        Path(__file__).resolve(),
        *(Path(value).resolve() for value in protected_paths if value),
    ):
        if protected.is_relative_to(resolved):
            raise ValueError(
                f"Output directory contains {protected}; choose a separate output directory."
            )
    shutil.rmtree(resolved)


def _sample_meets_assignment_score(
    sample, minimum_assignment_score, minimum_assignment_score_without_precursor
):
    """A score of None means it could not be computed for that sample (e.g. zero
    total intensity), so it never fails a threshold it can't be checked against."""
    full = sample["assignmentScore"]
    without = sample["assignmentScoreWithoutPrecursor"]
    return (full is None or full >= minimum_assignment_score) and (
        without is None or without >= minimum_assignment_score_without_precursor
    )


PROGRESS_WIDTH = 100
PROGRESS_FORMAT = "{desc:<18} {percentage:3.0f}%|{bar:36}| {n_fmt:>5}/{total_fmt:<5} [{elapsed}<{remaining}]"


def progress_bar(*, total, description, position, leave):
    """Create one of the four fixed-position training progress bars."""
    return tqdm(
        total=total,
        desc=description,
        position=position,
        leave=leave,
        ncols=PROGRESS_WIDTH,
        dynamic_ncols=False,
        bar_format=PROGRESS_FORMAT,
        mininterval=0.1,
        maxinterval=1.0,
    )


def _write_metric_artifacts(output: Path, report: dict) -> None:
    """Atomically publish one live TSV containing step and epoch rows."""
    events = [
        {"event": "train_step", **item} for item in report.get("train_step_history", [])
    ]
    events.extend({"event": "epoch_end", **item} for item in report.get("history", []))
    events.sort(
        key=lambda item: (
            int(item.get("global_step", 0)),
            item["event"] != "train_step",
        )
    )
    headers = list(dict.fromkeys(key for item in events for key in item)) or ["event"]
    text = "\t".join(key.replace("/", "_") for key in headers) + "\n"
    text += "\n".join(
        "\t".join(str(item.get(key, "")) for key in headers) for item in events
    )
    if events:
        text += "\n"
    temporary = output / "metrics.tsv.tmp"
    temporary.write_text(text)
    temporary.replace(output / "metrics.tsv")


def _read_metric_events(path: Path) -> list[dict]:
    """Read complete TSV rows, tolerating an interrupted live final row."""
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    if not lines:
        return []
    headers = lines[0].split("\t")
    events = []
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(headers):
            continue
        row = {}
        for key, value in zip(headers, cells):
            if key == "event":
                row[key] = value
                continue
            if not value:
                continue
            try:
                number = float(value)
                row[key] = int(number) if number.is_integer() else number
            except ValueError:
                row[key] = value
        events.append(row)
    return events


def _mean_training_interval(
    *,
    totals: dict,
    batch_metrics: dict,
    samples: int,
    batches: int,
    epoch: int,
    global_step: int,
    elapsed_seconds: float,
) -> dict:
    """Build one interval row: sample means for losses, batch means for optimizer state."""
    row = {
        "epoch": epoch,
        "global_step": global_step,
        "interval_steps": batches,
        "train/samples": samples,
        "train/batches": batches,
        "interval_seconds": elapsed_seconds,
        "train_loss": totals["loss"] / max(samples, 1),
    }
    row.update(
        {
            "train/" + key: value / max(samples, 1)
            for key, value in totals.items()
            if key != "loss"
        }
    )
    for key in ("gradient_norm", "learning_rate"):
        values = batch_metrics.get(key, [])
        row[key] = sum(values) / max(len(values), 1)
    return row


def validation_subset(pieces, fraction: float, seed: int):
    """Select a fixed, uniformly sampled subset of measurement records."""
    import random
    from ...input.reaction_batching import select_samples

    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("validation_fraction must be greater than 0 and at most 1")
    total = (
        pieces.total_samples
        if hasattr(pieces, "total_samples")
        else sum(piece.num_samples for piece in pieces)
    )
    if not total:
        raise ValueError("Validation dataset contains no samples")
    indices = sorted(
        random.Random(seed).sample(range(total), max(1, math.ceil(total * fraction)))
    )
    if hasattr(pieces, "sample_subset"):
        return pieces.sample_subset(indices), indices
    selected = []
    offset = cursor = 0
    for piece in pieces:
        end = offset + piece.num_samples
        start = cursor
        while cursor < len(indices) and indices[cursor] < end:
            cursor += 1
        if cursor > start:
            selected.append(
                select_samples(
                    piece, [index - offset for index in indices[start:cursor]]
                )
            )
        offset = end
    return selected, indices


def _validate_training_settings(settings: dict) -> SpectrumLossSettings:
    """Validate optimizer, loss, filtering and validation options before CUDA setup."""
    if (
        settings["epochs"] < 1
        or settings["batch_size"] < 1
        or settings["max_samples"] < 1
        or not math.isfinite(settings["lr"])
        or settings["lr"] <= 0
    ):
        raise ValueError("epochs, batch_size, max_samples and lr must be positive")
    spectrum_loss = SpectrumLossSettings(
        **{key: settings[key] for key in SPECTRUM_LOSS_KEYS}
    )
    if (
        any(
            not math.isfinite(value) or value < 0
            for value in (
                settings["weight_decay"],
                settings["gradient_clip"],
                settings["branch_weight"],
                settings["negative_weight"],
                settings["min_lr"],
            )
        )
        or not math.isfinite(settings["branch_mil_temperature"])
        or settings["branch_mil_temperature"] <= 0
    ):
        raise ValueError("Optimizer and loss settings must be finite and non-negative")
    if settings["branch_weight"] + spectrum_loss.intensity_loss_weight <= 0:
        raise ValueError("Enable at least one training loss")
    if (
        settings["min_lr"] > settings["lr"]
        or min(
            settings["warmup_steps"],
            settings["lr_patience"],
            settings["early_stopping_patience"],
        )
        < 0
    ):
        raise ValueError(
            "min_lr must not exceed lr; schedule patience/steps must be non-negative"
        )
    if (
        not isinstance(settings["validation_interval_steps"], int)
        or settings["validation_interval_steps"] < 0
    ):
        raise ValueError("validation_interval_steps must be a non-negative integer")
    if (
        not isinstance(settings["train_log_interval_steps"], int)
        or settings["train_log_interval_steps"] < 0
    ):
        raise ValueError("train_log_interval_steps must be a non-negative integer")
    if (
        not math.isfinite(settings["validation_fraction"])
        or not 0 < settings["validation_fraction"] <= 1
    ):
        raise ValueError("validation_fraction must be greater than 0 and at most 1")
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in (
            settings["minimum_assignment_score"],
            settings["minimum_assignment_score_without_precursor"],
        )
    ):
        raise ValueError("Assignment score thresholds must be between 0 and 1")
    if settings["initialize_from"] is not None and settings["resume"] is not None:
        raise ValueError("Use initialize_from or resume, not both.")
    return spectrum_loss


def _load_prepared_datasets(
    train_dir,
    val_dir,
    *,
    max_samples,
    minimum_assignment_score,
    minimum_assignment_score_without_precursor,
    cache_dataset_in_memory,
):
    """Index training and validation data without loading their tensor payloads."""
    from .prepared_dataset import prepared_piece_sequence

    datasets = []
    dataset_report = {}
    for name, directory in (("train", train_dir), ("validation", val_dir)):
        pieces = prepared_piece_sequence(
            directory,
            max_samples=max_samples,
            minimum_assignment_score=minimum_assignment_score,
            minimum_assignment_score_without_precursor=minimum_assignment_score_without_precursor,
            cache_in_memory=cache_dataset_in_memory,
        )
        stats = pieces.statistics
        datasets.append(pieces)
        dataset_report[name] = {
            "files": int(stats.get("num_sources", pieces.file_count)),
            "raw_records": stats.get("num_metadata_valid_records"),
            "samples": pieces.total_unfiltered_samples,
            "filtered_samples": pieces.total_samples,
            "teacher_states": stats.get("num_teacher_nodes"),
            "formula_targets": stats.get("num_physical_ion_candidates"),
            "chunks": len(pieces),
            "loading_mode": (
                "cache_on_first_use" if cache_dataset_in_memory else "streaming"
            ),
        }
    return datasets, dataset_report


def train_reactions(
    *,
    model_config: dict,
    train_dir: str | Path,
    val_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
    resume: str | Path | None = None,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    branch_weight: float = 1.0,
    negative_weight: float = 0.2,
    branch_mil_temperature: float = 0.1,
    intensity_loss_weight: float = 1.0,
    cosine_loss_weight: float = 0.5,
    max_samples: int = 128,
    seed: int = 42,
    warmup_steps: int = 100,
    lr_patience: int = 3,
    early_stopping_patience: int = 10,
    min_lr: float = 1e-6,
    train_log_interval_steps: int = 100,
    validation_interval_steps: int = 0,
    validation_fraction: float = 0.1,
    train_mol_encoder: bool = True,
    cache_dataset_in_memory: bool = False,
    minimum_assignment_score: float = 0.0,
    minimum_assignment_score_without_precursor: float = 0.0,
    initialize_from: str | Path | None = None,
) -> dict:
    from dataclasses import replace
    import random
    import time
    from .sources import inherit_model_config
    from ...input.reaction_batching import select_samples
    from ...specgen.fragment_spectrum_model import deduplicate_molecular_graphs
    from torch.utils.tensorboard import SummaryWriter

    model_config = dict(model_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        key: value
        for key, value in locals().copy().items()
        if key
        in (
            "train_dir",
            "val_dir",
            "epochs",
            "batch_size",
            "lr",
            "device",
            "resume",
            "weight_decay",
            "gradient_clip",
            "branch_weight",
            "negative_weight",
            "branch_mil_temperature",
            *SPECTRUM_LOSS_KEYS,
            "max_samples",
            "seed",
            "warmup_steps",
            "lr_patience",
            "early_stopping_patience",
            "min_lr",
            "train_mol_encoder",
            "cache_dataset_in_memory",
            "initialize_from",
            "train_log_interval_steps",
            "validation_interval_steps",
            "validation_fraction",
            "minimum_assignment_score",
            "minimum_assignment_score_without_precursor",
        )
    }
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in settings.items()
    }
    (output / "training_args.json").write_text(json.dumps(settings, indent=2))
    from .report import write_training_report

    write_training_report(output, status="initializing", settings=settings)
    spectrum_loss = _validate_training_settings(settings)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "Fragment Tree Training requires CUDA; CPU training is not supported"
        )
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    startup_progress = progress_bar(
        total=5, description="Startup", position=0, leave=True
    )
    checkpoint = (
        torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    )
    if checkpoint and "optimizer_state_dict" not in checkpoint:
        raise ValueError(
            "This checkpoint has no optimizer state; train it with --fine-tune-checkpoint and --head-only instead of --resume"
        )
    initialize_weights = (
        torch.load(initialize_from, map_location="cpu", weights_only=False)
        if initialize_from
        else None
    )
    startup_progress.update(1)
    subset_seed = (
        checkpoint["training_settings"]["validation_subset_seed"]
        if checkpoint
        else seed
    )
    settings["validation_subset_seed"] = subset_seed
    if checkpoint:
        if model_config.get("fine_tuning") and model_config.get(
            "fine_tuning"
        ) != checkpoint["model_config"].get("fine_tuning"):
            raise ValueError("Resume fine-tuning configuration differs from checkpoint")
        model_config = checkpoint["model_config"]
        prior = checkpoint["training_settings"]
        # Restoring optimizer and loss definitions makes a resumed curve comparable.
        for key in (
            "branch_weight",
            "negative_weight",
            "branch_mil_temperature",
            "train_mol_encoder",
            "gradient_clip",
            "warmup_steps",
            "lr_patience",
            "min_lr",
            "lr",
        ):
            settings[key] = prior[key]
        branch_weight = settings["branch_weight"]
        negative_weight = settings["negative_weight"]
        branch_mil_temperature = settings["branch_mil_temperature"]
        for key in SPECTRUM_LOSS_KEYS:
            settings[key] = prior[key]
        spectrum_loss = SpectrumLossSettings(
            **{key: settings[key] for key in SPECTRUM_LOSS_KEYS}
        )
        train_mol_encoder = settings["train_mol_encoder"]
        gradient_clip = settings["gradient_clip"]
        warmup_steps = settings["warmup_steps"]
        lr_patience = settings["lr_patience"]
        min_lr = settings["min_lr"]
        lr = settings["lr"]
    # Build lightweight sample/chunk references from preparation sidecars.
    # Tensor payloads remain unopened until their first training/validation use.
    startup_progress.set_description_str("Dataset index", refresh=True)
    datasets, dataset_report = _load_prepared_datasets(
        train_dir,
        val_dir,
        max_samples=max_samples,
        minimum_assignment_score=minimum_assignment_score,
        minimum_assignment_score_without_precursor=minimum_assignment_score_without_precursor,
        cache_dataset_in_memory=cache_dataset_in_memory,
    )
    startup_progress.update(1)
    startup_progress.set_description_str("Configuration", refresh=True)
    params = model_config
    model_config = inherit_model_config(
        model_config,
        train_dir,
        val_dir,
        saved_model=model_config if resume or params.get("fine_tuning") else None,
    )
    from .pattern_models import pattern_config
    from fredis.domain.fragment import Fragmenter

    reaction_params = model_config.setdefault("reaction_model_params", {})
    reaction_params["pattern_model_params"] = {
        str(p.pattern_id): pattern_config(model_config, p.pattern_id)
        for p in Fragmenter.from_dict(
            model_config["fragmenter_params"]
        ).reaction_pattern_set.patterns
    }
    model_config["max_samples"] = max_samples
    (output / "training_config.json").write_text(
        json.dumps({"model_config": model_config, "training": settings}, indent=2)
    )
    startup_progress.update(1)
    startup_progress.set_description_str("CUDA model", refresh=True)
    params = dict(model_config)
    generator = create_spectrum_generator(params).to(device)
    if not train_mol_encoder:
        generator.feature_model.freeze_mol_encoder()
    model = ReactionFragmentTreeTrainingModel(
        generator.feature_model,
        spectrum_model=generator.spectrum_model,
        branch_weight=branch_weight,
        negative_weight=negative_weight,
        branch_mil_temperature=branch_mil_temperature,
        spectrum_loss=spectrum_loss,
    ).to(device)
    model.set_checkpoint_model_config(model_config)
    startup_progress.update(1)
    startup_progress.set_description_str("Optimizer", refresh=True)
    optimizer_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=min_lr
    )
    start = 0
    global_step = 0
    best = float("inf")
    bad_epochs = 0
    history = []
    intermediate_history = []
    train_step_history = []
    if checkpoint:
        if checkpoint.get("architecture") != generator.architecture:
            raise ValueError("Incompatible checkpoint architecture")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step", 0))
        best = float(checkpoint.get("best_validation_loss", best))
        history = checkpoint.get("history", [])
        intermediate_history = checkpoint.get("intermediate_validation_history", [])
        train_step_history = checkpoint.get("train_step_history", [])
        prior_events = [
            item
            for item in _read_metric_events(Path(resume).parent / "metrics.tsv")
            if int(item.get("global_step", 0)) <= global_step
        ]
        if not history:
            history = [
                {key: value for key, value in item.items() if key != "event"}
                for item in prior_events
                if item.get("event") == "epoch_end"
            ]
        if not train_step_history:
            train_step_history = [
                {key: value for key, value in item.items() if key != "event"}
                for item in prior_events
                if item.get("event") == "train_step"
            ]
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
    elif initialize_from:
        if params.get("fine_tuning"):
            raise ValueError(
                "Weight initialization cannot be combined with frozen-base expansion"
            )
        model.load_state_dict(initialize_weights["model_state_dict"])
    elif params.get("fine_tuning"):
        from .fine_tuning import initialize_from_base

        initialize_from_base(model, params)
    from .fine_tuning import parameter_report

    (output / "trainable_parameters.json").write_text(
        json.dumps(parameter_report(model), indent=2)
    )
    startup_progress.update(1)
    startup_progress.set_description_str("Ready", refresh=True)
    startup_progress.close()
    # Chemistry ends here; CPU tensor collation can continue per batch.
    # Forward/backward, validation and compatibility masks use CUDA tensors.
    (output / "dataset_summary.json").write_text(json.dumps(dataset_report, indent=2))
    write_training_report(
        output,
        status="running",
        settings=settings,
        datasets=dataset_report,
        completed_epochs=len(history),
        global_step=global_step,
    )
    intermediate_pieces = []
    if validation_interval_steps:
        intermediate_pieces, subset_indices = validation_subset(
            datasets[1], validation_fraction, subset_seed
        )
        (output / "intermediate_validation_subset.json").write_text(
            json.dumps(
                {
                    "seed": subset_seed,
                    "fraction": validation_fraction,
                    "total_samples": dataset_report["validation"]["samples"],
                    "sample_indices": subset_indices,
                },
                indent=2,
            )
        )
    parameters = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    writer = SummaryWriter(
        str(output / "tensorboard"), purge_step=start + 1 if resume else None
    )
    iteration_writer = SummaryWriter(
        str(output / "tensorboard" / "iterations"),
        purge_step=global_step + 1 if resume else None,
    )
    writer.add_text("configuration/model", json.dumps(model_config, indent=2))
    writer.add_text("configuration/training", json.dumps(settings, indent=2))
    started = time.perf_counter()
    stop_reason = "epochs_completed"

    def metric_report(reason="running"):
        return {
            "history": history,
            "train_step_history": train_step_history,
            "intermediate_validation_history": intermediate_history,
            "architecture": generator.architecture,
            "model_config": model_config,
            "training": settings,
            "datasets": dataset_report,
            "parameters": parameters,
            "best_validation_loss": best,
            "elapsed_seconds": time.perf_counter() - started,
            "validation_mode": "free-running spectrum generation plus teacher-forced loss; original-peak cosine distributions recorded",
            "stop_reason": reason,
        }

    def publish_metrics(reason="running"):
        report = metric_report(reason)
        _write_metric_artifacts(output, report)
        return report

    def batch_groups(pieces, shuffle):
        order = (
            pieces.ordered_indices(shuffle)
            if hasattr(pieces, "ordered_indices")
            else list(range(len(pieces)))
        )
        if shuffle and not hasattr(pieces, "ordered_indices"):
            random.shuffle(order)
        groups = []
        group = []
        count = 0
        for index in order:
            piece_samples = (
                pieces.sample_count(index)
                if hasattr(pieces, "sample_count")
                else pieces[index].num_samples
            )
            if group and (
                len(group) >= batch_size or count + piece_samples > max_samples
            ):
                groups.append(tuple(group))
                group = []
                count = 0
            group.append(index)
            count += piece_samples
        if group:
            groups.append(tuple(group))
        return groups

    def grouped_batches(pieces, groups):
        for group in groups:
            yield SourceReactionStructure.from_structures(
                [pieces[index] for index in group]
            )

    def metric_values(result):
        return {"loss": result.loss, **result.metrics}

    from .spectrum_validation import validate_spectra, append_metric_distributions

    train_window_totals = {}
    train_window_batch_metrics = {}
    train_window_samples = 0
    train_window_batches = 0
    train_window_started = time.perf_counter()

    def publish_train_window(epoch):
        nonlocal train_window_totals, train_window_batch_metrics, train_window_samples, train_window_batches, train_window_started
        row = _mean_training_interval(
            totals=train_window_totals,
            batch_metrics=train_window_batch_metrics,
            samples=train_window_samples,
            batches=train_window_batches,
            epoch=epoch,
            global_step=global_step,
            elapsed_seconds=time.perf_counter() - train_window_started,
        )
        train_step_history.append(row)
        append_metric_distributions(
            output, global_step, "train_window", train_window_batch_metrics, epoch=epoch
        )
        for key, value in row.items():
            if key not in ("epoch", "global_step"):
                iteration_writer.add_scalar("train_interval/" + key, value, global_step)
        iteration_writer.flush()
        publish_metrics()
        write_training_report(
            output,
            status="running",
            settings=settings,
            datasets=dataset_report,
            completed_epochs=len(history),
            global_step=global_step,
        )
        train_window_totals = {}
        train_window_batch_metrics = {}
        train_window_samples = 0
        train_window_batches = 0
        train_window_started = time.perf_counter()

    def intermediate_validation(epoch):
        was_training = model.training
        totals = {}
        sample_count = batch_count = 0
        batch_metrics = {}
        validation_started = time.perf_counter()
        groups = batch_groups(intermediate_pieces, False)
        stage_progress.reset(total=len(groups) + len(intermediate_pieces))
        stage_progress.set_description_str("Stage check")
        stage_progress.set_postfix_str("validation loss", refresh=True)
        try:
            model.eval()
            with torch.no_grad():
                for data in grouped_batches(intermediate_pieces, groups):
                    data = replace(
                        data,
                        spectrum_batch=deduplicate_molecular_graphs(
                            data.spectrum_batch, data.source_smiles
                        ),
                    ).to(device)
                    result = model(data)
                    for key, value in metric_values(result).items():
                        scalar = float(value.detach())
                        if not math.isfinite(scalar):
                            raise FloatingPointError(
                                f"Non-finite intermediate validation metric {key}"
                            )
                        totals[key] = totals.get(key, 0.0) + scalar * data.num_samples
                        batch_metrics.setdefault(key, []).append(scalar)
                    sample_count += data.num_samples
                    batch_count += 1
                    del result, data, value
                    stage_progress.update(1)
        finally:
            model.train(was_training)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "validation_samples": sample_count,
            "total_validation_samples": dataset_report["validation"]["samples"],
            "validation_fraction": validation_fraction,
            "validation_batches": batch_count,
            "validation_seconds": time.perf_counter() - validation_started,
            "validation_loss": totals.pop("loss") / sample_count,
        }
        row.update(
            {"validation/" + key: value / sample_count for key, value in totals.items()}
        )
        append_metric_distributions(
            output, global_step, "intermediate_validation", batch_metrics, epoch=epoch
        )
        stage_progress.set_postfix_str("generate spectrum", refresh=True)
        row.update(
            {
                "validation/" + k: v
                for k, v in validate_spectra(
                    generator,
                    intermediate_pieces,
                    output,
                    f"step_{global_step}",
                    global_step=global_step,
                    progress=lambda _, phase, *rest: (
                        stage_progress.update(1) if phase == "end" else None
                    ),
                    epoch=epoch,
                ).items()
            }
        )
        row["validation_seconds"] = time.perf_counter() - validation_started
        intermediate_history.append(row)
        headers = list(
            dict.fromkeys(key for item in intermediate_history for key in item)
        )
        temporary = output / "intermediate_validation.tsv.tmp"
        temporary.write_text(
            "\t".join(headers)
            + "\n"
            + "\n".join(
                "\t".join(str(item.get(key, "")) for key in headers)
                for item in intermediate_history
            )
            + "\n"
        )
        temporary.replace(output / "intermediate_validation.tsv")
        for key, value in row.items():
            if key not in ("epoch", "global_step"):
                iteration_writer.add_scalar(
                    "intermediate_validation/" + key, value, global_step
                )
        iteration_writer.flush()

    try:
        epoch_progress = progress_bar(
            total=epochs, description="Epoch", position=0, leave=True
        )
        early_stopping = EarlyStopping(
            early_stopping_patience, best=best, bad_epochs=bad_epochs
        )
        for epoch in range(start, start + epochs):
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            row = {"epoch": epoch + 1}
            gradient_values = []
            epoch_groups = [
                batch_groups(datasets[0], True),
                batch_groups(datasets[1], False),
            ]
            # Train and validation each get their own 0->100% bar (validation's
            # includes the free-running spectrum-generation pass below), so
            # neither one looks stalled while the other is what's actually running.
            iteration_progress = {
                "train": progress_bar(
                    total=len(epoch_groups[0]),
                    description="Iteration(Train)",
                    position=1,
                    leave=False,
                ),
                "validation": progress_bar(
                    total=len(epoch_groups[1]) + len(datasets[1]),
                    description="Iteration(Validation)",
                    position=2,
                    leave=False,
                ),
            }
            stage_progress = progress_bar(
                total=1, description="Stage", position=3, leave=False
            )
            for name, pieces, groups in zip(
                ("train", "validation"), datasets, epoch_groups
            ):
                model.train(name == "train")
                totals = {}
                sample_count = 0
                batch_count = 0
                batch_metrics = {}
                with torch.set_grad_enabled(name == "train"):
                    for data in grouped_batches(pieces, groups):
                        stages = (
                            ("prepare", "forward", "backward", "optimizer", "metrics")
                            if name == "train"
                            else ("prepare", "forward", "metrics")
                        )
                        stage_progress.reset(total=len(stages))
                        stage_progress.set_description_str(f"Stage {name}")
                        stage_progress.set_postfix_str(stages[0], refresh=True)
                        # Repeated molecular graphs (including Source roots) are
                        # encoded once and gathered for each sample's tree.
                        source_keys = data.source_smiles
                        data = replace(
                            data,
                            spectrum_batch=deduplicate_molecular_graphs(
                                data.spectrum_batch, source_keys
                            ),
                        ).to(device)
                        stage_progress.update(1)
                        stage_progress.set_postfix_str("forward", refresh=True)
                        if name == "train":
                            if global_step < warmup_steps:
                                for group in optimizer.param_groups:
                                    group["lr"] = (
                                        lr * (global_step + 1) / max(warmup_steps, 1)
                                    )
                            optimizer.zero_grad(set_to_none=True)
                        result = model(data)
                        stage_progress.update(1)
                        if not torch.isfinite(result.loss):
                            raise FloatingPointError(
                                f"Non-finite {name} loss at epoch {epoch+1}"
                            )
                        if name == "train":
                            stage_progress.set_postfix_str("backward", refresh=True)
                            result.loss.backward()
                            norm = torch.nn.utils.clip_grad_norm_(
                                (p for p in model.parameters() if p.requires_grad),
                                gradient_clip if gradient_clip > 0 else float("inf"),
                                error_if_nonfinite=True,
                            )
                            stage_progress.update(1)
                            stage_progress.set_postfix_str("optimizer", refresh=True)
                            gradient_values.append(float(norm))
                            optimizer.step()
                            global_step += 1
                            batch_metrics.setdefault("gradient_norm", []).append(
                                float(norm)
                            )
                            batch_metrics.setdefault("learning_rate", []).append(
                                float(optimizer.param_groups[0]["lr"])
                            )
                            if train_log_interval_steps:
                                train_window_batch_metrics.setdefault(
                                    "gradient_norm", []
                                ).append(float(norm))
                                train_window_batch_metrics.setdefault(
                                    "learning_rate", []
                                ).append(float(optimizer.param_groups[0]["lr"]))
                            iteration_writer.add_scalar(
                                "batch/train_loss",
                                float(result.loss.detach()),
                                global_step,
                            )
                            iteration_writer.add_scalar(
                                "batch/gradient_norm", float(norm), global_step
                            )
                            iteration_writer.add_scalar(
                                "batch/learning_rate",
                                optimizer.param_groups[0]["lr"],
                                global_step,
                            )
                            stage_progress.update(1)
                        stage_progress.set_postfix_str("metrics", refresh=True)
                        values = metric_values(result)
                        live = {}
                        for key, value in values.items():
                            scalar = float(value.detach())
                            if not math.isfinite(scalar):
                                raise FloatingPointError(
                                    f"Non-finite metric {name}/{key}"
                                )
                            totals[key] = (
                                totals.get(key, 0.0) + scalar * data.num_samples
                            )
                            batch_metrics.setdefault(key, []).append(scalar)
                            if name == "train" and train_log_interval_steps:
                                train_window_totals[key] = (
                                    train_window_totals.get(key, 0.0)
                                    + scalar * data.num_samples
                                )
                                train_window_batch_metrics.setdefault(key, []).append(
                                    scalar
                                )
                            # A quick read on how training is actually going, right on
                            # the bar; the full picture still lives in the report.
                            if key == "loss":
                                live["loss"] = f"{scalar:.4g}"
                            elif key == "full_spectrum_cosine":
                                live["cos"] = f"{scalar:.3f}"
                        iteration_progress[name].set_postfix(live, refresh=False)
                        sample_count += data.num_samples
                        batch_count += 1
                        if name == "train" and train_log_interval_steps:
                            train_window_samples += data.num_samples
                            train_window_batches += 1
                        # Release the training graph before evaluating a subset.
                        del values, result, data, value
                        stage_progress.update(1)
                        iteration_progress[name].update(1)
                        if (
                            name == "train"
                            and train_log_interval_steps
                            and train_window_batches >= train_log_interval_steps
                        ):
                            publish_train_window(epoch + 1)
                        if (
                            name == "train"
                            and validation_interval_steps
                            and global_step % validation_interval_steps == 0
                        ):
                            intermediate_validation(epoch + 1)
                row[name + "_loss"] = totals.pop("loss") / max(sample_count, 1)
                row.update(
                    {
                        name + "/" + key: value / max(sample_count, 1)
                        for key, value in totals.items()
                    }
                )
                row[name + "/samples"] = sample_count
                row[name + "/batches"] = batch_count
                append_metric_distributions(
                    output, global_step, name, batch_metrics, epoch=epoch + 1
                )
            cosine_so_far = []

            def spectrum_progress(_, phase, mean_cosine=None):
                if phase == "start":
                    stage_progress.reset(total=1)
                    stage_progress.set_description_str("Stage inference")
                    stage_progress.set_postfix_str("generate spectrum", refresh=True)
                else:
                    stage_progress.update(1)
                    iteration_progress["validation"].update(1)
                    if mean_cosine is not None:
                        cosine_so_far.append(mean_cosine)
                        iteration_progress["validation"].set_postfix(
                            cos=f"{sum(cosine_so_far)/len(cosine_so_far):.3f}",
                            refresh=False,
                        )

            row.update(
                {
                    "validation/" + k: v
                    for k, v in validate_spectra(
                        generator,
                        datasets[1],
                        output,
                        f"epoch_{epoch+1}",
                        global_step=global_step,
                        progress=spectrum_progress,
                        epoch=epoch + 1,
                    ).items()
                }
            )
            stage_progress.close()
            iteration_progress["train"].close()
            iteration_progress["validation"].close()
            row["global_step"] = global_step
            row["learning_rate"] = optimizer.param_groups[0]["lr"]
            row["gradient_norm"] = sum(gradient_values) / max(len(gradient_values), 1)
            row["epoch_seconds"] = time.perf_counter() - epoch_start
            row["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            improved = early_stopping.update(row["validation_loss"])
            best = early_stopping.best
            bad_epochs = early_stopping.bad_epochs
            if global_step >= warmup_steps:
                scheduler.step(row["validation_loss"])
            history.append(row)
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch + 1)
            writer.flush()
            iteration_writer.flush()
            report = publish_metrics()
            saved = dict(
                model_config=model_config,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch + 1,
                global_step=global_step,
                intermediate_validation_history=intermediate_history,
                train_step_history=train_step_history,
                architecture=generator.architecture,
                history=history,
                training_settings=settings,
                best_validation_loss=best,
                bad_epochs=bad_epochs,
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all(),
                python_rng_state=random.getstate(),
            )
            torch.save(saved, output / "last.pt")
            if improved:
                torch.save(saved, output / "best.pt")
            write_training_report(
                output,
                status="running",
                settings=settings,
                datasets=dataset_report,
                completed_epochs=len(history),
                global_step=global_step,
                latest_validation=f"spectrum_validation/epoch_{epoch+1}.json",
            )
            epoch_progress.update(1)
            if early_stopping.should_stop:
                stop_reason = "early_stopping"
                break
        (output / "training_failure.json").unlink(missing_ok=True)
        report = publish_metrics(stop_reason)
        (output / "training_report.json").write_text(json.dumps(report, indent=2))
        write_training_report(
            output,
            status="completed",
            settings=settings,
            datasets=dataset_report,
            completed_epochs=len(history),
            global_step=global_step,
            latest_validation=(
                f'spectrum_validation/epoch_{history[-1]["epoch"]}.json'
                if history
                else None
            ),
        )
        lines = [
            "# Fragment Tree Training Report",
            "",
            f"Device: {device}; trainable parameters: {parameters['trainable']:,}",
            f"Best validation loss: {best:.6g}",
            f"Stop reason: {stop_reason}",
            "",
            "Validation generates spectra in inference mode and records original-peak cosine distributions in spectrum_validation/. Prepared-path losses are recorded separately.",
            "",
            "| Epoch | Train loss | Validation loss | Positive branch MIL | Weak-negative branch | Spectrum cosine | LR |",
            "|---|---|---|---|---|---|---|",
        ]
        for item in history:
            lines.append(
                f"| {item['epoch']} | {item['train_loss']:.6g} | {item['validation_loss']:.6g} | {item.get('validation/branch_positive_mil_loss',0):.4f} | {item.get('validation/branch_negative_loss',0):.4f} | {item.get('validation/spectrum_cosine_similarity',0):.4f} | {item.get('learning_rate',0):.3g} |"
            )
        if intermediate_history:
            lines.extend(
                [
                    "",
                    "## Intermediate validation",
                    "",
                    "These fixed-subset checks are observational; scheduling and best-checkpoint selection use full epoch validation.",
                    "",
                    "| Step | Epoch | Samples | Validation loss | Spectrum cosine |",
                    "|---|---|---|---|---|",
                ]
            )
            for item in intermediate_history:
                lines.append(
                    f"| {item['global_step']} | {item['epoch']} | {item['validation_samples']} | {item['validation_loss']:.6g} | {item.get('validation/spectrum_cosine_similarity',0):.4f} |"
                )
        (output / "training_report.md").write_text("\n".join(lines) + "\n")
        return report
    except Exception as error:
        (output / "training_failure.json").write_text(
            json.dumps(
                {
                    "error": str(error),
                    "global_step": global_step,
                    "completed_epochs": len(history),
                },
                indent=2,
            )
        )
        write_training_report(
            output,
            status="failed",
            settings=settings,
            datasets=dataset_report if "dataset_report" in locals() else {},
            completed_epochs=len(history),
            global_step=global_step,
            latest_validation=(
                f'spectrum_validation/epoch_{history[-1]["epoch"]}.json'
                if history
                else None
            ),
            error=error,
        )
        raise
    finally:
        if "stage_progress" in locals():
            stage_progress.close()
        if "iteration_progress" in locals():
            for bar in iteration_progress.values():
                bar.close()
        if "epoch_progress" in locals():
            epoch_progress.close()
        writer.close()
        iteration_writer.close()


class MLPWidths(list):
    def __str__(self):
        return ",".join(map(str, self))


def mlp_dims(value):
    """Comma-separated hidden widths; empty means a single linear layer."""
    try:
        dims = [int(v.strip()) for v in value.split(",")] if value.strip() else []
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated positive integer MLP widths"
        ) from exc
    if any(v < 1 for v in dims):
        raise argparse.ArgumentTypeError("MLP widths must be positive")
    return MLPWidths(dims)


# Model architecture and generation options; each maps to one model_config section.
MODEL_OPTIONS = {
    "reaction-hidden-dim": ("reaction_model_params", "hidden_dim", int, 128),
    "reaction-main-adduct-dim": (
        "reaction_model_params",
        "branch_main_adduct_dim",
        int,
        128,
    ),
    "reaction-num-heads": ("reaction_model_params", "num_heads", int, 4),
    "reaction-state-layers": ("reaction_model_params", "state_num_layers", int, 2),
    "adduct-embedding-dim": ("spectrum_model_params", "adduct_embedding_dim", int, 32),
    "ion-embedding-dim": ("spectrum_model_params", "ion_embedding_dim", int, 32),
    "hydrogen-shift-embedding-dim": (
        "spectrum_model_params",
        "hydrogen_shift_embedding_dim",
        int,
        16,
    ),
    "ion-state-hidden-dim": ("spectrum_model_params", "state_hidden_dim", int, 128),
    "main-adduct-embedding-dim": ("spectrum_model_params", "main_adduct_dim", int, 128),
    "collision-energy-feature-dim": (
        "spectrum_model_params",
        "collision_energy_dim",
        int,
        16,
    ),
    "ce-fc-dims": ("spectrum_model_params", "ce_fc_dims", mlp_dims, mlp_dims("32")),
    "ce-center": ("spectrum_model_params", "ce_center", float, 0.0),
    "ce-scale": ("spectrum_model_params", "ce_scale", float, 100.0),
    "ion-state-fc-dims": (
        "spectrum_model_params",
        "state_fc_dims",
        mlp_dims,
        mlp_dims("128"),
    ),
    "transition-fc-dims": (
        "spectrum_model_params",
        "transition_fc_dims",
        mlp_dims,
        mlp_dims("128,64"),
    ),
    "transition-dropout": ("spectrum_model_params", "transition_dropout", float, 0.0),
    "propagation-steps": ("spectrum_model_params", "propagation_steps", int, 6),
    "branch-path-threshold": ("generation_params", "branch_path_threshold", float, 0.0),
    "max-fragment-nodes": ("generation_params", "max_fragment_nodes", int, 100),
    "peak-intensity-threshold": (
        "generation_params",
        "peak_intensity_threshold",
        float,
        0.0,
    ),
}
MODEL_OPTION_HELP = {
    "transition-fc-dims": "Comma-separated transition MLP hidden widths, e.g. 128,64. Empty selects a linear map.",
    "transition-dropout": "Dropout after hidden transition MLP layers.",
    "ion-state-fc-dims": "Comma-separated adduct x hydrogen-shift combination MLP hidden widths.",
    "ion-state-hidden-dim": "Combination MLP output width in competitive mode.",
    "ce-fc-dims": "Comma-separated CE MLP hidden widths. Empty selects a linear map.",
    "ce-center": "Normalize CE as (CE - center) / scale; saved in the model.",
    "ce-scale": "Positive CE normalization scale (default 100). Does not convert measurement units.",
    "collision-energy-feature-dim": "CE MLP output width.",
    "propagation-steps": "Number of Markov propagation steps including self transitions, not reaction count.",
}
# Training objective of the spectrum model: (default, help).
SPECTRUM_LOSS_OPTIONS = {
    "intensity-loss-weight": (
        1.0,
        "Weight of the independent spectrum intensity objective.",
    ),
    "cosine-loss-weight": (
        0.5,
        "Weight of the spectrum cosine term added to the intensity objective.",
    ),
}
SPECTRUM_LOSS_KEYS = tuple(flag.replace("-", "_") for flag in SPECTRUM_LOSS_OPTIONS)
MOL_ENCODER_OPTIONS = {
    flag: (key, kind, default)
    for flag, key, kind, default in SHARED_MOL_ENCODER_OPTIONS
}
# reaction_model_params.max_roles is intentionally absent from MODEL_OPTIONS.
# Fresh streaming runs use the generator's safe pattern-derived capacity;
# resumed checkpoints retain the exact capacity stored in their model config.


def training_model_config(args):
    config = {
        "max_samples": args.max_samples,
        "architecture": "fragment-tree-competitive-ion",
        "reaction_model_params": {},
        "spectrum_model_params": {},
        "generation_params": {},
    }
    config["mol_encoder_params"] = {
        key: getattr(args, flag.replace("-", "_"))
        for flag, (key, _, _) in MOL_ENCODER_OPTIONS.items()
    }
    config["mol_encoder_params"]["dropout"] = args.dropout
    config["spectrum_model_params"]["descriptors"] = list(args.descriptors)
    for flag, (section, key, _, default) in MODEL_OPTIONS.items():
        value = getattr(args, flag.replace("-", "_"))
        if value is not None or default is not None:
            config[section][key] = value if value is not None else default
    # Mol Encoder dropout uses --dropout. The reaction-state and spectrum modules have their
    # own setting to avoid overloading it.
    fragment_dropout = args.fragment_tree_dropout
    config["reaction_model_params"]["state_dropout"] = fragment_dropout
    # A closed-choice architecture setting, kept out of the generic int/float
    # MODEL_OPTIONS loop below.
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_mol_encoder_arguments(parser)
    parser.add_argument(
        "--fragment-tree-dropout",
        type=float,
        default=0.5,
        help="Dropout for the reaction state encoder and the spectrum model.",
    )
    for flag, (_, _, kind, default) in MODEL_OPTIONS.items():
        parser.add_argument(
            "--" + flag, type=kind, default=default, help=MODEL_OPTION_HELP.get(flag)
        )
    for flag, (default, text) in SPECTRUM_LOSS_OPTIONS.items():
        parser.add_argument("--" + flag, type=float, default=default, help=text)
    for name in ("train-dir", "val-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-dir", required=True, type=output_directory)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model-architecture-json",
        help="Inline architecture assembly JSON; new models only",
    )
    parser.add_argument(
        "--model-architecture",
        help="Architecture JSON: model_config and pretrained_patterns; new models only",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume")
    parser.add_argument(
        "--initialize-from",
        help="Initialize compatible joint-model weights with a fresh optimizer.",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze all pattern features and train downstream prediction only; requires --fine-tune-checkpoint.",
    )
    parser.add_argument(
        "--fine-tune-checkpoint",
        help="Checkpoint to extend with immutable pattern features; fine-tuned/integrated checkpoints are supported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reuse an existing, non-empty output directory without asking for confirmation.",
    )
    for name, default in (
        ("weight-decay", 0.01),
        ("gradient-clip", 1.0),
        ("branch-weight", 1.0),
        ("negative-weight", 0.2),
        ("branch-mil-temperature", 0.1),
        ("min-lr", 1e-6),
        ("minimum-assignment-score", 0.0),
        ("minimum-assignment-score-without-precursor", 0.0),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    for name, default in (
        ("max-samples", 128),
        ("seed", 42),
        ("warmup-steps", 100),
        ("lr-patience", 3),
        ("early-stopping-patience", 10),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.set_defaults(train_mol_encoder=True)
    parser.add_argument(
        "--cache-dataset-in-memory",
        "--all-data-in-ram",
        action="store_true",
        help="Cache each prepared tensor in RAM after its first use. The default streams from disk and retains only the current source.",
    )
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        default=0,
        help="Evaluate a validation subset every N optimizer steps; 0 disables intermediate validation.",
    )
    parser.add_argument(
        "--train-log-interval-steps",
        type=int,
        default=100,
        help="Write sample-weighted mean training metrics every N optimizer steps; 0 disables step logging (default: 100).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of validation samples used for intermediate checks (0 < fraction <= 1; default: 0.1).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.head_only and not args.fine_tune_checkpoint:
        raise ValueError("--head-only requires --fine-tune-checkpoint")
    output = Path(args.output_dir)
    confirm_output_overwrite(
        output,
        overwrite=args.overwrite,
        resume=args.resume,
        protected_paths=(
            args.train_dir,
            args.val_dir,
            args.fine_tune_checkpoint,
            args.initialize_from,
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_args.json").write_text(json.dumps(vars(args), indent=2))
    model_config = training_model_config(args)
    if args.model_architecture or args.model_architecture_json:
        if args.resume or args.fine_tune_checkpoint:
            raise ValueError("--model-architecture cannot change a resumed checkpoint")
        from .sources import inherit_model_config
        from .pattern_models import assemble_config

        architecture = json.loads(
            args.model_architecture_json or Path(args.model_architecture).read_text()
        )
        model_config = inherit_model_config(
            architecture.get("model_config", architecture),
            args.train_dir,
            args.val_dir,
            saved_model=None,
        )
        if architecture.get("pretrained_patterns"):
            from .sources import dataset_sources

            if dataset_sources(args.train_dir, args.val_dir).get("reaction_models"):
                raise ValueError(
                    "Reaction models are fixed by dataset preparation; do not override them at training time"
                )
            model_config = assemble_config(
                model_config, architecture["pretrained_patterns"]
            )
    # A new expansion must be materialized before training.  Ordinary and
    # resumed runs defer inheritance to train_reactions(), where the already
    # loaded dataset supplies max_roles without another complete disk scan.
    if args.fine_tune_checkpoint and not args.resume:
        from .sources import dataset_sources, inherit_model_config

        saved_model = torch.load(
            args.fine_tune_checkpoint, map_location="cpu", weights_only=False
        )["model_config"]
        saved_params = saved_model
        inherited = dataset_sources(args.train_dir, args.val_dir)
        saved_model = {
            **saved_params,
            "fragmenter_params": inherited["fragmenter_params"],
            "adduct_type_strs": inherited["adduct_type_strs"],
        }
        model_config = inherit_model_config(
            saved_model, args.train_dir, args.val_dir, saved_model=saved_model
        )
        model_config.pop("fine_tuning", None)
        model_config["max_samples"] = args.max_samples
        from .fine_tuning import prepare_model_config

        with tempfile.TemporaryDirectory(prefix="fredis-training-config-") as directory:
            config_path = Path(directory) / "model.json"
            config_path.write_text(json.dumps(model_config))
            patterns_path = Path(directory) / "patterns.json"
            patterns_path.write_text(
                json.dumps(
                    model_config["fragmenter_params"]["fragment_ion_tree_builder"][
                        "reaction_pattern_set"
                    ]
                )
            )
            model_config = prepare_model_config(
                checkpoint_path=args.fine_tune_checkpoint,
                pattern_set_path=patterns_path,
                new_params_path=config_path,
                head_only=args.head_only,
            )
            model_config["fine_tuning"].pop("pattern_set_path", None)
    train_reactions(
        model_config=model_config,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        resume=args.resume,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        branch_weight=args.branch_weight,
        negative_weight=args.negative_weight,
        branch_mil_temperature=args.branch_mil_temperature,
        **{key: getattr(args, key) for key in SPECTRUM_LOSS_KEYS},
        max_samples=args.max_samples,
        minimum_assignment_score=args.minimum_assignment_score,
        minimum_assignment_score_without_precursor=args.minimum_assignment_score_without_precursor,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        lr_patience=args.lr_patience,
        early_stopping_patience=args.early_stopping_patience,
        min_lr=args.min_lr,
        train_mol_encoder=args.train_mol_encoder,
        cache_dataset_in_memory=args.cache_dataset_in_memory,
        initialize_from=args.initialize_from,
        train_log_interval_steps=args.train_log_interval_steps,
        validation_interval_steps=args.validation_interval_steps,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()
