"""Long-lived isolated native worker; stdout is a JSON-lines protocol only."""

from __future__ import annotations
import contextlib
import json
from pathlib import Path
import sys
import traceback
import torch
from fredis.data.core.MSDataset import MSDataset
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.input.reaction_batching import select_samples
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.training.fragment_tree_training.model import (
    ReactionFragmentTreeTrainingModel,
)
from fredis.ml.training.fragment_tree_training.pattern_models import initialize_imports
from fredis.ml.data_preparation.fragment_tree.create_training_data import _prepare_group


def finite_backward(model, data, seed):
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    model.train()
    target = data.spectrum_batch.target_intensity
    if target is None or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite or missing training labels")
    output = model(data)
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Nonfinite loss")
    if any(not torch.isfinite(value).all() for value in output.metrics.values()):
        raise FloatingPointError("Nonfinite training metrics")
    output.loss.backward()
    gradients = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    if not gradients:
        raise FloatingPointError("No gradients produced")
    if any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("Nonfinite gradients")
    return float(output.loss.detach().cpu())


def evaluate(task, config, generator, model):
    dataset = MSDataset.load(task["input"], load_peak_metadata=False)
    positions = task.get("positions", list(range(len(dataset))))
    dataset = dataset[positions]
    indices = [task["record_indices"][i] for i in positions]

    def excluded(index, stage, error):
        return dict(
            index=index,
            Status="excluded",
            Stage=stage,
            Reason=str(error),
            ReasonCategory=type(error).__name__,
        )

    directory = Path(task["workdir"])
    directory.mkdir(parents=True, exist_ok=True)
    builder = ReactionStructureBuilder(generator, **config["limits"])
    options = dict(
        output_dir=str(directory),
        adduct_type_column=config["mapping"]["adductTypeColumn"],
        collision_energy_column=config["mapping"]["collisionEnergyColumn"],
        precursor_mz_column=config["mapping"]["precursorMzColumn"],
        minimum_relative_intensity=config["minimum_relative_intensity"],
    )
    try:
        path, kept, annotations, summary, search_stats = _prepare_group(
            (0, task["smiles"], indices, dataset), builder, options
        )
        structure = torch.load(path, map_location="cpu", weights_only=False)[
            "structure"
        ]
    except Exception as error:
        if len(indices) > 1:
            # A spectrum-specific preparation error must not exclude its siblings.
            result = []
            for position in positions:
                result.extend(
                    evaluate(
                        {**task, "positions": [position]}, config, generator, model
                    )
                )
            return result
        return [excluded(indices[0], "preparation", error)]
    rows = {
        i: dict(
            index=i,
            Status="excluded",
            Stage="preparation",
            Reason="Unresolved precursor reaction sequence",
            ReasonCategory="UnresolvedPrecursorError",
        )
        for i in indices
    }

    def check(local):
        sliced = select_samples(structure, local).to(config["device"])
        loss = finite_backward(model, sliced, config["seed"])
        for i in local:
            annotation = annotations[i]
            rows[kept[i]] = dict(
                index=kept[i],
                Status="trainable",
                Stage="forward_loss_backward",
                Reason="",
                ReasonCategory="",
                TrainingAdduct=annotation.get("adduct"),
                TrainingMainAdduct=annotation.get("mainAdduct"),
                TrainingCollisionEnergy=annotation.get("collisionEnergy"),
                Loss=loss,
                GradientsFinite=True,
                AssignmentScore=annotation.get("assignmentScore"),
            )

    for start in range(0, len(kept), config["batch_size"]):
        local = list(range(start, min(start + config["batch_size"], len(kept))))
        try:
            check(local)
        except Exception:
            for i in local:
                try:
                    check([i])
                except Exception as error:
                    rows[kept[i]] = excluded(kept[i], "forward_loss_backward", error)
    return [rows[i] for i in indices]


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    with contextlib.redirect_stdout(sys.stderr):
        generator = create_spectrum_generator(config["model_config"])
        model = ReactionFragmentTreeTrainingModel(
            generator.feature_model, spectrum_model=generator.spectrum_model
        )
        if config["model_config"].get("fine_tuning", {}).get("pattern_imports"):
            initialize_imports(model, config["model_config"])
        model.to(config["device"])
    for line in sys.stdin:
        task = {}
        try:
            task = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = evaluate(task, config, generator, model)
            print(
                json.dumps(
                    {"ok": True, "request_id": task["request_id"], "records": result}
                ),
                flush=True,
            )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(
                json.dumps(
                    {
                        "ok": False,
                        "request_id": task.get("request_id"),
                        "error": f"{type(error).__name__}: {error}",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
