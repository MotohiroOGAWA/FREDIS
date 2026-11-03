"""Estimate peak training memory for a configured model and prepared dataset.

Peak memory = resident + per-batch, where

* resident memory is computed exactly from the model (weights and buffers plus
  the two fp32 AdamW moments per trainable parameter) plus the GPU memory held
  outside PyTorch's allocator (CUDA context, other processes);
* per-batch memory (activations kept for backward plus gradients) is measured by
  running forward/backward on real training batches and fitted as
  ``intercept + a * ion_states + b * transition_states``.

Both predictors are known for every chunk from the preparation sidecars
(``assignment_scores.tsv`` / ``manifest.tsv``), so the fit can be applied to
every batch the configured batching would build, and to candidate
``max_batch_ion_states`` limits when searching for the largest one that fits.
On NIST-scale data the two-term fit explained ~98% of the variance, against
~89% for ion states alone.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch

GIB = 1024**3
PREDICTORS = ("ion_states", "transition_states")


def _quantile(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _histogram(values, bins=30):
    if not values:
        return dict(edges=[], counts=[])
    high = max(values) or 1
    width = high / bins
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int(value / width))] += 1
    return dict(edges=[round(width * i) for i in range(bins + 1)], counts=counts)


def fit_memory_model(points, predictors=PREDICTORS):
    """Least squares with non-negative coefficients (drops predictors that go negative).

    ``margin`` is the largest relative under-prediction among the larger half of
    the measurements; predictions are multiplied by ``1 + margin``.
    """
    points = [p for p in points if p.get("dynamic_bytes")]
    if not points:
        return None
    y = torch.tensor([float(p["dynamic_bytes"]) for p in points], dtype=torch.float64)
    active = [k for k in predictors if all(p.get(k) is not None for p in points)]
    if not active:
        return None
    while True:
        if len(points) <= len(active):
            # Too few points for an intercept: proportional to the first predictor.
            key = active[0]
            ratio = max(float(p["dynamic_bytes"]) / max(1.0, float(p[key])) for p in points)
            coefficients = {"intercept": 0.0, key: ratio}
            break
        A = torch.stack(
            [torch.ones(len(points), dtype=torch.float64)]
            + [torch.tensor([float(p[k]) for p in points], dtype=torch.float64) for k in active],
            1,
        )
        solution = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1).tolist()
        coefficients = dict(zip(["intercept", *active], solution))
        negative = [k for k in active if coefficients[k] < 0]
        if negative and len(active) > 1:
            active.remove(min(negative, key=lambda k: coefficients[k]))
            continue
        coefficients = {k: max(0.0, v) for k, v in coefficients.items()}
        break
    for key in predictors:
        coefficients.setdefault(key, 0.0)
    predicted = [
        coefficients["intercept"] + sum(coefficients[k] * float(p.get(k) or 0) for k in predictors)
        for p in points
    ]
    observed = y.tolist()
    mean = sum(observed) / len(observed)
    total = sum((v - mean) ** 2 for v in observed)
    residual = sum((v - q) ** 2 for v, q in zip(observed, predicted))
    large = sorted(observed)[len(observed) // 2]
    margin = max(
        (max(0.0, (v - q) / q) for v, q in zip(observed, predicted) if q > 0 and v >= large),
        default=0.0,
    )
    return dict(
        intercept_bytes=coefficients["intercept"],
        bytes_per_ion_state=coefficients["ion_states"],
        bytes_per_transition_state=coefficients["transition_states"],
        r2=(1 - residual / total) if total > 0 and len(points) > len(active) + 1 else None,
        margin=margin,
        points=len(points),
    )


def predict_dynamic(fit, ion_states, transition_states):
    if not fit:
        return None
    value = (
        fit["intercept_bytes"]
        + fit["bytes_per_ion_state"] * ion_states
        + fit["bytes_per_transition_state"] * transition_states
    )
    return value * (1 + fit["margin"])


@dataclass
class _Chunk:
    samples: int
    ion_states: int
    transition_states: int


class _ChunkPlan:
    """Duck-typed piece sequence for plan_batch_groups that never loads tensors."""

    def __init__(self, chunks, blocks):
        self.chunks = chunks
        self.blocks = blocks  # chunk indices per source file, in file order

    def __len__(self):
        return len(self.chunks)

    def ordered_indices(self, shuffle):
        blocks = [list(block) for block in self.blocks]
        if shuffle:
            random.shuffle(blocks)
        return [i for block in blocks for i in block]

    def sample_count(self, index):
        return self.chunks[index].samples

    def ion_state_count(self, index):
        return self.chunks[index].ion_states

    @classmethod
    def from_pieces(cls, pieces, max_batch_ion_states=0):
        """Re-chunk with an ion-state limit, approximating prepared_piece_sequence.

        Per-sample ion states are not kept in the index, so a chunk above the
        limit is split into equal parts; if even one sample exceeds the limit
        on average, the chunk's samples are counted as excluded.
        """
        chunks, blocks, dropped, last_path = [], [], 0, None
        for index in range(len(pieces)):
            piece = pieces.piece(index)
            if piece.path != last_path:
                blocks.append([])
                last_path = piece.path
            parts = 1
            if max_batch_ion_states and piece.ion_states > max_batch_ion_states:
                parts = math.ceil(piece.ion_states / max_batch_ion_states)
                if parts > piece.num_samples:
                    dropped += piece.num_samples
                    continue
            for part in range(parts):
                samples = piece.num_samples // parts + (part < piece.num_samples % parts)
                blocks[-1].append(len(chunks))
                chunks.append(
                    _Chunk(
                        samples,
                        round(piece.ion_states * samples / max(1, piece.num_samples)),
                        piece.transition_states,
                    )
                )
        return cls(chunks, [block for block in blocks if block]), dropped


def _batch_features(data):
    """Measured sizes of a loaded batch: the predictors plus diagnostics."""

    def size(getter):
        try:
            return int(getter())
        except Exception:
            return None

    counts = None
    graph = getattr(data.compounds, "graph", None)
    if getattr(graph, "batch", None) is not None:
        counts = torch.bincount(graph.batch)
    return dict(
        ion_states=size(lambda: data.spectrum_explanations.formula_explanation_index.numel()),
        transition_states=size(lambda: data.execution_graph.compound_index.numel()),
        samples=size(lambda: data.num_samples),
        formula_explanations=size(lambda: data.formula_explanations.formula_index.numel()),
        targets=size(lambda: data.targets.formula_index.numel()),
        compounds=size(lambda: len(data.compounds.compounds)),
        atoms=int(counts.sum()) if counts is not None else None,
    )


class _SavedTensorMeter:
    """CPU fallback: sum the unique storages saved for backward."""

    def __init__(self):
        self.bytes = 0
        self._seen = set()

    def pack(self, tensor):
        try:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in self._seen:
                self._seen.add(key)
                self.bytes += storage.nbytes()
        except Exception:
            self.bytes += tensor.numel() * tensor.element_size()
        return tensor

    @staticmethod
    def unpack(tensor):
        return tensor


def estimate_training_memory(
    *,
    model,
    optimizer_parameters,
    datasets,
    plan_batch_groups,
    batch_size: int,
    max_samples: int,
    max_batch_ion_states: int,
    device: torch.device,
    profile_batches: int = 6,
    seed: int = 42,
    target_memory_gib: float | None = None,
    headroom: float = 0.1,
    log=print,
) -> dict:
    from fredis.ml.input.source_reaction_structure import SourceReactionStructure

    train = datasets[0]
    use_cuda = device.type == "cuda"
    if use_cuda and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    parameters = list(model.parameters())
    trainable = [p for p in optimizer_parameters if p.requires_grad]
    weight_bytes = sum(p.numel() * p.element_size() for p in parameters) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    trainable_bytes = sum(p.numel() * p.element_size() for p in trainable)
    optimizer_bytes = 2 * trainable_bytes  # AdamW exp_avg + exp_avg_sq
    static = dict(
        parameters=sum(p.numel() for p in parameters),
        trainable_parameters=sum(p.numel() for p in trainable),
        weights_bytes=weight_bytes,
        gradients_bytes=trainable_bytes,
        optimizer_state_bytes=optimizer_bytes,
    )

    def plan(limit):
        """Chunks and batches training would build for an ion-state limit (seeded)."""
        chunks, dropped = _ChunkPlan.from_pieces(train, limit)
        state = random.getstate()
        random.seed(seed)
        groups = plan_batch_groups(
            chunks, True, batch_size=batch_size, max_samples=max_samples, max_batch_ion_states=limit
        )
        random.setstate(state)
        totals = [
            (
                sum(chunks.chunks[i].ion_states for i in group),
                sum(chunks.chunks[i].transition_states for i in group),
            )
            for group in groups
        ]
        return chunks, totals, dropped

    chunk_plan, totals, _ = plan(max_batch_ion_states)
    batch_ions = [ions for ions, _ in totals]

    # Profile real batches from small to the largest, stopping at the first OOM.
    real_groups = plan_batch_groups(
        train, True, batch_size=batch_size, max_samples=max_samples,
        max_batch_ion_states=max_batch_ion_states,
    )
    real_ions = [sum(train.ion_state_count(i) for i in group) for group in real_groups]
    order = sorted(range(len(real_groups)), key=lambda i: real_ions[i])
    count = max(2, profile_batches)
    chosen = []
    for i in range(count):
        q = max(0.05, 1 - (1 - i / (count - 1)) ** 3)  # quantiles packed towards the tail
        index = order[min(len(order) - 1, max(0, math.ceil(q * len(order)) - 1))]
        if index not in chosen:
            chosen.append(index)

    measurements, oom = [], None
    model.train(True)
    for rank, group_index in enumerate(chosen, 1):
        group = real_groups[group_index]
        log(f"Profiling batch {rank}/{len(chosen)} (~{real_ions[group_index]:,} ion states)", flush=True)
        data = result = None
        try:
            data = SourceReactionStructure.from_structures([train[i] for i in group])
            features = _batch_features(data)
            started = time.perf_counter()
            for p in trainable:
                p.grad = None
            if use_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                data = data.to(device)
                result = model(data)
                result.loss.backward()
                torch.cuda.synchronize(device)
                peak_allocated = torch.cuda.max_memory_allocated(device)
                dynamic = peak_allocated - base
                slack = max(0, torch.cuda.max_memory_reserved(device) - peak_allocated)
            else:
                meter = _SavedTensorMeter()
                with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
                    result = model(data)
                result.loss.backward()
                dynamic, slack = meter.bytes + trainable_bytes, 0
            measurements.append(
                dict(
                    chunks=len(group),
                    **features,
                    dynamic_bytes=int(dynamic),
                    allocator_slack_bytes=int(slack),
                    seconds=round(time.perf_counter() - started, 3),
                )
            )
        except torch.cuda.OutOfMemoryError as error:
            oom = dict(
                ion_states=real_ions[group_index],
                chunks=len(group),
                message=str(error).split("\n")[0],
            )
            log(f"Out of memory at ~{real_ions[group_index]:,} ion states; stopping profiling.", flush=True)
            break
        finally:
            del result, data
            for p in trainable:
                p.grad = None
            if use_cuda:
                torch.cuda.empty_cache()

    fit = fit_memory_model(measurements)
    gpu = None
    if use_cuda:
        free, total = torch.cuda.mem_get_info(device)
        gpu = dict(
            name=torch.cuda.get_device_name(device),
            total_bytes=int(total),
            non_pytorch_bytes=int(max(0, total - free - torch.cuda.memory_reserved(device))),
        )
    capacity = int(target_memory_gib * GIB) if target_memory_gib else (gpu["total_bytes"] if gpu else None)
    slack = max((m["allocator_slack_bytes"] for m in measurements), default=0)
    resident = weight_bytes + optimizer_bytes + (gpu["non_pytorch_bytes"] if gpu else 0)

    def peak(ions, transitions):
        dynamic = predict_dynamic(fit, ions, transitions)
        return None if dynamic is None else int(resident + slack + dynamic)

    peaks = [peak(i, t) for i, t in totals] if fit else []
    # Shuffle-independent bound: the largest chunks meeting in one batch.
    top_ions = sorted((c.ion_states for c in chunk_plan.chunks), reverse=True)[: max(1, batch_size)]
    top_transitions = sorted((c.transition_states for c in chunk_plan.chunks), reverse=True)[: max(1, batch_size)]
    worst = (
        min(sum(top_ions), max_batch_ion_states) if max_batch_ion_states else sum(top_ions),
        sum(top_transitions),
    )
    batches = dict(
        count=len(totals),
        ion_states_median=_quantile(batch_ions, 0.5),
        ion_states_p99=_quantile(batch_ions, 0.99),
        ion_states_max=max(batch_ions, default=0),
        max_chunk_ion_states=max((c.ion_states for c in chunk_plan.chunks), default=0),
        ion_state_counts="per_sample" if train.ion_state_counts_exact else "file_average",
        samples_over_ion_state_limit=train.oversized_samples,
        histogram=_histogram(batch_ions),
        predicted_peak_histogram=_histogram(peaks) if peaks else None,
    )
    predictions = dict(
        median=dict(ion_states=batches["ion_states_median"], peak_bytes=_quantile(peaks, 0.5) if peaks else None),
        p99=dict(ion_states=batches["ion_states_p99"], peak_bytes=_quantile(peaks, 0.99) if peaks else None),
        max=dict(ion_states=batches["ion_states_max"], peak_bytes=max(peaks) if peaks else None),
        worst_case=dict(ion_states=worst[0], transition_states=worst[1], peak_bytes=peak(*worst)),
    )

    recommendation = None
    if fit and capacity:
        usable = capacity * (1 - headroom)

        def evaluate(limit):
            _, candidate, dropped = plan(limit)
            highest = max((peak(i, t) for i, t in candidate), default=0)
            return highest, dropped

        current_peak, _ = evaluate(max_batch_ion_states)
        # Largest ion-state limit whose largest simulated batch fits (binary search).
        low, high = 1_000, max(1_000, batches["max_chunk_ion_states"] * max(1, batch_size))
        best = None
        if evaluate(0)[0] <= usable:
            best = 0  # the data fits without a limit
        else:
            while high - low > max(1_000, low // 50):
                middle = (low + high) // 2
                if evaluate(middle)[0] <= usable:
                    low = middle
                else:
                    high = middle
            if evaluate(low)[0] <= usable:
                scale = 10 ** max(0, len(str(low)) - 2)
                best = max(1_000, low // scale * scale)  # two significant figures
        best_peak, best_dropped = evaluate(best) if best is not None else (None, 0)
        recommendation = dict(
            capacity_bytes=int(capacity),
            usable_bytes=int(usable),
            headroom=headroom,
            current_settings_fit=current_peak <= usable,
            current_settings_peak_bytes=int(current_peak),
            max_batch_ion_states=best,
            recommended_peak_bytes=None if best_peak is None else int(best_peak),
            samples_excluded=int(best_dropped),
        )
    return dict(
        schema="fredis.training-memory-estimate",
        method="cuda-peak-allocated" if use_cuda else "cpu-saved-tensors (approximate; CUDA peak is usually higher)",
        device=str(device),
        gpu=gpu,
        settings=dict(batch_size=batch_size, max_samples=max_samples, max_batch_ion_states=max_batch_ion_states),
        static=static,
        resident_bytes=int(resident),
        allocator_slack_bytes=int(slack),
        batches=batches,
        measurements=measurements,
        out_of_memory=oom,
        fit=fit,
        predictions=predictions,
        recommendation=recommendation,
        notes=[
            "Peak = weights + AdamW state + non-PyTorch GPU memory + allocator slack + fitted (activations + gradients).",
            "Batches are simulated from the dataset index with the training seed; other epochs reshuffle, so the worst case combines the largest chunks.",
            "Validation runs without gradients and free-running spectrum generation is not profiled.",
        ],
    )
