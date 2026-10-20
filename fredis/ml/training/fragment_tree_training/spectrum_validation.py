"""Free-running validation and report-ready spectrum diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fredis.libs.mmkit.mmkit import Compound, Adduct

QUANTILES = (
    ("q10", 0.10),
    ("q25", 0.25),
    ("median", 0.50),
    ("q75", 0.75),
    ("q90", 0.90),
)


def matched_peak_pairs(mz, intensity, observed, tolerance):
    """Return greedy one-to-one matches ordered by predicted/observed product."""
    candidates = sorted(
        (
            (float(intensity[i] * observed[j]["intensity"]), i, j)
            for i, mass in enumerate(mz)
            for j, peak in enumerate(observed)
            if tolerance.within(float(peak["mz"]), float(mass))
        ),
        reverse=True,
    )
    used_x, used_y, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_x or j in used_y:
            continue
        used_x.add(i)
        used_y.add(j)
        pairs.append((i, j))
    return pairs


def matched_cosine(mz, intensity, observed, tolerance):
    """One-to-one tolerance cosine; unmatched peaks retain norm mass."""
    y = np.asarray([p["intensity"] for p in observed], dtype=float)
    x = np.asarray(intensity, dtype=float)
    dot = sum(
        float(x[i] * y[j]) for i, j in matched_peak_pairs(mz, x, observed, tolerance)
    )
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    return min(1.0, max(0.0, dot / norm)) if norm else 0.0


def assignment_score(mz, observed, tolerance):
    """Intensity-weighted observed-m/z coverage, independent of predicted intensity."""
    total = sum(float(peak["intensity"]) for peak in observed)
    if total <= 0:
        return None
    matched = {
        j for _, j in matched_peak_pairs(mz, np.ones(len(mz)), observed, tolerance)
    }
    return sum(float(observed[j]["intensity"]) for j in matched) / total


TOP_N_VALUES = (1, 5, 10, 20)


def top_n_assignment_metrics(
    generated, observed, tolerance, *, ns=TOP_N_VALUES, suffix=""
):
    """Recall/precision of the model's own top-N generated peaks (ranked by its
    own predicted intensity) against observed peaks with prepared assignment
    metadata (peak['assigned']), i.e. against the actual ground truth rather
    than against every observed m/z regardless of whether it was ever
    explained. A high assignment_score is cheap to reach by generating many
    fragment nodes (more candidates, more chances to match); precision@N
    (fixed denominator N, the standard "precision at k") makes that costly
    instead of rewarding it, while recall@N still favors covering the ground
    truth. N is capped by the model's own candidate count, so predicting
    fewer than N peaks is scored as missing the remainder, not excused.
    """
    assigned = [peak for peak in observed if peak.get("assigned")]
    ranked = sorted(generated, key=lambda peak: peak["intensity"], reverse=True)
    result = {}
    for n in ns:
        top = ranked[:n]
        matched = (
            {
                j
                for _, j in matched_peak_pairs(
                    [p["mz"] for p in top],
                    [p["intensity"] for p in top],
                    assigned,
                    tolerance,
                )
            }
            if top and assigned
            else set()
        )
        # No assigned ground truth at all means the question isn't answerable,
        # not that the model scored zero (mirrors assignment_score's None).
        result[f"peak_recall_top{n}{suffix}"] = (
            (len(matched) / len(assigned)) if assigned else None
        )
        result[f"peak_precision_top{n}{suffix}"] = (
            (len(matched) / n) if assigned else None
        )
    return result


def distribution(values):
    values = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if not len(values):
        return None
    result = {
        name: float(np.quantile(values, quantile)) for name, quantile in QUANTILES
    }
    result.update(mean=float(values.mean()), count=int(len(values)))
    return result


def _node_depths(spectrum_batch, sample, precursor_mz, tolerance):
    """Measure graph depth in the predicted tree from precursor-emitting nodes (precursor is depth 0)."""
    decoded = spectrum_batch.decoded
    sample_nodes = set(
        torch.nonzero(decoded.node_sample_index == sample, as_tuple=False)
        .flatten()
        .tolist()
    )
    candidate_rows = (
        torch.nonzero(
            spectrum_batch.physical_candidate_sample_index == sample, as_tuple=False
        )
        .flatten()
        .tolist()
    )
    seeds = {
        int(spectrum_batch.physical_candidate_node_index[row])
        for row in candidate_rows
        if tolerance.within(
            float(spectrum_batch.physical_candidate_mz[row]), float(precursor_mz)
        )
    }
    if not seeds and sample_nodes:
        seeds.add(min(sample_nodes))
    depths = {node: 0 for node in seeds}
    edges = decoded.edge_index.detach().cpu().T.tolist()
    for _ in range(len(sample_nodes)):
        changed = False
        for source, target in edges:
            if source in depths and target in sample_nodes:
                value = depths[source] + 1
                if target not in depths or value < depths[target]:
                    depths[target] = value
                    changed = True
        if not changed:
            break
    return depths


def _formula_depths(spectrum_batch, sample, node_depths, precursor_mz, tolerance):
    result = {}
    for row in (
        torch.nonzero(
            spectrum_batch.physical_candidate_sample_index == sample, as_tuple=False
        )
        .flatten()
        .tolist()
    ):
        node = int(spectrum_batch.physical_candidate_node_index[row])
        if tolerance.within(
            float(spectrum_batch.physical_candidate_mz[row]), float(precursor_mz)
        ):
            result[row] = 0
        elif node in node_depths:
            result[row] = node_depths[node]
    return result


def _original_depths(data, sample, annotation, tolerance):
    spectrum_batch = data.spectrum_batch
    if spectrum_batch is None:
        return [0 if peak.get("precursor") else None for peak in annotation["peaks"]]
    node_depths = _node_depths(
        spectrum_batch, sample, annotation["precursorMz"], tolerance
    )
    result = []
    for peak in annotation["peaks"]:
        if peak.get("precursor"):
            result.append(0)
            continue
        candidates = [
            node_depths[node]
            for match in peak.get("matches", ())
            for node in match.get("nodeIndices", ())
            if node in node_depths
        ]
        result.append(min(candidates) if candidates else None)
    return result


def _ce_categories(records):
    """Assign low/mid/high CE using empirical tertile boundaries."""
    energies = np.asarray(
        [record["collision_energy"] for record in records], dtype=float
    )
    low, high = np.quantile(energies, (1 / 3, 2 / 3))
    fmt = lambda value: f"{value:g}"
    labels = (
        (f"Low (< {fmt(low)} eV)", f"{fmt(low)} eV", f"High (> {fmt(high)} eV)")
        if low == high
        else (
            f"Low (≤ {fmt(low)} eV)",
            f"Mid ({fmt(low)}–{fmt(high)} eV)",
            f"High (> {fmt(high)} eV)",
        )
    )
    for record in records:
        value = record["collision_energy"]
        record["collision_energy_bin"] = (
            labels[0] if value <= low else labels[1] if value <= high else labels[2]
        )
    return {
        "boundaries": [float(low), float(high)],
        "groups": [
            {
                "label": label,
                "count": sum(r["collision_energy_bin"] == label for r in records),
            }
            for label in labels
        ],
    }


def _representatives(records):
    result = {}
    for metric in (
        "cosine_similarity",
        "cosine_similarity_without_precursor",
        "assignment_score",
        "assignment_score_without_precursor",
    ):
        available = [
            (index, row[metric])
            for index, row in enumerate(records)
            if row.get(metric) is not None
        ]
        if not available:
            continue
        values = np.asarray([value for _, value in available], dtype=float)
        result[metric] = {}
        for name, quantile in QUANTILES:
            target = float(np.quantile(values, quantile))
            index, value = min(
                available, key=lambda item: (abs(item[1] - target), item[0])
            )
            result[metric][name] = {
                "spectrum_index": index,
                "value": float(value),
                "target": target,
            }
    return result


def append_metric_distributions(output_dir, global_step, split, metrics, *, epoch=None):
    """Append q10/q25/median/q75/q90/mean series to the live TSV."""
    rows = []
    for metric, values in metrics.items():
        stats = distribution(values)
        if stats is None:
            continue
        for statistic in ("q10", "q25", "median", "q75", "q90", "mean"):
            rows.append(
                (
                    global_step,
                    "" if epoch is None else epoch,
                    split,
                    f"{metric}_{statistic}",
                    stats[statistic],
                )
            )
    if not rows:
        return
    target = Path(output_dir) / "metric_distributions.tsv"
    with target.open("a") as stream:
        if target.stat().st_size == 0:
            stream.write("global_step\tepoch\tsplit\tmetric\tvalue\n")
        for step, item_epoch, item_split, metric, value in rows:
            stream.write(f"{step}\t{item_epoch}\t{item_split}\t{metric}\t{value}\n")


def _report_metrics(records):
    metrics = {}
    depth_names = (
        "cosine_similarity",
        "cosine_similarity_without_precursor",
        "assignment_score",
        "assignment_score_without_precursor",
    )
    # Top-N recall/precision are already a coarse, whole-spectrum ranking; a
    # per-depth split of a handful of candidates isn't meaningful, so they get
    # the flat + collision-energy/adduct facets only, not the depth facet.
    top_n_names = tuple(
        f"peak_{kind}_top{n}{suffix}"
        for kind in ("recall", "precision")
        for n in TOP_N_VALUES
        for suffix in ("", "_without_precursor")
    )
    names = depth_names + top_n_names
    for name in names:
        metrics[name] = [record.get(name) for record in records]
    for dimension, field in (
        ("collision_energy", "collision_energy_bin"),
        ("main_adduct", "main_adduct"),
    ):
        for category in dict.fromkeys(record[field] for record in records):
            selected = [record for record in records if record[field] == category]
            for name in names:
                metrics[f"{name}@{dimension}:{category}"] = [
                    record.get(name) for record in selected
                ]
    depths = sorted(
        {int(depth) for record in records for depth in record.get("depth_metrics", {})}
    )
    for depth in depths:
        for name in depth_names:
            metrics[f"{name}@depth:{depth}"] = [
                record.get("depth_metrics", {}).get(str(depth), {}).get(name)
                for record in records
            ]
    return metrics


@torch.no_grad()
def validate_spectra(
    generator, pieces, output_dir, label, *, global_step=None, progress=None, epoch=None
):
    was_training = generator.training
    generator.eval()
    records = []
    tolerance = generator.fragmenter.mass_tolerance
    try:
        for piece_index, data in enumerate(pieces):
            if progress is not None:
                progress(piece_index, "start")
            if len(data.sample_annotations) != data.num_samples:
                raise ValueError(
                    "Spectrum validation requires original peak annotations; regenerate structures"
                )
            sources = [
                Compound.from_smiles(data.source_smiles[int(t)])
                for t in data.sample_tree_index
            ]
            annotations = data.sample_annotations
            adducts = [Adduct.parse(a["adduct"]) for a in annotations]
            energies = [a["collisionEnergy"] for a in annotations]
            for output in generator.predict_batches(sources, adducts, energies):
                for sample, input_index in enumerate(output.sample_input_index):
                    annotation = annotations[input_index]
                    formula_rows = (
                        torch.nonzero(
                            output.spectrum_batch.physical_candidate_sample_index
                            == sample,
                            as_tuple=False,
                        )
                        .flatten()
                        .tolist()
                    )
                    mz = [
                        float(output.spectrum_batch.physical_candidate_mz[row])
                        for row in formula_rows
                    ]
                    intensity = [
                        float(output.spectra.intensity[row]) for row in formula_rows
                    ]
                    confidence = [
                        float(output.spectra.confidence[row]) for row in formula_rows
                    ]
                    if not np.isfinite(intensity).all():
                        raise FloatingPointError("Non-finite generated spectrum")
                    observed = [
                        dict(
                            mz=float(p["mz"]),
                            intensity=float(p["intensity"]),
                            precursor=bool(p.get("precursor")),
                            assigned=bool(p.get("matches")),
                        )
                        for p in annotation["peaks"]
                    ]
                    precursor_mz = float(annotation["precursorMz"])
                    node_depths = _node_depths(
                        output.spectrum_batch, sample, precursor_mz, tolerance
                    )
                    formula_depth = _formula_depths(
                        output.spectrum_batch,
                        sample,
                        node_depths,
                        precursor_mz,
                        tolerance,
                    )
                    # confidence never changes which peaks are scored (matched_cosine/
                    # assignment_score below still see every candidate), it only lets a
                    # renderer dim or hide an unconfident adduct/hydrogen-shift guess.
                    generated = [
                        dict(
                            mz=mass,
                            intensity=value,
                            confidence=conf,
                            depth=formula_depth.get(row),
                            precursor=bool(tolerance.within(mass, precursor_mz)),
                        )
                        for row, mass, value, conf in zip(
                            formula_rows, mz, intensity, confidence
                        )
                    ]
                    grouped = {}
                    for row, peak in zip(formula_rows, generated):
                        key = tuple(
                            output.spectra.formula_tensor[row].detach().cpu().tolist()
                        )
                        if key not in grouped:
                            grouped[key] = dict(peak)
                        else:
                            grouped[key]["intensity"] += peak["intensity"]
                            depths = [
                                v
                                for v in (grouped[key]["depth"], peak["depth"])
                                if v is not None
                            ]
                            grouped[key]["depth"] = min(depths) if depths else None
                    generated = list(grouped.values())
                    mz = [p["mz"] for p in generated]
                    intensity = [p["intensity"] for p in generated]
                    original_depth = _original_depths(
                        data, input_index, annotation, tolerance
                    )
                    for peak, depth in zip(observed, original_depth):
                        peak["depth"] = depth
                    generated_np = [peak for peak in generated if not peak["precursor"]]
                    observed_np = [peak for peak in observed if not peak["precursor"]]
                    depth_metrics = {}
                    all_depths = sorted(
                        {
                            peak["depth"]
                            for peak in observed + generated
                            if peak["depth"] is not None
                        }
                    )
                    for depth in all_depths:
                        predicted = [
                            peak for peak in generated if peak["depth"] == depth
                        ]
                        target = [peak for peak in observed if peak["depth"] == depth]
                        if not target:
                            continue
                        predicted_np = [
                            peak for peak in predicted if not peak["precursor"]
                        ]
                        target_np = [peak for peak in target if not peak["precursor"]]
                        depth_metrics[str(depth)] = {
                            "cosine_similarity": matched_cosine(
                                [p["mz"] for p in predicted],
                                [p["intensity"] for p in predicted],
                                target,
                                tolerance,
                            ),
                            "cosine_similarity_without_precursor": (
                                matched_cosine(
                                    [p["mz"] for p in predicted_np],
                                    [p["intensity"] for p in predicted_np],
                                    target_np,
                                    tolerance,
                                )
                                if target_np
                                else None
                            ),
                            "assignment_score": assignment_score(
                                [p["mz"] for p in predicted], target, tolerance
                            ),
                            "assignment_score_without_precursor": (
                                assignment_score(
                                    [p["mz"] for p in predicted_np],
                                    target_np,
                                    tolerance,
                                )
                                if target_np
                                else None
                            ),
                            "observed_peak_count": len(target),
                            "generated_peak_count": len(predicted),
                        }
                    branch_group = int(data.sample_branch_group_index[sample])
                    nodes = output.fragments.node_sample_index == branch_group
                    raw = output.selection.decoded.state_sample_index == branch_group
                    path_depth = (
                        float(
                            (output.selection.decoded.state_reaction_index[raw] >= 0)
                            .sum(1)
                            .float()
                            .mean()
                        )
                        if torch.any(raw)
                        else 0.0
                    )
                    top_metrics = top_n_assignment_metrics(
                        generated, observed, tolerance
                    )
                    top_metrics.update(
                        top_n_assignment_metrics(
                            generated_np,
                            observed_np,
                            tolerance,
                            suffix="_without_precursor",
                        )
                        if observed_np
                        else {
                            f"peak_{kind}_top{n}_without_precursor": None
                            for kind in ("recall", "precision")
                            for n in TOP_N_VALUES
                        }
                    )
                    records.append(
                        dict(
                            piece_index=piece_index,
                            sample_index=input_index,
                            adduct=annotation["adduct"],
                            main_adduct=annotation.get(
                                "mainAdduct",
                                str(
                                    generator.fragmenter._resolve_main_adduct_type(
                                        adducts[input_index]
                                    )
                                ),
                            ),
                            collision_energy=float(annotation["collisionEnergy"]),
                            precursor_mz=precursor_mz,
                            cosine_similarity=matched_cosine(
                                mz, intensity, observed, tolerance
                            ),
                            cosine_similarity_without_precursor=(
                                matched_cosine(
                                    [p["mz"] for p in generated_np],
                                    [p["intensity"] for p in generated_np],
                                    observed_np,
                                    tolerance,
                                )
                                if observed_np
                                else None
                            ),
                            assignment_score=assignment_score(mz, observed, tolerance),
                            assignment_score_without_precursor=(
                                assignment_score(
                                    [p["mz"] for p in generated_np],
                                    observed_np,
                                    tolerance,
                                )
                                if observed_np
                                else None
                            ),
                            preparation_assignment_score=annotation.get(
                                "assignmentScore"
                            ),
                            preparation_assignment_score_without_precursor=annotation.get(
                                "assignmentScoreWithoutPrecursor"
                            ),
                            generated_peak_count=sum(value > 0 for value in intensity),
                            generated_fragment_nodes=int(raw.sum()),
                            unique_fragment_nodes=int(nodes.sum()),
                            mean_path_depth=path_depth,
                            duplicate_states_removed=(
                                float(output.selection.decoded.duplicate_states_removed)
                                if output.selection.decoded.duplicate_states_removed
                                is not None
                                else 0.0
                            ),
                            generated_mz=mz,
                            generated_intensity=intensity,
                            generated_peaks=generated,
                            original_peaks=observed,
                            depth_metrics=depth_metrics,
                            **top_metrics,
                        )
                    )
            if progress is not None:
                progress(
                    piece_index,
                    "end",
                    float(np.mean([r["cosine_similarity"] for r in records])),
                )
    finally:
        generator.train(was_training)
    if not records:
        raise ValueError("No validation spectra")
    ce = _ce_categories(records)
    cosine = [record["cosine_similarity"] for record in records]
    stats = distribution(cosine)
    hist, edges = np.histogram(cosine, bins=np.linspace(0, 1, 21))
    cosine_quantiles = {
        "min": float(np.min(cosine)),
        "p10": stats["q10"],
        "p25": stats["q25"],
        "median": stats["median"],
        "p75": stats["q75"],
        "p90": stats["q90"],
        "max": float(np.max(cosine)),
        **{name: stats[name] for name, _ in QUANTILES},
    }
    summary = dict(
        samples=len(records),
        cosine_mean=stats["mean"],
        cosine_std=float(np.std(cosine)),
        cosine_quantiles=cosine_quantiles,
        histogram_counts=hist.tolist(),
        histogram_edges=edges.tolist(),
        nonempty_spectrum_fraction=sum(r["generated_peak_count"] > 0 for r in records)
        / len(records),
        collision_energy_tertiles=ce,
    )
    payload = dict(
        schema="fredis.spectrum-validation",
        mode="free-running spectrum generation",
        summary=summary,
        representatives=_representatives(records),
        spectra=records,
    )
    target_dir = Path(output_dir) / "spectrum_validation"
    target_dir.mkdir(exist_ok=True, parents=True)
    (target_dir / f"{label}.json").write_text(json.dumps(payload, indent=2))
    if global_step is not None:
        split = (
            "validation" if label.startswith("epoch_") else "intermediate_validation"
        )
        append_metric_distributions(
            output_dir, global_step, split, _report_metrics(records), epoch=epoch
        )
    mean = lambda values: (distribution(values) or {"mean": 0.0})["mean"]
    top_n_summary = {
        f"peak_{kind}_top{n}{suffix}": mean(
            r[f"peak_{kind}_top{n}{suffix}"] for r in records
        )
        for kind in ("recall", "precision")
        for n in TOP_N_VALUES
        for suffix in ("", "_without_precursor")
    }
    return dict(
        spectrum_cosine_similarity=summary["cosine_mean"],
        spectrum_cosine_std=summary["cosine_std"],
        spectrum_cosine_similarity_without_precursor=mean(
            r["cosine_similarity_without_precursor"] for r in records
        ),
        assignment_score=mean(r["assignment_score"] for r in records),
        assignment_score_without_precursor=mean(
            r["assignment_score_without_precursor"] for r in records
        ),
        spectrum_nonempty_fraction=summary["nonempty_spectrum_fraction"],
        generated_fragment_nodes=float(
            np.mean([r["generated_fragment_nodes"] for r in records])
        ),
        unique_fragment_nodes=float(
            np.mean([r["unique_fragment_nodes"] for r in records])
        ),
        mean_generated_fragment_nodes=float(
            np.mean([r["unique_fragment_nodes"] for r in records])
        ),
        mean_path_depth=float(np.mean([r["mean_path_depth"] for r in records])),
        duplicate_states_removed=float(
            np.mean([r["duplicate_states_removed"] for r in records])
        ),
        **top_n_summary,
    )
