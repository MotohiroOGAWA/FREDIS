"""Free-running validation and report-ready spectrum diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fredis.libs.mmkit.mmkit import Adduct

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
        result[f"peak_recall_top{n:02d}{suffix}"] = (
            (len(matched) / len(assigned)) if assigned else None
        )
        result[f"peak_precision_top{n:02d}{suffix}"] = (
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
    if spectrum_batch.canonical_data is not None:
        core = spectrum_batch.canonical_data
        group = int(core.sample_branch_group_index[sample])
        sample_nodes = set(
            core.execution_graph.compound_index[
                core.execution_graph.branch_group_index == group
            ].tolist()
        )
    else:
        sample_nodes = set(
            torch.nonzero(decoded.node_sample_index == sample, as_tuple=False)
            .flatten()
            .tolist()
        )
    candidate_rows = (
        torch.nonzero(spectrum_batch.state_sample_index == sample, as_tuple=False)
        .flatten()
        .tolist()
    )
    seeds = {
        int(spectrum_batch.state_node_index[row])
        for row in candidate_rows
        if tolerance.within(float(spectrum_batch.state_mz[row]), float(precursor_mz))
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
        torch.nonzero(spectrum_batch.state_sample_index == sample, as_tuple=False)
        .flatten()
        .tolist()
    ):
        node = int(spectrum_batch.state_node_index[row])
        if tolerance.within(float(spectrum_batch.state_mz[row]), float(precursor_mz)):
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
            for node in match.get("nodeIds", match.get("nodeIndices", ()))
            if node in node_depths
        ]
        result.append(min(candidates) if candidates else None)
    return result


def ion_assignment_metrics(
    batch, sample, annotation, intensities, *, retained_rows=None
):
    """Evaluate the top ion state conditional on an annotated fragment identity.

    Ambiguous annotations remain sets of allowed (fragment, adduct, shift)
    tuples. Accuracy is weighted by the selected SMILES fragment's total
    retained-state predicted intensity. Missing fragments affect coverage, not
    conditional accuracy.
    """
    names = (
        "fragment_ion_adduct_accuracy",
        "hydrogen_shift_accuracy",
        "ion_assignment_joint_accuracy",
    )
    values = {name: [] for name in names}
    core = batch.canonical_data
    candidates = {}
    fragment_weights = {}
    weights = []
    for row in torch.nonzero(batch.state_sample_index == sample).flatten().tolist():
        if retained_rows is not None and row not in retained_rows:
            continue
        node = int(batch.state_node_index[row])
        smiles = (
            core.compounds.smiles[node] if core is not None else batch.node_smiles[node]
        )
        candidate = (
            float(intensities[row]),
            batch.ion_type_strs[int(batch.state_adduct_index[row])],
            int(batch.state_hydrogen_shift[row]),
        )
        fragment_weights[smiles] = fragment_weights.get(smiles, 0.0) + candidate[0]
        if smiles not in candidates or candidate[0] > candidates[smiles][0]:
            candidates[smiles] = candidate
    eligible = evaluated = 0
    for peak in annotation["peaks"]:
        if peak.get("precursor"):
            continue
        allowed = {
            (m["smiles"], m["fragmentIonAdduct"], int(m["hydrogenShift"]))
            for m in peak.get("matches", ())
            if all(k in m for k in ("smiles", "fragmentIonAdduct", "hydrogenShift"))
        }
        if not allowed:
            continue
        eligible += 1
        available = {smiles for smiles, _, _ in allowed if smiles in candidates}
        if not available:
            continue
        # Select one state before checking its labels; never pick the best label match.
        smiles = max(sorted(available), key=lambda key: candidates[key][0])
        _, adduct, shift = candidates[smiles]
        labels = {(a, h) for sm, a, h in allowed if sm == smiles}
        evaluated += 1
        weights.append(fragment_weights[smiles])
        values[names[0]].append(float(any(a == adduct for a, _ in labels)))
        values[names[1]].append(float(any(h == shift for _, h in labels)))
        values[names[2]].append(float((adduct, shift) in labels))
    total_weight = sum(weights)
    result = {
        name: (
            sum(weight * correct for weight, correct in zip(weights, outcomes))
            / total_weight
            if total_weight > 0
            else None
        )
        for name, outcomes in values.items()
    }
    result["ion_assignment_coverage"] = evaluated / eligible if eligible else None
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


def _compact_spectra(records):
    """Keep five quantile examples per metric, deduplicating shared spectra."""
    representatives = _representatives(records)
    spectra = []
    indices = {}
    for entries in representatives.values():
        for item in entries.values():
            source_index = item["spectrum_index"]
            if source_index not in indices:
                indices[source_index] = len(spectra)
                spectra.append(records[source_index])
            item["source_spectrum_index"] = source_index
            item["spectrum_index"] = indices[source_index]
    return representatives, spectra


def append_metric_distributions(output_dir, global_step, split, metrics, *, epoch=None):
    """Append q10/q25/median/q75/q90/mean series to per-chart TSV files."""
    rows = []
    for metric, values in metrics.items():
        if metric.startswith("branch_count_"):
            continue
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
    from .metric_store import write_series_rows

    write_series_rows(
        output_dir,
        (
            (f"distributions/{item_split}/{metric}", step, item_epoch, "", value)
            for step, item_epoch, item_split, metric, value in rows
        ),
    )



def _report_metrics(records):
    metrics = {}
    depth_names = (
        "cosine_similarity",
        "cosine_similarity_without_precursor",
        "assignment_score",
        "assignment_score_without_precursor",
        "fragment_ion_adduct_accuracy",
        "hydrogen_shift_accuracy",
        "ion_assignment_joint_accuracy",
        "ion_assignment_coverage",
    )
    # Top-N recall/precision are already a coarse, whole-spectrum ranking; a
    # per-depth split of a handful of candidates isn't meaningful, so they get
    # the flat + collision-energy/adduct facets only, not the depth facet.
    top_n_names = tuple(
        f"peak_{kind}_top{n:02d}{suffix}"
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
    generator,
    pieces,
    output_dir,
    label,
    *,
    global_step=None,
    progress=None,
    epoch=None,
    status=None,
):
    was_training = generator.training
    generator.eval()
    records = []
    tolerance = generator.fragmenter.mass_tolerance
    def generated_outputs(data):
        iterator = iter(generator.predict_prepared_batches(data))
        while True:
            if status is not None:
                status("generate")
            try:
                output = next(iterator)
            except StopIteration:
                return
            if status is not None:
                status("score")
            yield output

    try:
        for piece_index, data in enumerate(pieces):
            if progress is not None:
                progress(piece_index, "start")
            if len(data.sample_annotations) != data.num_samples:
                raise ValueError(
                    "Spectrum validation requires original peak annotations; regenerate structures"
                )
            annotations = data.sample_annotations
            for output in generated_outputs(data):
                for sample, input_index in enumerate(output.sample_input_index):
                    annotation = annotations[input_index]
                    threshold = generator.generation.peak_intensity_threshold
                    maximum = max(
                        (float(p["intensity"]) for p in annotation["peaks"]),
                        default=0.0,
                    )
                    annotation = {
                        **annotation,
                        "peaks": [
                            p
                            for p in annotation["peaks"]
                            if float(p["intensity"]) > maximum * threshold
                        ],
                    }

                    formula_rows = (
                        torch.nonzero(
                            output.spectrum_batch.state_sample_index == sample,
                            as_tuple=False,
                        )
                        .flatten()
                        .tolist()
                    )
                    mz = [
                        float(output.spectrum_batch.state_mz[row])
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
                    row_keys = {}
                    for row, peak in zip(formula_rows, generated):
                        key = tuple(
                            output.spectra.formula_tensor[row].detach().cpu().tolist()
                        )
                        if any(v < 0 for v in key[:-1]) or peak["mz"] <= 0:
                            continue
                        row_keys[row] = key
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
                    # Match inference export: merge physical formulas, then filter.
                    grouped = {
                        key: peak
                        for key, peak in grouped.items()
                        if peak["intensity"] > threshold
                    }
                    retained_rows = {
                        row for row, key in row_keys.items() if key in grouped
                    }
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
                        depth_metrics[str(depth)].update(
                            ion_assignment_metrics(
                                output.spectrum_batch,
                                sample,
                                {
                                    **annotation,
                                    "peaks": [
                                        peak
                                        for peak, assigned_depth in zip(
                                            annotation["peaks"], original_depth
                                        )
                                        if assigned_depth == depth
                                    ],
                                },
                                output.spectra.intensity,
                                retained_rows=retained_rows,
                            )
                        )
                    branch_group = int(data.sample_branch_group_index[sample])
                    core = output.spectrum_batch.canonical_data
                    group = int(core.sample_branch_group_index[sample])
                    execution = core.execution_graph
                    raw = execution.branch_group_index == group
                    node_count = execution.compound_index[raw].unique().numel()
                    path_depth = (
                        float(core.state_reaction_count[raw].float().mean())
                        if bool(raw.any())
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
                            f"peak_{kind}_top{n:02d}_without_precursor": None
                            for kind in ("recall", "precision")
                            for n in TOP_N_VALUES
                        }
                    )
                    records.append(
                        dict(
                            **ion_assignment_metrics(
                                output.spectrum_batch,
                                sample,
                                annotation,
                                output.spectra.intensity,
                                retained_rows=retained_rows,
                            ),
                            smiles=data.compounds.smiles[
                                int(data.sample_tree_index[input_index])
                            ],
                            piece_index=piece_index,
                            sample_index=input_index,
                            adduct=annotation["adduct"],
                            main_adduct=annotation.get(
                                "mainAdduct",
                                str(
                                    generator.fragmenter._resolve_main_adduct_type(
                                        Adduct.parse(annotation["adduct"])
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
                            unique_fragment_nodes=int(node_count),
                            mean_path_depth=path_depth,
                            duplicate_states_removed=0.0,
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
    if status is not None:
        status("save")
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
    representatives, saved_spectra = _compact_spectra(records)
    summary["saved_spectra"] = len(saved_spectra)
    payload = dict(
        schema="fredis.spectrum-validation",
        mode="free-running spectrum generation",
        summary=summary,
        representatives=representatives,
        spectra=saved_spectra,
    )
    target_dir = Path(output_dir) / "spectrum_validation"
    target_dir.mkdir(exist_ok=True, parents=True)
    from .validation_storage import save_validation_msdataset

    save_validation_msdataset(
        saved_spectra,
        representatives,
        target_dir / f"{label}.msds",
        epoch=epoch,
        global_step=global_step,
        summary=summary,
    )
    payload.update(epoch=epoch, global_step=global_step, msds_file=f"{label}.msds")
    result_path = target_dir / f"{label}.json"
    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(json.dumps(payload))
    temporary_result.replace(result_path)
    if global_step is not None:
        split = (
            "validation" if label.startswith("epoch_") else "intermediate_validation"
        )
        append_metric_distributions(
            output_dir, global_step, split, _report_metrics(records), epoch=epoch
        )
    mean = lambda values: (distribution(values) or {"mean": 0.0})["mean"]
    top_n_summary = {
        f"peak_{kind}_top{n:02d}{suffix}": mean(
            r[f"peak_{kind}_top{n:02d}{suffix}"] for r in records
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
