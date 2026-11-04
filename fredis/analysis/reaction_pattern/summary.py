"""Sample-level node/edge rows and their compound and node distributions.

Distributions are always aggregated from the saved sample-level rows
(``sample_node_summary``), so they can be recomputed and verified.  The
"mean" of a compound x node is the mean of its samples' values; the
"weighted" value instead divides compound totals (peaks or intensity).

Rows per node: a root has a row for every compound and absolute metrics; an
edge has rows only for compounds whose precursor matches a pattern its set
adds, with parent, child (absolute) and delta metrics.  ``overall_*`` node
columns instead cover every sample of the node's population, including
compounds for which the child condition equals the parent condition.
"""

from __future__ import annotations

from collections import defaultdict
import dataclasses
import json
import math
from typing import Iterable

import numpy as np

from .comparison import CompoundComparison
from .constants import (
    COMPLETED,
    COVERAGE_METRICS,
    DISTRIBUTION_METRICS,
    EDGE_METRICS,
    FAILED,
    HISTOGRAM_BINS,
    HISTOGRAM_RANGE,
    SKIPPED,
)

WITHOUT_SUFFIX = "_without_precursor"
SUFFIXES = (("", 0), (WITHOUT_SUFFIX, 1))
SAMPLE_METRICS = (
    "assigned_peak_count",
    "assigned_intensity",
    "peak_coverage",
    "intensity_coverage",
    "parent_assigned_peak_count",
    "parent_assigned_intensity",
    "parent_peak_coverage",
    "parent_intensity_coverage",
    "newly_assigned_peak_count",
    "newly_assigned_intensity",
    "delta_peak_coverage",
    "delta_intensity_coverage",
)
MEAN_METRICS = (
    "peak_coverage",
    "intensity_coverage",
    "parent_peak_coverage",
    "parent_intensity_coverage",
    "delta_peak_coverage",
    "delta_intensity_coverage",
)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else math.nan


def distribution(
    values: Iterable[float], metric: str, *, bins: int = HISTOGRAM_BINS
) -> dict:
    """n/mean/std/min/quartiles/max and a histogram of finite values.

    ``std`` is the population standard deviation (ddof=0).  Coverage metrics
    use a fixed 0-1 range; count/intensity metrics use 0..max.
    """
    array = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=float,
    )
    if metric in COVERAGE_METRICS:
        low, high = HISTOGRAM_RANGE
    else:
        low, high = 0.0, float(array.max()) if array.size and array.max() > 0 else 1.0
    counts, edges = np.histogram(
        np.clip(array, low, high), bins=bins, range=(low, high)
    )
    histogram = json.dumps(
        {
            "edges": [round(float(edge), 12) for edge in edges],
            "counts": counts.astype(int).tolist(),
        }
    )
    if not array.size:
        nan = math.nan
        return dict(
            n=0,
            mean=nan,
            std=nan,
            min=nan,
            q25=nan,
            median=nan,
            q75=nan,
            max=nan,
            histogram_json=histogram,
        )
    q25, median, q75 = np.percentile(array, [25, 50, 75])
    return dict(
        n=int(array.size),
        mean=float(array.mean()),
        std=float(array.std()),
        min=float(array.min()),
        q25=float(q25),
        median=float(median),
        q75=float(q75),
        max=float(array.max()),
        histogram_json=histogram,
    )


class Tables:
    """Column-oriented rows destined for one HDF5 table each."""

    def __init__(self):
        self.tables: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def add(self, table: str, **row) -> None:
        columns = self.tables[table]
        for key, value in row.items():
            columns[key].append(value)

    def get(self, table: str) -> dict[str, list]:
        return self.tables[table]


def _metrics_for(is_root: bool) -> tuple[str, ...]:
    return tuple(
        name for name in DISTRIBUTION_METRICS if not is_root or name not in EDGE_METRICS
    )


def _sample_row(item, metric_source: dict, suffix: str, is_root: bool) -> dict:
    row = {}
    for name in SAMPLE_METRICS:
        value = metric_source.get(name, math.nan)
        row[name + suffix] = float(value) if value is not None else math.nan
    return row


FORMULA_CATEGORIES = ("both", "formula_only", "pattern_only", "neither")


def formula_node_totals(comparisons: list[CompoundComparison]) -> dict[int, dict] | None:
    """Peaks and intensity of every node by formula x pattern assignment.

    Only samples whose formula assignment succeeded count; None when formula
    assignment was not requested.
    """
    if not any(
        sample.formula_status
        for comparison in comparisons
        for sample in comparison.group.samples
    ):
        return None
    totals: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    for comparison in comparisons:
        for item in comparison.nodes:
            if not item.evaluated or not (item.is_root or item.applicable):
                continue
            aggregate = totals[item.node.node_row]
            for result in item.samples:
                sample = result.sample
                if sample.formulas is None:
                    continue
                aggregate["formula_sample_count"] += 1
                formula = np.asarray([bool(value) for value in sample.formulas], dtype=bool)
                pattern = np.asarray(result.assigned, dtype=bool)
                categories = {
                    "both": formula & pattern,
                    "formula_only": formula & ~pattern,
                    "pattern_only": ~formula & pattern,
                    "neither": ~formula & ~pattern,
                }
                for suffix, flag in SUFFIXES:
                    mask = ~sample.is_precursor if flag else np.ones(len(sample.mz), dtype=bool)
                    aggregate["peaks" + suffix] += int(mask.sum())
                    aggregate["intensity" + suffix] += float(sample.intensity[mask].sum())
                    for name, selected in categories.items():
                        aggregate[f"{name}_peaks{suffix}"] += int((selected & mask).sum())
                        aggregate[f"{name}_intensity{suffix}"] += float(
                            sample.intensity[selected & mask].sum()
                        )
    return totals


def formula_node_row(aggregate: dict | None) -> dict:
    """node_summary columns: ``formula_<category>_{peak,intensity}_fraction[_without_precursor]``."""
    aggregate = aggregate or {}
    row = {"formula_sample_count": int(aggregate.get("formula_sample_count", 0))}
    for suffix, _ in SUFFIXES:
        for name in FORMULA_CATEGORIES:
            row[f"formula_{name}_peak_count{suffix}"] = int(aggregate.get(f"{name}_peaks{suffix}", 0))
            row[f"formula_{name}_peak_fraction{suffix}"] = _ratio(
                aggregate.get(f"{name}_peaks{suffix}", 0), aggregate.get("peaks" + suffix, 0)
            )
            row[f"formula_{name}_intensity_fraction{suffix}"] = _ratio(
                aggregate.get(f"{name}_intensity{suffix}", 0),
                aggregate.get("intensity" + suffix, 0),
            )
    return row


def restrict_to_dataset(
    comparison: CompoundComparison, dataset_id: int
) -> CompoundComparison | None:
    """The comparison seen through one dataset's spectra only; None without any.

    Condition outcomes are shared, so only the evaluated sample positions
    (``kept``, ``population`` and node sample rows) are narrowed.
    """
    group = comparison.group
    keep = {
        index
        for index, sample in enumerate(group.samples)
        if sample.dataset_id == dataset_id
    }
    if not keep:
        return None
    narrowed = {}

    def outcome(value):
        if value is None:
            return None
        if id(value) not in narrowed:
            narrowed[id(value)] = dataclasses.replace(
                value, kept=tuple(index for index in value.kept if index in keep)
            )
        return narrowed[id(value)]

    nodes = [
        dataclasses.replace(
            item,
            outcome=outcome(item.outcome),
            parent_outcome=outcome(item.parent_outcome),
            samples=[
                result for result in item.samples if result.sample.dataset_id == dataset_id
            ],
        )
        for item in comparison.nodes
    ]
    return CompoundComparison(
        group,
        nodes,
        {key: outcome(value) for key, value in comparison.outcomes.items()},
        tuple(index for index in comparison.population if index in keep),
        comparison.matched_patterns,
    )


def _rows(columns: dict[str, list]) -> list[dict]:
    names = list(columns)
    return [dict(zip(names, values)) for values in zip(*(columns[name] for name in names))]


PATTERN_COUNTS = ("matched", "evaluated", "reacted", "new")
UNKNOWN = -1
HIT_COLUMNS = (
    "sample_id", "compound_row", "dataset_id", "node_row", "pattern_id", "rule_id",
    "evaluated", "reacted", "new",
)


def pattern_rules(fragmenter, tree) -> dict[int, list[tuple[int, str, str]]]:
    """Every product rule ``(rule id, name, product SMARTS)`` of the patterns the nodes add."""
    used = sorted({pid for node in tree.nodes for pid in node.added_pattern_ids})
    pattern_set = fragmenter.fragment_ion_tree_builder.reaction_pattern_set
    result = {}
    for pid in used:
        pattern = pattern_set.get_pattern(int(pid))
        result[int(pid)] = [
            (int(rule.id), str(rule.source_rule.name), str(rule.source_rule.smarts))
            for rule in sorted(pattern.reaction_rules, key=lambda item: item.id)
        ]
    return result


def complete_instance_rules(fragmenter, smiles: str, matched, known: dict) -> dict:
    """Rules with a reaction instance on the precursor for every matched pattern.

    Workers report the patterns they enumerated; the others (e.g. conditions
    reused from an existing result) are enumerated here, on the precursor only.
    """
    result = {int(key): frozenset(value) for key, value in (known or {}).items()}
    missing = [int(pid) for pid in matched if int(pid) not in result]
    if missing:
        from fredis.libs.mmkit.mmkit import Compound

        instances = fragmenter.fragment_ion_tree_builder.create_reaction_instances(
            Compound.from_smiles(smiles), pattern_ids=frozenset(missing)
        )
        for pid in missing:
            result[pid] = frozenset(
                int(item.reaction_rule_id) for item in instances if item.reaction_pattern_id == pid
            )
    return result


def write_metadata(output, comparisons) -> None:
    """``samples.metadata.parquet``: every input column of every analysed spectrum."""
    import pandas as pd
    from .constants import METADATA_NAME

    rows = []
    for comparison in comparisons:
        for index in comparison.population:
            sample = comparison.group.samples[index]
            row = {
                key: (None if value is None else str(value))
                for key, value in (sample.metadata or {}).items()
                if key not in ("sample_id", "dataset_id", "record_index")
            }
            rows.append(dict(sample_id=int(sample.sample_id), dataset_id=int(sample.dataset_id),
                             record_index=int(sample.record_index), **row))
    frame = pd.DataFrame(rows)
    if not len(frame):
        frame = pd.DataFrame({name: pd.Series(dtype="int64") for name in ("sample_id", "dataset_id", "record_index")})
    target = output / METADATA_NAME
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(target)


def pattern_hits(tables: Tables, comparisons) -> None:
    """``sample_pattern_hits``: one row per spectrum x node x introduced, matched pattern.

    ``rule_id`` is -1 for the pattern itself; each product rule with a
    reaction instance on the precursor has its own row.  ``evaluated``: the
    node's condition was calculated for the spectrum; ``reacted``: the
    pattern (product) explains an assigned peak; ``new``: it explains a peak
    the parent condition does not assign.  -1 = unknown (``new`` of a root).
    """
    for compound_row, comparison in enumerate(comparisons):
        population = comparison.population
        if not population:
            continue
        group = comparison.group
        for item in comparison.nodes:
            node = item.node
            matched = [pid for pid in node.added_pattern_ids if pid in comparison.matched_patterns]
            if not matched:
                continue
            results = {
                result.sample.sample_index: result for result in (item.samples if item.evaluated else ())
            }
            for index in population:
                sample = group.samples[index]
                result = results.get(sample.sample_index)
                for pid in matched:
                    keys = [(pid, -1, None)] + [
                        (pid, rule, (pid, rule))
                        for rule in sorted((comparison.instance_rules or {}).get(pid, ()))
                    ]
                    for pattern_id, rule_id, rule_key in keys:
                        if result is None:
                            reacted, new = 0, (UNKNOWN if item.is_root else 0)
                        else:
                            masks = result.patterns if rule_key is None else result.rules
                            mask = None if masks is None else masks.get(pid if rule_key is None else rule_key)
                            if masks is None:
                                reacted = new = UNKNOWN
                            else:
                                reacted = int(mask is not None and bool((result.assigned & mask).any()))
                                new = (
                                    int(mask is not None and result.new is not None and bool((result.new & mask).any()))
                                    if not item.is_root else UNKNOWN
                                )
                            if item.is_root:
                                new = UNKNOWN
                        tables.add(
                            "sample_pattern_hits",
                            sample_id=int(sample.sample_id),
                            compound_row=compound_row,
                            dataset_id=int(sample.dataset_id),
                            node_row=int(node.node_row),
                            pattern_id=int(pattern_id),
                            rule_id=int(rule_id),
                            evaluated=int(result is not None),
                            reacted=int(reacted),
                            new=int(new),
                        )


def aggregate_pattern_hits(hits, population, nodes, rules_by_pattern) -> list[dict]:
    """``pattern_summary`` rows of one scope from its hit rows and analysed spectra.

    ``hits``/``population`` are DataFrames (``population``: sample_id,
    compound_row).  One row per node (``pattern_id`` = -1: the whole pattern
    set it introduces), per introduced pattern (``rule_id`` = -1) and per
    product rule.  Counts are -1 when unknown or not applicable.
    """
    import pandas as pd

    compound_count = int(population["compound_row"].nunique()) if len(population) else 0
    sample_count = int(len(population))
    hits = hits if len(hits) else pd.DataFrame({name: pd.Series(dtype="int64") for name in HIT_COLUMNS})
    patterns = hits[hits["rule_id"] < 0]
    # Pattern-set level: a spectrum counts once per node, when any pattern does.
    per_sample = patterns.assign(
        unknown_reacted=(patterns["evaluated"] > 0) & (patterns["reacted"] < 0),
        unknown_new=(patterns["evaluated"] > 0) & (patterns["new"] < 0),
    ).groupby(["node_row", "sample_id", "compound_row"], as_index=False).agg(
        evaluated=("evaluated", "max"), reacted=("reacted", "max"), new=("new", "max"),
        unknown_reacted=("unknown_reacted", "any"), unknown_new=("unknown_new", "any"))
    per_sample["pattern_id"] = -1
    per_sample["rule_id"] = -1
    frames = pd.concat([
        per_sample,
        hits.assign(unknown_reacted=(hits["evaluated"] > 0) & (hits["reacted"] < 0),
                    unknown_new=(hits["evaluated"] > 0) & (hits["new"] < 0)),
    ], ignore_index=True)
    stats = {}
    for key, rows in frames.groupby(["node_row", "pattern_id", "rule_id"], sort=False):
        evaluated = rows[rows["evaluated"] > 0]
        reacted = evaluated[evaluated["reacted"] > 0]
        new = evaluated[evaluated["new"] > 0]
        stats[tuple(int(value) for value in key)] = dict(
            matched_compound_count=int(rows["compound_row"].nunique()),
            matched_sample_count=int(len(rows)),
            evaluated_compound_count=int(evaluated["compound_row"].nunique()),
            evaluated_sample_count=int(len(evaluated)),
            reacted_compound_count=UNKNOWN if rows["unknown_reacted"].any() else int(reacted["compound_row"].nunique()),
            reacted_sample_count=UNKNOWN if rows["unknown_reacted"].any() else int(len(reacted)),
            new_compound_count=UNKNOWN if rows["unknown_new"].any() else int(new["compound_row"].nunique()),
            new_sample_count=UNKNOWN if rows["unknown_new"].any() else int(len(new)),
        )
    result = []
    for node in nodes:
        is_root = node.parent_node_id is None
        keys = [(node.node_row, -1, -1)]
        for pid in node.added_pattern_ids:
            keys.append((node.node_row, int(pid), -1))
            keys.extend((node.node_row, int(pid), int(rule)) for rule, *_ in rules_by_pattern.get(int(pid), ()))
        for key in keys:
            row = dict(node_row=key[0], pattern_id=key[1], rule_id=key[2],
                       compound_count=compound_count, sample_count=sample_count)
            values = stats.get(key, {})
            for name in PATTERN_COUNTS:
                for unit in ("compound", "sample"):
                    column = f"{name}_{unit}_count"
                    row[column] = UNKNOWN if (name == "new" and is_root) else int(values.get(column, 0))
            result.append(row)
    return result


def _pattern_tables(tables: Tables, comparisons, nodes, datasets, rules_by_pattern) -> None:
    import pandas as pd

    pattern_hits(tables, comparisons)
    for pid, rules in sorted(rules_by_pattern.items()):
        for rule_id, name, smarts in rules:
            tables.add("pattern_rules", pattern_id=int(pid), rule_id=int(rule_id), name=name, product_smarts=smarts)
    hits = pd.DataFrame({name: tables.get("sample_pattern_hits").get(name, []) for name in HIT_COLUMNS})
    columns = tables.get("sample_summary")
    population = pd.DataFrame({name: columns.get(name, []) for name in ("sample_id", "compound_row", "dataset_id")})
    for row in aggregate_pattern_hits(hits, population, nodes, rules_by_pattern):
        tables.add("pattern_summary", **row)
    if len(datasets) > 1:
        for dataset in datasets:
            dataset_id = int(dataset["dataset_id"])
            scoped = aggregate_pattern_hits(
                hits[hits["dataset_id"] == dataset_id],
                population[population["dataset_id"] == dataset_id],
                nodes,
                rules_by_pattern,
            )
            for row in scoped:
                tables.add("dataset_pattern_summary", dataset_id=dataset_id, **row)


def _dataset_tables(tables: Tables, comparisons, *, tree, datasets, with_formula: bool) -> None:
    """``dataset_*`` tables: node summaries, distributions and overall per dataset."""
    for dataset in datasets:
        dataset_id = int(dataset["dataset_id"])
        subset = [
            value
            for value in (restrict_to_dataset(item, dataset_id) for item in comparisons)
            if value is not None
        ]
        scoped, overall = summarize(
            subset, tree=tree, datasets=[dataset], by_dataset=False
        )
        for row in _rows(scoped.get("node_summary")):
            if with_formula and "formula_sample_count" not in row:
                row.update(formula_node_row(None))
            tables.add("dataset_node_summary", dataset_id=dataset_id, **row)
        for row in _rows(scoped.get("node_distribution")):
            tables.add("dataset_node_distribution", dataset_id=dataset_id, **row)
        tables.add(
            "dataset_overall_summary",
            dataset_id=dataset_id,
            summary_json=json.dumps(overall, ensure_ascii=False),
        )


def summarize(
    comparisons: list[CompoundComparison],
    *,
    tree,
    datasets: list[dict],
    relative_path=lambda value: value,
    by_dataset: bool = True,
    rules: dict | None = None,
) -> tuple[Tables, dict]:
    """Return every A table plus the overall summary document.

    With more than one dataset, ``dataset_*`` tables repeat the node summary,
    node distributions and overall summary for each dataset's spectra alone;
    the unprefixed tables stay the total over every dataset.
    """
    tables = Tables()
    nodes = tree.nodes
    formula_totals = formula_node_totals(comparisons)
    node_samples: dict[int, list[dict]] = defaultdict(list)
    node_totals: dict[int, dict] = {node.node_row: defaultdict(float) for node in nodes}
    dataset_stats = {int(row["dataset_id"]): defaultdict(int) for row in datasets}
    dataset_compounds: dict[int, set] = defaultdict(set)
    overall = defaultdict(float)
    improved_samples: set[int] = set()
    improved_compounds: set[str] = set()

    for compound_row, comparison in enumerate(comparisons):
        group = comparison.group
        compound_id = group.compound_id
        population = comparison.population
        totals = defaultdict(float)
        for index in population:
            sample = group.samples[index]
            everything = len(sample.mz)
            without = ~sample.is_precursor
            values = dict(
                peak_count=everything,
                total_intensity=float(sample.coverage_intensity.sum()),
                peak_count_without_precursor=int(without.sum()),
                total_intensity_without_precursor=float(
                    sample.coverage_intensity[without].sum()
                ),
            )
            tables.add(
                "sample_summary",
                sample_id=sample.sample_id,
                compound_id=compound_id,
                compound_row=compound_row,
                dataset_id=sample.dataset_id,
                record_index=sample.record_index,
                sample_index=sample.sample_index,
                adduct=sample.adduct,
                collision_energy=sample.collision_energy,
                precursor_mz=sample.precursor_mz,
                **values,
            )
            for key, value in values.items():
                totals[key] += value
            stats = dataset_stats.setdefault(sample.dataset_id, defaultdict(int))
            stats["sample_count"] += 1
            stats["peak_count"] += everything
            dataset_compounds[sample.dataset_id].add(compound_id)

        samples_by_index = {sample.sample_index: sample for sample in group.samples}
        union_new = {
            group.samples[index].sample_index: np.zeros(len(group.samples[index].mz), dtype=bool)
            for index in population
        }
        evaluated_nodes = improving_edges = 0
        applicable_rows = []
        for item in comparison.nodes:
            node, outcome, parent = item.node, item.outcome, item.parent_outcome
            is_root = parent is None
            aggregate = node_totals[node.node_row]
            # Absolute coverage of this node's condition over its whole population.
            effective = outcome if (is_root or item.applicable) else parent
            if effective.completed and (is_root or parent.completed):
                for index in effective.kept:
                    sample = group.samples[index]
                    for suffix, flag in SUFFIXES:
                        mask = (
                            np.ones(len(sample.mz), dtype=bool)
                            if not flag
                            else ~sample.is_precursor
                        )
                        assigned = effective.assigned[index] & mask
                        aggregate["overall_peaks" + suffix] += int(mask.sum())
                        aggregate["overall_intensity" + suffix] += float(
                            sample.coverage_intensity[mask].sum()
                        )
                        aggregate["overall_assigned_peaks" + suffix] += int(
                            assigned.sum()
                        )
                        aggregate["overall_assigned_intensity" + suffix] += float(
                            sample.coverage_intensity[assigned].sum()
                        )
                aggregate["overall_compound_count"] += 1
                aggregate["overall_sample_count"] += len(effective.kept)
            if not is_root and not item.applicable:
                continue
            if not is_root:
                applicable_rows.append(str(node.node_row))
            rows = []
            for result in item.samples:
                sample = result.sample
                row = dict(
                    sample_id=sample.sample_id,
                    compound_row=compound_row,
                    node_row=node.node_row,
                )
                for suffix, flag in SUFFIXES:
                    row.update(
                        _sample_row(
                            result,
                            result.metrics_without if flag else result.metrics,
                            suffix,
                            is_root,
                        )
                    )
                tables.add("sample_node_summary", **row)
                rows.append(row)
                node_samples[node.node_row].append(row)
                if result.new is not None:
                    union_new[sample.sample_index] |= result.new
                    if row["newly_assigned_peak_count"] > 0:
                        improved_samples.add(sample.sample_id)
            evaluated_nodes += int(item.evaluated)
            summed = lambda name: float(
                sum(row[name] for row in rows if math.isfinite(row[name]))
            )
            sample_totals = {
                key
                + suffix: float(
                    sum(
                        result.metrics_without[key] if flag else result.metrics[key]
                        for result in item.samples
                    )
                )
                for key in ("peaks", "intensity")
                for suffix, flag in SUFFIXES
            }
            newly = int(summed("newly_assigned_peak_count")) if not is_root else 0
            improved = (
                sum(1 for row in rows if (row["newly_assigned_peak_count"] or 0) > 0)
                if not is_root
                else 0
            )
            improving_edges += int(newly > 0)
            means = {}
            for metric in _metrics_for(is_root):
                for suffix, flag in SUFFIXES:
                    if not rows:
                        continue
                    stats = distribution((row[metric + suffix] for row in rows), metric)
                    means[metric + suffix] = stats["mean"]
                    tables.add(
                        "compound_node_distribution",
                        compound_id=compound_id,
                        compound_row=compound_row,
                        node_row=node.node_row,
                        metric=metric,
                        without_precursor=flag,
                        **stats,
                    )
            summary_row = dict(
                compound_id=compound_id,
                compound_row=compound_row,
                node_row=node.node_row,
                is_root=is_root,
                applicable=bool(item.applicable),
                status=outcome.status,
                skip_category=outcome.skip_category,
                reason=outcome.reason,
                error=outcome.error,
                sample_count=len(rows),
                improved_sample_count=improved,
            )
            for suffix, flag in SUFFIXES:
                peaks, intensity = (
                    sample_totals["peaks" + suffix],
                    sample_totals["intensity" + suffix],
                )
                assigned_peaks, assigned_intensity = summed(
                    "assigned_peak_count" + suffix
                ), summed("assigned_intensity" + suffix)
                summary_row["peak_count" + suffix] = int(peaks)
                summary_row["total_intensity" + suffix] = intensity
                summary_row["assigned_peak_count_total" + suffix] = int(assigned_peaks)
                summary_row["assigned_intensity_total" + suffix] = assigned_intensity
                summary_row["weighted_peak_coverage" + suffix] = (
                    _ratio(assigned_peaks, peaks) if rows else math.nan
                )
                summary_row["weighted_intensity_coverage" + suffix] = (
                    _ratio(assigned_intensity, intensity) if rows else math.nan
                )
                if is_root or not rows:
                    for name in (
                        "parent_assigned_peak_count_total",
                        "parent_assigned_intensity_total",
                        "newly_assigned_peak_count_total",
                        "newly_assigned_intensity_total",
                    ):
                        summary_row[name + suffix] = (
                            0 if name.endswith("count_total") else 0.0
                        )
                    for name in (
                        "parent_weighted_peak_coverage",
                        "parent_weighted_intensity_coverage",
                        "weighted_delta_peak_coverage",
                        "weighted_delta_intensity_coverage",
                    ):
                        summary_row[name + suffix] = math.nan
                else:
                    parent_peaks = summed("parent_assigned_peak_count" + suffix)
                    parent_intensity = summed("parent_assigned_intensity" + suffix)
                    new_peaks = summed("newly_assigned_peak_count" + suffix)
                    new_intensity = summed("newly_assigned_intensity" + suffix)
                    summary_row["parent_assigned_peak_count_total" + suffix] = int(
                        parent_peaks
                    )
                    summary_row["parent_assigned_intensity_total" + suffix] = (
                        parent_intensity
                    )
                    summary_row["newly_assigned_peak_count_total" + suffix] = int(
                        new_peaks
                    )
                    summary_row["newly_assigned_intensity_total" + suffix] = (
                        new_intensity
                    )
                    summary_row["parent_weighted_peak_coverage" + suffix] = _ratio(
                        parent_peaks, peaks
                    )
                    summary_row["parent_weighted_intensity_coverage" + suffix] = _ratio(
                        parent_intensity, intensity
                    )
                    summary_row["weighted_delta_peak_coverage" + suffix] = _ratio(
                        new_peaks, peaks
                    )
                    summary_row["weighted_delta_intensity_coverage" + suffix] = _ratio(
                        new_intensity, intensity
                    )
                for name in MEAN_METRICS:
                    summary_row["mean_" + name + suffix] = means.get(
                        name + suffix, math.nan
                    )
            tables.add("compound_node_summary", **summary_row)

            aggregate["applicable_compound_count"] += 1
            aggregate[
                {
                    COMPLETED: "calculated_compound_count",
                    SKIPPED: "skipped_compound_count",
                    FAILED: "failed_compound_count",
                }.get(outcome.status, "skipped_compound_count")
            ] += 1
            aggregate["evaluated_sample_count"] += len(rows)
            aggregate["improved_compound_count"] += int(newly > 0)
            aggregate["improved_sample_count"] += improved
            if rows:
                aggregate["evaluated_compound_count"] += 1
                for suffix, flag in SUFFIXES:
                    aggregate["peaks" + suffix] += sample_totals["peaks" + suffix]
                    aggregate["intensity" + suffix] += sample_totals[
                        "intensity" + suffix
                    ]
                    for name in (
                        "assigned_peak_count",
                        "assigned_intensity",
                        "parent_assigned_peak_count",
                        "parent_assigned_intensity",
                        "newly_assigned_peak_count",
                        "newly_assigned_intensity",
                    ):
                        aggregate[name + suffix] += summed(name + suffix)

        new_peaks = sum(int(mask.sum()) for mask in union_new.values())
        new_intensity = sum(
            float(samples_by_index[index].coverage_intensity[mask].sum())
            for index, mask in union_new.items()
        )
        if new_peaks:
            improved_compounds.add(compound_id)
        root_outcomes = [item.outcome for item in comparison.nodes if item.is_root]
        status = (
            COMPLETED
            if any(outcome.completed for outcome in root_outcomes)
            else (root_outcomes[0].status if root_outcomes else SKIPPED)
        )
        reason = (
            ""
            if status == COMPLETED
            else (root_outcomes[0].reason if root_outcomes else "")
        )
        tables.add(
            "compounds",
            compound_id=compound_id,
            canonical_smiles=group.canonical_smiles,
            original_smiles=group.original_smiles,
            dataset_count=len(group.dataset_ids),
            dataset_ids=",".join(map(str, group.dataset_ids)),
            sample_count=len(population),
            peak_count=int(totals["peak_count"]),
            total_intensity=totals["total_intensity"],
            peak_count_without_precursor=int(totals["peak_count_without_precursor"]),
            total_intensity_without_precursor=totals[
                "total_intensity_without_precursor"
            ],
            evaluated_node_count=evaluated_nodes,
            applicable_node_rows=",".join(applicable_rows),
            improving_edge_count=improving_edges,
            new_peak_union_count=new_peaks,
            new_intensity_union=new_intensity,
            peak_summary_file=group.peak_summary_file,
            status=status,
            reason=reason,
        )
        for condition_id in sorted(comparison.outcomes):
            outcome = comparison.outcomes[condition_id]
            tables.add(
                "compound_condition",
                compound_id=compound_id,
                compound_row=compound_row,
                condition_id=condition_id,
                status=outcome.status,
                skip_category=outcome.skip_category,
                reason=outcome.reason,
                error=outcome.error,
                kept=",".join(
                    str(group.samples[index].sample_index) for index in outcome.kept
                ),
                detail_path=relative_path(outcome.file) if outcome.file else "",
                seed_path=relative_path(outcome.seed_file) if outcome.seed_file else "",
                alias_of=-1 if outcome.alias_of is None else int(outcome.alias_of),
            )
        for key in (
            "peak_count",
            "total_intensity",
            "peak_count_without_precursor",
            "total_intensity_without_precursor",
        ):
            overall[key] += totals[key]
        overall["samples"] += len(population)
        overall["new_peaks"] += new_peaks
        overall["new_intensity"] += new_intensity

    for node in nodes:
        is_root = node.parent_node_id is None
        aggregate = node_totals[node.node_row]
        samples = node_samples.get(node.node_row, [])
        row = dict(node_row=node.node_row, node_id=node.node_id, is_root=is_root)
        for name in (
            "applicable_compound_count",
            "calculated_compound_count",
            "skipped_compound_count",
            "failed_compound_count",
            "evaluated_compound_count",
            "evaluated_sample_count",
            "improved_compound_count",
            "improved_sample_count",
            "overall_compound_count",
            "overall_sample_count",
        ):
            row[name] = int(aggregate.get(name, 0))
        for suffix, _ in SUFFIXES:
            peaks, intensity = aggregate.get("peaks" + suffix, 0), aggregate.get(
                "intensity" + suffix, 0
            )
            row["weighted_peak_coverage" + suffix] = _ratio(
                aggregate.get("assigned_peak_count" + suffix, 0), peaks
            )
            row["weighted_intensity_coverage" + suffix] = _ratio(
                aggregate.get("assigned_intensity" + suffix, 0), intensity
            )
            row["newly_assigned_peak_count" + suffix] = int(
                aggregate.get("newly_assigned_peak_count" + suffix, 0)
            )
            row["newly_assigned_intensity" + suffix] = float(
                aggregate.get("newly_assigned_intensity" + suffix, 0.0)
            )
            if is_root:
                for name in (
                    "parent_weighted_peak_coverage",
                    "parent_weighted_intensity_coverage",
                    "weighted_delta_peak_coverage",
                    "weighted_delta_intensity_coverage",
                ):
                    row[name + suffix] = math.nan
            else:
                row["parent_weighted_peak_coverage" + suffix] = _ratio(
                    aggregate.get("parent_assigned_peak_count" + suffix, 0), peaks
                )
                row["parent_weighted_intensity_coverage" + suffix] = _ratio(
                    aggregate.get("parent_assigned_intensity" + suffix, 0), intensity
                )
                row["weighted_delta_peak_coverage" + suffix] = _ratio(
                    aggregate.get("newly_assigned_peak_count" + suffix, 0), peaks
                )
                row["weighted_delta_intensity_coverage" + suffix] = _ratio(
                    aggregate.get("newly_assigned_intensity" + suffix, 0), intensity
                )
            row["overall_peak_coverage" + suffix] = _ratio(
                aggregate.get("overall_assigned_peaks" + suffix, 0),
                aggregate.get("overall_peaks" + suffix, 0),
            )
            row["overall_intensity_coverage" + suffix] = _ratio(
                aggregate.get("overall_assigned_intensity" + suffix, 0),
                aggregate.get("overall_intensity" + suffix, 0),
            )
        for metric in _metrics_for(is_root):
            for suffix, flag in SUFFIXES:
                stats = distribution(
                    (sample[metric + suffix] for sample in samples), metric
                )
                if metric in MEAN_METRICS:
                    row["mean_" + metric + suffix] = stats["mean"]
                tables.add(
                    "node_distribution",
                    node_row=node.node_row,
                    metric=metric,
                    without_precursor=flag,
                    **stats,
                )
        for name in MEAN_METRICS:
            for suffix, _ in SUFFIXES:
                row.setdefault("mean_" + name + suffix, math.nan)
        if formula_totals is not None:
            row.update(formula_node_row(formula_totals.get(node.node_row)))
        tables.add("node_summary", **row)

    for row in datasets:
        dataset_id = int(row["dataset_id"])
        stats = dataset_stats.get(dataset_id, {})
        tables.add(
            "datasets",
            **row,
            compound_count=len(dataset_compounds.get(dataset_id, ())),
            sample_count=int(stats.get("sample_count", 0)),
            peak_count=int(stats.get("peak_count", 0)),
        )
    _tree_tables(tables, tree)
    if by_dataset:
        _pattern_tables(tables, comparisons, nodes, datasets, rules or {})
    if by_dataset and len(datasets) > 1:
        _dataset_tables(
            tables,
            comparisons,
            tree=tree,
            datasets=datasets,
            with_formula=formula_totals is not None,
        )

    summary = {
        "datasets": len(datasets),
        "compounds": len(comparisons),
        "samples": int(overall["samples"]),
        "peaks": int(overall["peak_count"]),
        "totalIntensity": overall["total_intensity"],
        "peaksWithoutPrecursor": int(overall["peak_count_without_precursor"]),
        "totalIntensityWithoutPrecursor": overall["total_intensity_without_precursor"],
        "patternSets": len(tree.config.used_pattern_sets()),
        "nodes": len(nodes),
        "roots": sum(1 for node in nodes if node.parent_node_id is None),
        "edges": sum(1 for node in nodes if node.parent_node_id is not None),
        "conditions": len(tree.conditions),
        "applicableCompoundEdges": int(
            sum(
                node_totals[node.node_row].get("applicable_compound_count", 0)
                for node in nodes
                if node.parent_node_id is not None
            )
        ),
        "compoundsWithImprovement": len(improved_compounds),
        "samplesWithImprovement": len(improved_samples),
        "newlyAssignedPeaks": int(overall["new_peaks"]),
        "newlyAssignedIntensity": overall["new_intensity"],
    }
    return tables, summary


def _tree_tables(tables: Tables, tree) -> None:
    config, registry = tree.config, tree.registry
    used = config.used_pattern_sets()
    sets_by_pattern: dict[int, list[str]] = defaultdict(list)
    for item in used:
        pattern_ids = tree.set_pattern_ids[item.id]
        for pattern_id in pattern_ids:
            sets_by_pattern[pattern_id].append(item.id)
        tables.add(
            "pattern_sets",
            pattern_set_id=item.id,
            name=item.name,
            pattern_count=len(pattern_ids),
            pattern_ids=",".join(map(str, pattern_ids)),
            node_count=sum(1 for node in tree.nodes if node.pattern_set_id == item.id),
            source=item.source,
        )
    for pattern_id in tree.used_pattern_ids:
        definition = registry.definition(pattern_id)
        tables.add(
            "patterns",
            pattern_id=pattern_id,
            name=str(definition.get("name", "")),
            reactant_smarts=str(definition.get("reactant_smarts", "")),
            definition_json=json.dumps(definition, ensure_ascii=False),
            pattern_set_ids=",".join(sets_by_pattern.get(pattern_id, ())),
        )
    for condition in tree.conditions:
        tables.add(
            "conditions",
            condition_id=condition.condition_id,
            condition_key=condition.key,
            pattern_ids=",".join(map(str, sorted(condition.pattern_ids))),
            label=condition.label,
            detail_directory=condition.detail_directory,
        )
    for node in tree.nodes:
        tables.add(
            "nodes",
            node_row=node.node_row,
            node_id=node.node_id,
            parent_node_row=node.parent_row,
            parent_node_id=node.parent_node_id or "",
            pattern_set_id=node.pattern_set_id,
            pattern_set_name=node.pattern_set_name,
            depth=node.depth,
            is_root=node.parent_node_id is None,
            condition_id=node.condition_id,
            parent_condition_id=(
                -1 if node.parent_condition_id is None else node.parent_condition_id
            ),
            set_pattern_ids=",".join(map(str, node.set_pattern_ids)),
            added_pattern_ids=",".join(map(str, node.added_pattern_ids)),
            path_label=node.path_label,
            condition_label=node.condition_label,
        )
