"""Peak-level metrics of every tree node and edge of one compound.

Peaks are joined by ``(sample_id, peak_index)``.  Every annotation is binary
per peak.  A condition's ``assigned`` mask is absolute and comes from that
condition's own tree.  A child usually assigns every parent peak, but a new
pattern that reaches a spectrum's precursor with fewer reactions can also
remove parent explanations.

A root node reports absolute metrics only.  An edge additionally reports its
parent condition, ``delta = child - parent`` and newly assigned peaks
(``child and not parent``).  Each condition also records which patterns
explain each peak, so a pattern can be counted as having reacted (it explains
an assigned peak) or as newly reacting (it explains a newly assigned peak).  An edge is evaluated only for compounds whose
precursor matches a pattern its set adds; for any other compound the child
condition is identical to the parent condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .compounds import CompoundGroup, SampleRecord
from .constants import COMPLETED


@dataclass
class ConditionOutcome:
    """One cumulative condition of one compound.

    ``assigned``/``counts`` are aligned to every sample of the compound group;
    only positions in ``kept`` form the evaluated population.
    """

    status: str
    skip_category: str = ""
    reason: str = ""
    error: str = ""
    kept: tuple[int, ...] = ()
    assigned: list[np.ndarray] = field(default_factory=list)
    counts: list[np.ndarray] = field(default_factory=list)
    file: str = ""  # this condition's own tree ("" when inherited or not calculated)
    seed_file: str = ""  # tree whose reaction-sequence compounds seed child conditions
    alias_of: int | None = (
        None  # identical to this condition (no added reactions for the compound)
    )
    search_stats: dict | None = None
    # Per sample, pattern id -> peaks explained by a history with that pattern.
    # None when unknown (a result written before patterns were recorded).
    patterns: list[dict[int, np.ndarray]] | None = None
    # Per sample, (pattern id, rule id) -> peaks explained with that product.
    rules: list[dict[tuple[int, int], np.ndarray]] | None = None

    @property
    def completed(self) -> bool:
        return self.status == COMPLETED


def empty_outcome(group: CompoundGroup, status: str, **values) -> ConditionOutcome:
    return ConditionOutcome(
        status=status,
        assigned=[np.zeros(len(sample.mz), dtype=bool) for sample in group.samples],
        counts=[np.zeros(len(sample.mz), dtype=np.int32) for sample in group.samples],
        patterns=[{} for _ in group.samples],
        rules=[{} for _ in group.samples],
        **values,
    )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else math.nan


def coverage(
    assigned: np.ndarray, intensity: np.ndarray, population: np.ndarray
) -> dict:
    """Peak/intensity totals and coverages over ``population`` peaks."""
    peaks = int(population.sum())
    total = float(intensity[population].sum())
    hit = assigned & population
    count = int(hit.sum())
    value = float(intensity[hit].sum())
    return dict(
        peaks=peaks,
        intensity=total,
        assigned_peak_count=count,
        assigned_intensity=value,
        peak_coverage=_ratio(count, peaks),
        intensity_coverage=_ratio(value, total),
    )


def node_metrics(
    current: dict,
    parent: dict | None,
    new: np.ndarray | None,
    intensity: np.ndarray,
    population: np.ndarray,
) -> dict:
    metrics = dict(current)
    if parent is None:
        return metrics
    newly = new & population
    metrics.update(
        parent_assigned_peak_count=parent["assigned_peak_count"],
        parent_assigned_intensity=parent["assigned_intensity"],
        parent_peak_coverage=parent["peak_coverage"],
        parent_intensity_coverage=parent["intensity_coverage"],
        newly_assigned_peak_count=int(newly.sum()),
        newly_assigned_intensity=float(intensity[newly].sum()),
        delta_peak_coverage=current["peak_coverage"] - parent["peak_coverage"],
        delta_intensity_coverage=current["intensity_coverage"]
        - parent["intensity_coverage"],
    )
    return metrics


@dataclass
class SampleNodeMetrics:
    sample: SampleRecord
    assigned: np.ndarray
    new: np.ndarray | None
    metrics: dict
    metrics_without: dict
    # Pattern id -> peaks explained with it under this node's condition (None: unknown).
    patterns: dict[int, np.ndarray] | None = None
    rules: dict[tuple[int, int], np.ndarray] | None = None


@dataclass
class CompoundNode:
    """One tree node for one compound.

    ``evaluated`` is True when this node contributes rows: a completed root,
    or an applicable edge whose child condition completed.
    """

    node: object  # tree.ResolvedNode
    outcome: ConditionOutcome
    parent_outcome: ConditionOutcome | None
    applicable: bool
    samples: list[SampleNodeMetrics] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.parent_outcome is None

    @property
    def evaluated(self) -> bool:
        return self.outcome.completed and (self.is_root or self.applicable)


@dataclass
class CompoundComparison:
    group: CompoundGroup
    nodes: list[CompoundNode]
    outcomes: dict[int, ConditionOutcome]
    # Samples evaluated by at least one root condition, in group order.
    population: tuple[int, ...] = ()
    # Ids of the reported patterns whose reactant SMARTS matches the precursor.
    matched_patterns: frozenset[int] = frozenset()
    # Pattern id -> rule (product) ids with a reaction instance on the precursor;
    # None when unknown.
    instance_rules: dict[int, frozenset[int]] | None = None


def compare_compound(
    group: CompoundGroup,
    nodes: list,
    outcomes: dict[int, ConditionOutcome],
    applicable: dict[str, bool],
    matched_patterns=(),
    instance_rules: dict | None = None,
) -> CompoundComparison:
    compound_nodes = []
    population: set[int] = set()
    for node in nodes:
        outcome = outcomes[node.condition_id]
        parent = (
            outcomes[node.parent_condition_id]
            if node.parent_condition_id is not None
            else None
        )
        item = CompoundNode(
            node,
            outcome,
            parent,
            True if parent is None else bool(applicable.get(node.node_id)),
        )
        if parent is None and outcome.completed:
            population.update(outcome.kept)
        if item.evaluated:
            # Edges use the parent's population, which the child condition must retain.
            positions = outcome.kept if parent is None else parent.kept
            for index in positions:
                sample = group.samples[index]
                everything = np.ones(len(sample.mz), dtype=bool)
                without = ~sample.is_precursor
                current = outcome.assigned[index]
                if parent is None:
                    new = None
                    metrics = node_metrics(
                        coverage(current, sample.coverage_intensity, everything),
                        None,
                        None,
                        sample.coverage_intensity,
                        everything,
                    )
                    metrics_without = node_metrics(
                        coverage(current, sample.coverage_intensity, without),
                        None,
                        None,
                        sample.coverage_intensity,
                        without,
                    )
                else:
                    before = parent.assigned[index]
                    new = current & ~before
                    metrics = node_metrics(
                        coverage(current, sample.coverage_intensity, everything),
                        coverage(before, sample.coverage_intensity, everything),
                        new,
                        sample.coverage_intensity,
                        everything,
                    )
                    metrics_without = node_metrics(
                        coverage(current, sample.coverage_intensity, without),
                        coverage(before, sample.coverage_intensity, without),
                        new,
                        sample.coverage_intensity,
                        without,
                    )
                item.samples.append(
                    SampleNodeMetrics(
                        sample,
                        current,
                        new,
                        metrics,
                        metrics_without,
                        None if outcome.patterns is None else outcome.patterns[index],
                        None if outcome.rules is None else outcome.rules[index],
                    )
                )
        compound_nodes.append(item)
    return CompoundComparison(
        group,
        compound_nodes,
        outcomes,
        tuple(sorted(population)),
        frozenset(int(value) for value in matched_patterns),
        None
        if instance_rules is None
        else {int(key): frozenset(int(v) for v in value) for key, value in instance_rules.items()},
    )
