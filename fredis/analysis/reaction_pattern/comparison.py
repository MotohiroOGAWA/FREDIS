"""Peak-level metrics of every tree node and edge of one compound.

Peaks are joined by ``(sample_id, peak_index)``.  Every annotation is binary
per peak.  A condition's ``assigned`` mask is absolute: a child condition's
mask is its parent's mask merged with the child tree's assignments (the
seeded child tree need not repeat parent histories at full depth).

A root node reports absolute metrics only.  An edge additionally reports its
parent condition, ``delta = child - parent`` and newly assigned peaks
(``child and not parent``).  An edge is evaluated only for compounds whose
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
    file: str = ""        # this condition's own tree ("" when inherited or not calculated)
    seed_file: str = ""   # tree whose reaction-sequence compounds seed child conditions
    alias_of: int | None = None  # identical to this condition (no added reactions for the compound)
    search_stats: dict | None = None

    @property
    def completed(self) -> bool:
        return self.status == COMPLETED


def empty_outcome(group: CompoundGroup, status: str, **values) -> ConditionOutcome:
    return ConditionOutcome(status=status, assigned=[np.zeros(len(sample.mz), dtype=bool) for sample in group.samples],
                            counts=[np.zeros(len(sample.mz), dtype=np.int32) for sample in group.samples], **values)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else math.nan


def coverage(assigned: np.ndarray, intensity: np.ndarray, population: np.ndarray) -> dict:
    """Peak/intensity totals and coverages over ``population`` peaks."""
    peaks = int(population.sum())
    total = float(intensity[population].sum())
    hit = assigned & population
    count = int(hit.sum())
    value = float(intensity[hit].sum())
    return dict(peaks=peaks, intensity=total, assigned_peak_count=count, assigned_intensity=value,
                peak_coverage=_ratio(count, peaks), intensity_coverage=_ratio(value, total))


def node_metrics(current: dict, parent: dict | None, new: np.ndarray | None, intensity: np.ndarray,
                 population: np.ndarray) -> dict:
    metrics = dict(current)
    if parent is None:
        return metrics
    newly = new & population
    metrics.update(
        parent_assigned_peak_count=parent["assigned_peak_count"], parent_assigned_intensity=parent["assigned_intensity"],
        parent_peak_coverage=parent["peak_coverage"], parent_intensity_coverage=parent["intensity_coverage"],
        newly_assigned_peak_count=int(newly.sum()), newly_assigned_intensity=float(intensity[newly].sum()),
        delta_peak_coverage=current["peak_coverage"] - parent["peak_coverage"],
        delta_intensity_coverage=current["intensity_coverage"] - parent["intensity_coverage"])
    return metrics


@dataclass
class SampleNodeMetrics:
    sample: SampleRecord
    assigned: np.ndarray
    new: np.ndarray | None
    metrics: dict
    metrics_without: dict


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


def compare_compound(group: CompoundGroup, nodes: list, outcomes: dict[int, ConditionOutcome],
                     applicable: dict[str, bool]) -> CompoundComparison:
    compound_nodes = []
    population: set[int] = set()
    for node in nodes:
        outcome = outcomes[node.condition_id]
        parent = outcomes[node.parent_condition_id] if node.parent_condition_id is not None else None
        item = CompoundNode(node, outcome, parent, True if parent is None else bool(applicable.get(node.node_id)))
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
                    metrics = node_metrics(coverage(current, sample.intensity, everything), None, None,
                                           sample.intensity, everything)
                    metrics_without = node_metrics(coverage(current, sample.intensity, without), None, None,
                                                   sample.intensity, without)
                else:
                    before = parent.assigned[index]
                    new = current & ~before
                    metrics = node_metrics(coverage(current, sample.intensity, everything),
                                           coverage(before, sample.intensity, everything), new,
                                           sample.intensity, everything)
                    metrics_without = node_metrics(coverage(current, sample.intensity, without),
                                                   coverage(before, sample.intensity, without), new,
                                                   sample.intensity, without)
                item.samples.append(SampleNodeMetrics(sample, current, new, metrics, metrics_without))
        compound_nodes.append(item)
    return CompoundComparison(group, compound_nodes, outcomes, tuple(sorted(population)))
