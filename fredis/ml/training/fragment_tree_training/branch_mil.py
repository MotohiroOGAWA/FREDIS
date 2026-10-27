"""Exact softmax-weighted path MIL using reaction history graph partition functions.

For scores s and inverse temperature a, log P(bag) = log Z(a+1)-log Z(a).
Each distinct prefix is counted once, regardless of its number of continuations.
No complete paths are enumerated or materialized.
"""

import math
import torch
from torch.nn import functional as F
from ...specgen.competitive_spectrum_model import segment_logsumexp


def softmax_mil_log_probability(scores, temperature=0.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("MIL temperature must be finite and positive")
    if not scores.numel() or not bool(torch.isfinite(scores).any()):
        raise ValueError("Positive MIL bag is empty or unreachable")
    scores = (
        scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    )
    return torch.logsumexp(torch.log_softmax(scores / temperature, 0) + scores, 0)


def branch_path_mil_loss(logits, data, negative_weight=0.5, temperature=0.1):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("MIL temperature must be finite and positive")
    if negative_weight < 0 or not math.isfinite(negative_weight):
        raise ValueError("Invalid negative weight")
    teacher = data.branch_teacher
    if teacher is None:
        raise ValueError("Branch teacher is missing; regenerate prepared datasets")
    labels = teacher.edge_label
    if labels.shape != logits.shape:
        raise ValueError("Branch logits must align with prepared execution transitions")
    scores = F.logsigmoid(logits.float())
    zero = logits[torch.isfinite(logits)].sum() * 0
    n = teacher.node_state_index.numel()
    depth = teacher.node_depth
    src, dst = teacher.source_index, teacher.target_index
    edge_scores = scores[teacher.edge_index]
    # Center each state's partition on its best detached path score. This
    # avoids subtracting huge numbers when temperature is very small.
    levels = [
        depth[dst] == d for d in range(1, int(depth.max()) + 1 if depth.numel() else 1)
    ]
    with torch.no_grad():
        best = scores.new_full((n,), -torch.inf)
        best[teacher.root_index] = 0
        for mask in levels:
            incoming = scores.new_full((n,), -torch.inf)
            incoming.scatter_reduce_(
                0,
                dst[mask],
                best[src[mask]] + edge_scores[mask],
                reduce="amax",
                include_self=True,
            )
            rows = torch.unique(dst[mask])
            best[rows] = incoming[rows]
    shift = best.masked_fill(~torch.isfinite(best), 0)
    centered_edges = edge_scores + shift[src] - shift[dst]
    # Two partition functions summarize all distinct paths to each state.
    a = 1.0 / temperature
    partitions = []
    for exponent in (a, a + 1):
        z = scores.new_full((n,), -torch.inf)
        z[teacher.root_index] = 0
        for d, mask in enumerate(levels, 1):
            incoming = segment_logsumexp(
                z[src[mask]] + exponent * centered_edges[mask], dst[mask], n
            )
            z = torch.where(depth == d, incoming, z)
        partitions.append(z)
    nodes, bags = teacher.node_index, teacher.node_bag_index
    bag_peaks = teacher.bag_peak_index
    peak_groups = teacher.peak_branch_group_index
    negative_rows, negative_bags = (
        teacher.negative_edge_index,
        teacher.negative_bag_index,
    )
    bag_count, peak_count = bag_peaks.numel(), peak_groups.numel()
    positive_count = torch.bincount(bags, minlength=bag_count)
    has_positive = positive_count > 0
    bag_reference = scores.new_full((bag_count,), -torch.inf)
    bag_reference.scatter_reduce_(
        0, bags, best[nodes], reduce="amax", include_self=True
    )
    if not bool(torch.isfinite(bag_reference[has_positive]).all()):
        raise ValueError("Prepared positive MIL bag is unreachable")
    bag_reference = bag_reference.masked_fill(~has_positive, 0)
    offsets = shift[nodes] - bag_reference[bags]
    denominator = segment_logsumexp(
        partitions[0][nodes] + a * offsets, bags, bag_count
    ).masked_fill(~has_positive, 0)
    numerator = segment_logsumexp(
        partitions[1][nodes] + (a + 1) * offsets, bags, bag_count
    ).masked_fill(~has_positive, 0)
    bag_loss = denominator - numerator - bag_reference
    negative_count = torch.bincount(negative_bags, minlength=bag_count)
    bag_negative = scores.new_zeros(bag_count).scatter_add_(
        0, negative_bags, F.softplus(logits[negative_rows].float())
    ) / negative_count.clamp_min(1)
    # All weights and normalization are runtime choices, never baked into data.
    positive_weight = has_positive.to(scores.dtype)
    negative_weights = negative_weight * (negative_count > 0).to(scores.dtype)
    bag_weight = positive_weight + negative_weights
    combined = (bag_loss + negative_weights * bag_negative) / bag_weight.clamp_min(
        torch.finfo(scores.dtype).tiny
    )
    active = bag_weight > 0
    peak_depths = scores.new_zeros(peak_count).scatter_add_(
        0, bag_peaks, active.to(scores.dtype)
    )
    group_count = data.num_branch_groups
    group_peaks = scores.new_zeros(group_count).scatter_add_(
        0, peak_groups, (peak_depths > 0).to(scores.dtype)
    )

    def reduce_bags(values):
        per_peak = scores.new_zeros(peak_count).scatter_add_(
            0, bag_peaks, values * active
        ) / peak_depths.clamp_min(1)
        per_group = scores.new_zeros(group_count).scatter_add_(
            0, peak_groups, per_peak
        ) / group_peaks.clamp_min(1)
        return per_group.sum() + zero

    return reduce_bags(combined), reduce_bags(bag_loss), reduce_bags(bag_negative)
