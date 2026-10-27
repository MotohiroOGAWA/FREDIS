"""Parameter-independent tensor indices for spectrum history propagation."""

from dataclasses import dataclass
import torch
from torch import Tensor


def probability_levels(source, target, count, depth=None):
    """Group events by dependency depth without copying graph edges to Python."""
    if bool(((source < 0) | (source >= count) | (target < -1) | (target >= count)).any()):
        raise ValueError("Invalid probability event endpoint")
    child = (target >= 0) & (target != source)
    self_event = target == source
    sink = target == -1
    if depth is None:
        parents, children = source[child], target[child]
        indegree = torch.bincount(children, minlength=count)
        frontier = indegree == 0
        depth = source.new_full((count,), -1)
        level = 0
        while bool(frontier.any()):
            depth.masked_fill_(frontier, level)
            removed = torch.bincount(children[frontier[parents]], minlength=count)
            indegree = indegree - removed
            frontier = (indegree == 0) & (depth < 0)
            level += 1
        if bool((depth < 0).any()):
            raise ValueError("Fragment reaction paths must not contain cycles")
    elif bool((depth[target[child]] <= depth[source[child]]).any()):
        raise ValueError("Fragment reaction paths must not contain cycles")
    event_depth = depth[source]
    return tuple(
        tuple(((event_depth == level) & mask).nonzero(as_tuple=True)[0]
              for mask in (self_event, sink, child))
        for level in range(int(depth.max()) + 1 if count else 0)
    )


def expand_group_rows(groups, selected, group_count):
    """Repeat source-local rows for selected groups while preserving row order."""
    order = torch.argsort(groups, stable=True)
    counts = torch.bincount(groups, minlength=group_count)
    starts = counts.cumsum(0) - counts
    repeats = counts[selected]
    owner = torch.repeat_interleave(
        torch.arange(selected.numel(), device=selected.device), repeats
    )
    output_starts = repeats.cumsum(0) - repeats
    slot = torch.arange(owner.numel(), device=selected.device) - output_starts[owner]
    rows = order[starts[selected[owner]] + slot]
    local = torch.empty_like(order)
    local[order] = torch.arange(order.numel(), device=order.device) - starts[groups[order]]
    return rows, owner, output_starts, local


@dataclass(frozen=True)
class CanonicalSpectrumTopology:
    execution_rows: Tensor
    compounds: Tensor
    samples: Tensor
    roots: Tensor
    source: Tensor
    target: Tensor
    reactions: Tensor
    event_type: Tensor
    fragments: Tensor
    history_to_fragment: Tensor
    state_fragment: Tensor
    physical_roots: Tensor
    levels: tuple


def canonical_spectrum_topology(data):
    """Expand histories across spectra using grouped tensor gathers, once per batch."""
    core = data.canonical_data
    execution = core.execution_graph
    selected = core.sample_branch_group_index
    rows, samples, starts, local = expand_group_rows(
        execution.branch_group_index, selected, core.num_branch_groups
    )
    compounds = execution.compound_index[rows]
    roots = starts + local[execution.root_index[selected]]
    edge_rows, edge_samples, _, _ = expand_group_rows(
        execution.branch_group_index[execution.source_index], selected, core.num_branch_groups
    )
    source = starts[edge_samples] + local[execution.source_index[edge_rows]]
    target = starts[edge_samples] + local[execution.target_index[edge_rows]]
    reactions = execution.reaction_index[edge_rows]
    events = torch.arange(rows.numel(), device=rows.device)
    source = torch.cat((source, events.repeat_interleave(2)))
    target = torch.cat((target, torch.stack((events, torch.full_like(events, -1)), 1).flatten()))
    reactions = torch.cat((reactions, rows.new_tensor([-1, -2]).repeat(rows.numel())))
    event_type = torch.cat((torch.zeros_like(edge_rows), rows.new_tensor([1, 2]).repeat(rows.numel())))
    fragments, inverse = torch.unique(
        torch.stack((samples, compounds), 1), dim=0, return_inverse=True
    )
    compound_count = len(core.compounds.compounds)
    physical_keys = fragments[:, 0] * compound_count + fragments[:, 1]
    sample_roots = execution.compound_index[execution.root_index[selected]]
    root_keys = torch.arange(selected.numel(), device=rows.device) * compound_count + sample_roots
    physical_roots = torch.searchsorted(physical_keys, root_keys)[fragments[:, 0]]
    state_keys = data.state_sample_index * compound_count + data.state_node_index
    state_fragment = torch.searchsorted(physical_keys, state_keys)
    if bool(((state_fragment >= physical_keys.numel()) |
             (physical_keys[state_fragment.clamp_max(physical_keys.numel() - 1)] != state_keys)).any()):
        raise ValueError("Ion states must reference reachable spectrum compounds")
    depth = core.reaction_sets.reaction_ptr.diff()[execution.reaction_set_index[rows]]
    return CanonicalSpectrumTopology(
        rows, compounds, samples, roots, source, target, reactions, event_type,
        fragments, inverse, state_fragment, physical_roots,
        probability_levels(source, target, rows.numel(), depth),
    )
