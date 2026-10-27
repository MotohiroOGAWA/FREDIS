"""Structural reaction pool and materialization transition tables."""

from __future__ import annotations
from dataclasses import dataclass, replace
import torch
from torch import Tensor


@dataclass(frozen=True)
class ReactionPool:
    reaction_index: Tensor
    valid: Tensor
    embeddings: Tensor
    conflict: Tensor
    invalidation: Tensor
    retained: Tensor
    source_atom_valid: Tensor


def build_reaction_pool(reaction_h: Tensor, data: object) -> ReactionPool:
    # There is intentionally no learned absolute prefilter. Every primitive
    # reaction belonging to a branch group's source is retained; chemistry-derived
    # compatibility masks decide which continuations are valid.
    s = data.num_branch_groups
    a = reaction_h.shape[0]
    device = reaction_h.device
    trees = data.branch_groups.source_compound_index
    eligible = data.reactions.reaction_source_compound_index[None, :] == trees[:, None]
    width = int(eligible.sum(1).max().item()) if s else 0
    ids = torch.argsort(eligible.to(torch.int8), dim=1, descending=True, stable=True)[
        :, :width
    ]
    valid = eligible.gather(1, ids)
    chosen_ids = ids.masked_fill(~valid, a)
    padded_h = torch.cat((reaction_h, reaction_h.new_zeros((1, reaction_h.shape[1]))))
    inverse = torch.full((s, a + 1), -1, dtype=torch.long, device=device)
    slots = torch.arange(width, device=device).expand(s, -1)
    inverse.scatter_(1, chosen_ids, slots)
    inverse[:, a] = -1
    sample = torch.arange(s, device=device)[:, None]

    registry = data.reactions
    retained, atom_valid = registry.retained_bitsets(
        inverse, data.source_graph.ptr, trees,
        data.source_atom_capacity or data.source_graph.num_nodes, width,
    )
    return ReactionPool(
        chosen_ids.masked_fill(~valid, -1),
        valid,
        padded_h[chosen_ids],
        registry.relation_mask(registry.reaction_conflict_index, inverse, width),
        registry.relation_mask(registry.reaction_invalidation_index, inverse, width),
        retained,
        atom_valid,
    )


@dataclass(frozen=True)
class ReactionDecoderOutput:
    state_sample_index: Tensor
    state_reaction_index: Tensor
    state_log_score: Tensor
    terminal: Tensor
    parent_state_index: Tensor
    added_reaction_index: Tensor
    pool: ReactionPool
    duplicate_states_removed: Tensor | None = None
    transition_parent_index: Tensor | None = None
    transition_child_index: Tensor | None = None
    transition_reaction_index: Tensor | None = None


def with_all_transitions(output, compatibility):
    """Keep every valid edge between accepted sets, despite state deduplication."""
    expansion = compatibility.expand(
        state_reaction_index=output.state_reaction_index,
        state_sample_index=output.state_sample_index,
        pool_valid=output.pool.valid,
        pool_conflict=output.pool.conflict,
        pool_invalidation=output.pool.invalidation,
        pool_retained=output.pool.retained,
        source_atom_valid=output.pool.source_atom_valid,
    )
    keys = {
        (s, tuple(state)): i
        for i, (s, state) in enumerate(
            zip(
                output.state_sample_index.tolist(), output.state_reaction_index.tolist()
            )
        )
    }
    parents = []
    children = []
    reactions = []
    valid = expansion.valid.nonzero().flatten()
    for p, r, state in zip(
        expansion.parent_state_index[valid].tolist(),
        expansion.candidate_reaction_index[valid].tolist(),
        expansion.child_reaction_index[valid].tolist(),
    ):
        child = keys.get((int(output.state_sample_index[p]), tuple(state)))
        if child is not None and child != p:
            parents.append(p)
            children.append(child)
            reactions.append(r)

    def tensor(values):
        return torch.tensor(
            values, device=output.state_sample_index.device, dtype=torch.long
        )

    return replace(
        output,
        transition_parent_index=tensor(parents),
        transition_child_index=tensor(children),
        transition_reaction_index=tensor(reactions),
    )
