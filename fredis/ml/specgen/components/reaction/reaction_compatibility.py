"""Tensor-only transitions of normalized Source reaction sets."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ReactionExpansion:
    parent_state_index: Tensor
    candidate_reaction_index: Tensor
    valid: Tensor
    child_reaction_index: Tensor
    child_reaction_count: Tensor
    retained_atom_mask: Tensor
    hard_conflict: Tensor
    invalidated_reaction: Tensor
    no_op: Tensor


class ReactionCompatibilityEngine(nn.Module):
    def __init__(self, max_reaction_count: int) -> None:
        super().__init__()
        if type(max_reaction_count) is not int or max_reaction_count < 1:
            raise ValueError("max_reaction_count must be a positive integer")
        self.max_reaction_count = max_reaction_count

    def expand(self, *, state_reaction_index: Tensor, state_sample_index: Tensor,
               pool_valid: Tensor, pool_conflict: Tensor, pool_invalidation: Tensor,
               pool_dominance: Tensor, pool_retained: Tensor, source_atom_valid: Tensor) -> ReactionExpansion:
        p, m = state_reaction_index.shape
        k = pool_valid.shape[1]
        device = state_reaction_index.device
        parent = torch.arange(p, device=device).repeat_interleave(k)
        candidate = torch.arange(k, device=device).repeat(p)
        sample = state_sample_index[parent]
        previous = state_reaction_index[parent]
        raw = torch.cat((previous, candidate[:, None]), dim=1)
        present = raw >= 0
        safe = raw.clamp_min(0)
        pair_valid = present[:, :, None] & present[:, None, :]
        rows, cols = safe[:, :, None], safe[:, None, :]
        conflict = (pool_conflict[sample[:, None, None], rows, cols] & pair_valid).any(dim=(1, 2))
        invalidated = (pool_invalidation[sample[:, None, None], rows, cols] & pair_valid).any(dim=(1, 2))
        dominated = (pool_dominance[sample[:, None, None], rows, cols] & pair_valid).any(dim=1)
        normalized = present & ~dominated
        count = normalized.sum(dim=1)
        sorted_ids = raw.masked_fill(~normalized, k).sort(dim=1).values
        child = sorted_ids[:, :m].masked_fill(sorted_ids[:, :m] == k, -1)
        no_op = (child == previous).all(dim=1) & (count == (previous >= 0).sum(dim=1))
        if pool_retained.dtype == torch.bool:
            retained = (pool_retained[sample[:, None], safe] | ~normalized[:, :, None]).all(dim=1)
            retained = retained & source_atom_valid[sample]
        else:
            retained = source_atom_valid[sample].clone()
            for position in range(m + 1):
                words=pool_retained[sample,safe[:,position]]
                retained=retained & torch.where(normalized[:,position,None],words,torch.full_like(words,-1))
        repeated = (previous == candidate[:, None]).any(dim=1)
        valid = (pool_valid[sample, candidate] & ~repeated & ~conflict & ~invalidated & ~no_op
                 & retained.any(dim=1) & (count <= self.max_reaction_count))
        return ReactionExpansion(parent, candidate, valid, child, count, retained, conflict, invalidated, no_op)
