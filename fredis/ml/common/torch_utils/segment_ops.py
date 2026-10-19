"""Differentiable normalization for sparse groups of competing transitions."""

import torch
from torch import Tensor


def segment_softmax(logits: Tensor, groups: Tensor, size: int) -> Tensor:
    """Normalize a flat event vector independently within each source group.

    ``logits`` and ``groups`` have shape ``[num_events]``. Group indices are
    integers in ``[0, size)``; absent groups produce no output rows. Subtracting
    the per-group maximum keeps exponentiation stable without changing gradients.
    """
    maximum = logits.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, groups, logits, reduce="amax", include_self=True)
    weights = (logits - maximum[groups]).exp()
    total = logits.new_zeros(size).scatter_add_(0, groups, weights)
    return weights / total[groups].clamp_min(torch.finfo(logits.dtype).tiny)
