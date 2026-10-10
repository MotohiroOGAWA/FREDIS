"""Differentiable reductions and sparse index lookup for ragged batches."""
import torch
from torch import Tensor


def segment_logsumexp(values: Tensor, groups: Tensor, size: int) -> Tensor:
    maximum = values.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, groups, values.detach(), reduce='amax', include_self=True)
    total = values.new_zeros(size).scatter_add(0, groups, (values - maximum[groups]).exp())
    # Empty segments remain -inf without a log(0) derivative in autograd.
    return maximum + total.clamp_min(torch.finfo(values.dtype).tiny).log()


def lookup_rows(keys: Tensor, queries: Tensor) -> Tensor:
    """Map query rows to the last matching key row, or -1 (dictionary semantics)."""
    _, inverse = torch.unique(torch.cat((keys, queries)), dim=0, return_inverse=True)
    lookup = inverse.new_full((keys.size(0) + queries.size(0),), -1)
    lookup.scatter_reduce_(0, inverse[:keys.size(0)],
                           torch.arange(keys.size(0), device=keys.device),
                           reduce='amax', include_self=True)
    return lookup[inverse[keys.size(0):]]
