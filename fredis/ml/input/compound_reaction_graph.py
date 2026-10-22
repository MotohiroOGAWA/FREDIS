"""Shared structural edges. Node IDs are directly CompoundBatch row IDs."""

from dataclasses import dataclass
from torch import Tensor
from .tensor_table import TensorTable


@dataclass(frozen=True)
class CompoundReactionGraph(TensorTable):
    source_compound_index: Tensor
    target_compound_index: Tensor
    reaction_index: Tensor


@dataclass(frozen=True)
class ReactionExecutionGraph(TensorTable):
    """Finite derived history expansion; molecular structures remain shared."""

    compound_index: Tensor
    branch_group_index: Tensor
    source_index: Tensor
    target_index: Tensor
    reaction_index: Tensor
    root_index: Tensor
    reaction_set_index: Tensor
