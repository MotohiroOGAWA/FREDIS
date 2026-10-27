"""Group-local context and transition rows referencing independent ReactionSet IDs."""

from dataclasses import dataclass, field
import torch
from torch import Tensor
from .tensor_table import TensorTable


@dataclass(frozen=True)
class ReactionContextData(TensorTable):
    # [Q]: branch group owning each reaction state.
    branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [Q]: representative parent state; -1 for roots. Full paths use transitions.
    parent_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [Q]: concrete reaction added from the representative parent; -1 for roots.
    added_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [Q]: runtime fragment occurrence index, -1 if not materialized; not a compound ID.
    fragment_node_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [E]: source state IDs into this Q-state table.
    transition_parent_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [E]: destination state IDs into this Q-state table.
    transition_child_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [E]: added concrete reaction IDs.
    transition_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [Q + 1]: CSR boundaries of chemically valid continuations.
    next_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # [N]: added concrete reaction IDs for valid continuations.
    next_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [N]: destination IDs into the candidate-state table (V rows).
    next_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [V]: branch group owning each candidate state.
    candidate_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [V]: FK into SourceReactionStructure.reaction_sets; membership is not copied.
    candidate_reaction_set_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [V]: materialized runtime fragment occurrence, -1 if not materialized.
    candidate_fragment_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [Q]: corresponding row in the V-row candidate table.
    candidate_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )


@dataclass(frozen=True)
class PeakPathAssignments(TensorTable):
    # [H]: observed peak IDs into annotations.spectra.mz; paths reference measurements.
    peak_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [H + 1]: CSR boundaries of alternative paths per supervised peak item.
    peak_path_ptr: Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long)
    )
    # [H]: branch group of each supervised peak item.
    peak_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [T + 1]: CSR boundaries of steps per alternative path.
    path_step_ptr: Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long)
    )
    # [U]: state IDs into ReactionContextData (Q rows).
    path_step_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [U]: concrete reaction ID for each path step.
    path_step_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
