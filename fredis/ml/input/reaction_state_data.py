"""Reaction states are chemical annotations; teachers reference them without copying them."""

from dataclasses import dataclass, field, fields, replace
import torch
from torch import Tensor


class TensorTable:
    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class ReactionStateData(TensorTable):
    # [Q]: branch group owning each reaction state.
    branch_group_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [Q]: representative parent state; -1 for roots. Full paths use transitions.
    parent_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [Q]: concrete reaction added from the representative parent; -1 for roots.
    added_reaction_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [Q]: runtime fragment occurrence index, -1 if not materialized; not a compound ID.
    fragment_node_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [E]: source state IDs into this Q-state table.
    transition_parent_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [E]: destination state IDs into this Q-state table.
    transition_child_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [E]: added concrete reaction IDs.
    transition_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [Q + 1]: CSR boundaries of chemically valid continuations.
    next_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # [N]: added concrete reaction IDs for valid continuations.
    next_reaction_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [N]: destination IDs into the candidate-state table (V rows).
    next_state_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [V]: branch group owning each candidate state.
    candidate_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [V + 1]: CSR boundaries of candidate reaction sets.
    candidate_reaction_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # [L]: candidate concrete reaction IDs.
    candidate_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [V]: materialized runtime fragment occurrence, -1 if not materialized.
    candidate_fragment_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [V]: unique global CompoundData ID, -1 if not materialized. Rebuilt when batching.
    # This is the chemical state -> compound relation; many states may share one ID.
    compound_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [Q]: corresponding row in the V-row candidate table.
    candidate_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    @property
    def reaction_count(self):
        """[Q] computed counts for the active-state projection; never stored twice."""
        ptr = self.candidate_reaction_ptr
        return ptr[self.candidate_index + 1] - ptr[self.candidate_index]

    @property
    def reaction_ptr(self):
        """[Q + 1] runtime CSR boundaries for active states selected from V base states."""
        counts = self.reaction_count
        return torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    @property
    def reaction_index(self):
        """Runtime reaction IDs gathered from the canonical V-state CSR table."""
        counts = self.reaction_count
        ptr = self.reaction_ptr
        starts = self.candidate_reaction_ptr[self.candidate_index]
        rows = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
        offsets = torch.arange(int(ptr[-1]), device=counts.device) - ptr[rows]
        return self.candidate_reaction_index[starts[rows] + offsets]


@dataclass(frozen=True)
class TeacherData(TensorTable):
    # [H]: observed peak IDs into annotations.spectra.mz; paths reference measurements.
    peak_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [Q + 1]: CSR boundaries of positive additions for each base reaction state.
    positive_reaction_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # [Kpositive]: positive concrete reaction IDs.
    positive_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [Q]: whether each base reaction state explains an observed ion.
    observed: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.bool))
    # [Q + 1]: CSR boundaries of weak negative additions.
    negative_reaction_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # [W]: weak negative concrete reaction IDs.
    negative_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [H + 1]: CSR boundaries of alternative paths per supervised peak item.
    peak_path_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # [H]: branch group of each supervised peak item.
    peak_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    # [T + 1]: CSR boundaries of steps per alternative path.
    path_step_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # [U]: state IDs into ReactionStateData (Q rows).
    path_step_state_index: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # [U]: concrete reaction ID for each path step.
    path_step_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )

    def spectrum_targets(self, annotations):
        """Derive [J] intensity targets from peak->formula assignments.

        No peak intensities or formula vectors are copied into TeacherData.
        Ambiguous explanations share the maximum assigned observed intensity.
        """
        if self.observed.numel() == 0:
            return None
        spectra, formulas = annotations.spectra, annotations.formulas
        fcount = formulas.composition.shape[0]
        scount = spectra.peak_ptr.numel() - 1
        target = spectra.intensity.new_zeros(scount * fcount)
        assignment = annotations.assignments
        peaks = torch.repeat_interleave(
            torch.arange(spectra.mz.numel(), device=target.device),
            assignment.formula_ptr[1:] - assignment.formula_ptr[:-1],
        )
        samples = torch.repeat_interleave(
            torch.arange(scount, device=target.device), spectra.peak_ptr[1:] - spectra.peak_ptr[:-1]
        )
        key = samples[peaks] * fcount + assignment.formula_index
        target.scatter_reduce_(0, key, spectra.intensity[peaks], reduce="amax", include_self=True)
        ions = annotations.ions
        formula = ions.formula_index[ions.runtime_index]
        return target[ions.runtime_sample_index * fcount + formula] * formulas.valid[formula]
