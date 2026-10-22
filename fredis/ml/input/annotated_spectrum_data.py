"""Canonical observations and formula explanations; no runtime occurrence IDs."""

from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor
from .tensor_table import TensorTable


@dataclass(frozen=True)
class SpectrumData(TensorTable):
    # [S + 1]: observed peak rows belonging to each spectrum.
    peak_ptr: Tensor
    # [P]: observed m/z and relative intensity; these are measurements, not predictions.
    mz: Tensor
    intensity: Tensor
    # [P]: precursor membership determined by the configured mass tolerance.
    precursor: Tensor
    # [S]: references to BranchGroup.
    branch_group_index: Tensor
    # [S]: prepared precursor m/z and observed collision energy.
    precursor_mz: Tensor
    collision_energy: Tensor
    # Length S: display metadata excluding peaks, conditions and assignments.
    metadata: tuple[dict, ...]

    def to(self, device):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )


@dataclass(frozen=True)
class FormulaData(TensorTable):
    # [F, D]: unique elemental counts; the last column is charge.
    composition: Tensor
    # [F]: exact m/z; formulas with negative element counts are marked invalid.
    mz: Tensor
    valid: Tensor


@dataclass(frozen=True)
class FormulaExplanationData(TensorTable):
    formula_index: Tensor
    compound_index: Tensor
    fragment_ion_adduct_index: Tensor
    hydrogen_shift_index: Tensor


@dataclass(frozen=True)
class PeakAssignmentData(TensorTable):
    # One CSR row per flattened observed peak; no duplicated formula IDs.
    formula_explanation_ptr: Tensor
    formula_explanation_index: Tensor


@dataclass(frozen=True)
class AnnotatedSpectrumData(TensorTable):
    spectra: SpectrumData
    assignments: PeakAssignmentData


@dataclass(frozen=True)
class SpectrumExplanationData(TensorTable):
    formula_explanation_ptr: Tensor
    formula_explanation_index: Tensor


@dataclass(frozen=True)
class SpectrumFormulaTarget(TensorTable):
    spectrum_index: Tensor
    formula_index: Tensor
    observed: Tensor
    intensity: Tensor
    supervision_mask: Tensor
