"""Canonical reference integrity and ambiguous chemical supervision."""

from dataclasses import replace
import pytest
import torch
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.input.annotated_spectrum_data import PeakAssignmentData
from .test_canonical_workflow import prepared


@pytest.mark.parametrize(
    "defect", ["source_mapping", "branch_labels", "edge_id", "assignment", "target"]
)
def test_invalid_references_and_column_lengths_are_rejected(defect):
    _, data = prepared()
    if defect == "source_mapping":
        data = replace(
            data,
            source_compounds=replace(
                data.source_compounds, compound_index=torch.tensor([9999])
            ),
        )
    elif defect == "branch_labels":
        data = replace(
            data,
            reaction_contexts=replace(
                data.reaction_contexts, edge_label=torch.tensor([1], dtype=torch.int8)
            ),
        )
    elif defect == "edge_id":
        data = replace(
            data,
            reaction_contexts=replace(
                data.reaction_contexts,
                edge_index=torch.full_like(data.reaction_contexts.edge_index, 9999),
            ),
        )
    elif defect == "assignment":
        data = replace(
            data,
            annotations=replace(
                data.annotations,
                assignments=replace(
                    data.annotations.assignments,
                    formula_explanation_index=torch.full_like(
                        data.annotations.assignments.formula_explanation_index, 9999
                    ),
                ),
            ),
        )
    else:
        data = replace(data, targets=replace(data.targets, intensity=torch.zeros(1)))
    with pytest.raises(ValueError):
        data.validate()


def test_multiple_explanations_share_formula_intensity_without_structural_duplication(
    tmp_path,
):
    _, data = prepared()
    e = data.formula_explanations
    formula = int((torch.bincount(e.formula_index) > 1).nonzero()[0])
    matches = (e.formula_index == formula).nonzero().flatten()[:2]
    candidate = set(
        data.spectrum_explanations.formula_explanation_index[
            : data.spectrum_explanations.formula_explanation_ptr[1]
        ].tolist()
    )
    assert set(matches.tolist()) <= candidate
    assignments = PeakAssignmentData(torch.tensor([0, 2, 2, 2, 2]), matches)
    data = replace(
        data,
        annotations=replace(data.annotations, assignments=assignments),
        targets=None,
    ).with_targets()
    data.validate()
    data.save(tmp_path / "ambiguous.pt")
    loaded = SourceReactionStructure.load(tmp_path / "ambiguous.pt")
    mask = (loaded.targets.spectrum_index == 0) & (
        loaded.targets.formula_index == formula
    )
    assert loaded.targets.intensity[mask].item() == 0.5
    assert len(loaded.sample_annotations[0]["peaks"][0]["matches"]) == 2


def test_collation_requires_identical_chemistry_vocabulary():
    _, data = prepared()
    other = replace(
        data,
        adducts=replace(
            data.adducts, hydrogen_shifts=data.adducts.hydrogen_shifts.flip(0)
        ),
    )
    with pytest.raises(ValueError, match="vocabularies"):
        SourceReactionStructure.from_structures([data, other])
