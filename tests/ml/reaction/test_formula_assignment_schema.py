"""Chemical registry foreign keys and ambiguous observed-peak assignments."""

from dataclasses import fields, replace

import pytest
import torch

from fredis.ml.input import (
    BranchGroup,
    FormulaData,
    FormulaExplanationData,
    PeakAssignmentData,
    SpectrumData,
)
from fredis.ml.input.reaction_batching import select_samples
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from .test_fragment_tree_model import prepared_two_ce


def test_schema_columns_and_shared_adduct_membership():
    generator, data = prepared_two_ce()
    assert {f.name for f in fields(FormulaData)} == {"composition", "mz", "valid"}
    assert {f.name for f in fields(FormulaExplanationData)} == {
        "formula_index",
        "compound_index",
        "fragment_ion_adduct_index",
        "hydrogen_shift_index",
    }
    assert {f.name for f in fields(PeakAssignmentData)} == {
        "formula_explanation_ptr",
        "formula_explanation_index",
    }
    assert {f.name for f in fields(SpectrumData)} == {
        "peak_ptr",
        "mz",
        "intensity",
        "precursor",
        "branch_group_index",
        "precursor_mz",
        "collision_energy",
        "metadata",
    }
    assert "sample_tree_index" not in vars(data)
    assert data.formulas.composition.dtype == torch.long
    adducts = data.adducts
    assert adducts.fragment_ion_adduct_ptr.shape == (len(adducts.main_adducts) + 1,)
    for i, rule in enumerate(generator.fragmenter.adduct_rule_set.adduct_rules):
        start, stop = adducts.fragment_ion_adduct_ptr[i : i + 2]
        labels = tuple(
            adducts.fragment_ion_adducts[j]
            for j in adducts.fragment_ion_adduct_index[start:stop]
        )
        assert labels == tuple(str(a) for a in rule.fragment_ion_adducts)
    data.validate()


def test_peak_with_multiple_formulas_and_multiple_explanations_survives_lifecycle(
    tmp_path,
):
    _, data = prepared_two_ce()
    annotations = data.annotations
    explanations = data.formula_explanations
    counts = torch.bincount(explanations.formula_index)
    formula = int((counts > 1).nonzero()[0])
    same = (explanations.formula_index == formula).nonzero().flatten()[:2]
    other = (explanations.formula_index != formula).nonzero().flatten()[:1]
    selected = torch.cat((same, other)).sort().values
    # All registry explanations here are present in both CE observations.
    assignment = PeakAssignmentData(
        torch.tensor([0, 3, 3, 3, 3]),
        selected,
    )
    data = replace(
        data, annotations=replace(annotations, assignments=assignment)
    ).with_targets()
    data.validate()
    path = tmp_path / "ambiguous.preft.pt"
    data.save(path)
    loaded = SourceReactionStructure.load(path)
    combined = SourceReactionStructure.from_structures([loaded, loaded])
    subset = select_samples(combined, [2, 0])
    subset.validate()
    for sample in subset.sample_annotations:
        matches = sample["peaks"][0]["matches"]
        assert len(matches) == 3
        assert len({match["formula"] for match in matches}) == 2
        assert sample["peaks"][1]["matches"] == []


def test_observed_adduct_is_metadata_and_main_adduct_is_derived():
    _, data = prepared_two_ce()
    spectra = data.annotations.spectra
    metadata = tuple(
        {**row, "observed_adduct": "[M+H-H2O]+"} for row in spectra.metadata
    )
    data = replace(
        data,
        annotations=replace(
            data.annotations, spectra=replace(spectra, metadata=metadata)
        ),
    )
    assert all("mainAdduct" not in row for row in metadata)
    for sample in data.sample_annotations:
        assert sample["adduct"] == data.conditions.observed_adducts[0]
        assert sample["mainAdduct"] == "[M+H]+"
    stripped = tuple(
        {k: v for k, v in row.items() if k != "observed_adduct"} for row in metadata
    )
    data = replace(
        data,
        annotations=replace(
            data.annotations, spectra=replace(spectra, metadata=stripped)
        ),
    )
    assert all(sample["adduct"] == "[M+H]+" for sample in data.sample_annotations)


@pytest.mark.parametrize(
    "defect",
    ["foreign_key", "duplicate_assignment", "disallowed_adduct", "duplicate_group"],
)
def test_invalid_registry_references_are_rejected(defect):
    _, data = prepared_two_ce()
    if defect == "foreign_key":
        e = data.formula_explanations
        data = replace(
            data,
            formula_explanations=replace(
                e, formula_index=torch.full_like(e.formula_index, len(data.formulas.mz))
            ),
        )
    elif defect == "duplicate_assignment":
        a = data.annotations.assignments
        data = replace(
            data,
            annotations=replace(
                data.annotations,
                assignments=replace(
                    a,
                    formula_explanation_ptr=torch.tensor([0, 2, 2, 2, 2]),
                    formula_explanation_index=torch.tensor([0, 0]),
                ),
            ),
        )
    elif defect == "disallowed_adduct":
        registry = replace(
            data.adducts,
            fragment_ion_adduct_ptr=torch.zeros_like(
                data.adducts.fragment_ion_adduct_ptr
            ),
            fragment_ion_adduct_index=torch.empty(0, dtype=torch.long),
        )
        data = replace(data, adducts=registry)
    else:
        data = replace(
            data, branch_groups=BranchGroup(torch.tensor([0, 0]), torch.tensor([0, 0]))
        )
    with pytest.raises(ValueError):
        data.validate()


def test_collation_rejects_adduct_ids_with_different_meanings():
    _, data = prepared_two_ce()
    other = replace(
        data,
        adducts=replace(
            data.adducts, hydrogen_shifts=data.adducts.hydrogen_shifts.flip(0)
        ),
    )
    with pytest.raises(ValueError, match="vocabularies"):
        SourceReactionStructure.from_structures([data, other])
