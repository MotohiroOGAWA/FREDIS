import math
from unittest.mock import patch
import pytest
import torch
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.components.reaction.reaction_compatibility import (
    ReactionCompatibilityEngine,
)
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from .TestReactionPipeline import config
from .test_canonical_workflow import prepared


def prepared_two_ce():
    return prepared()


def test_ce_spectra_share_one_branch_group_and_tree():
    generator, data = prepared_two_ce()
    assert data.num_samples == 2 and data.num_branch_groups == 1
    assert data.sample_branch_group_index.tolist() == [0, 0]
    generator.eval()
    source = Compound.from_smiles("CC(O)N")
    adduct = Adduct.parse("[M+H]+")
    output = generator.predict([source, source], [adduct, adduct], [10.0, 40.0])
    assert len(output.fragments.trees) == 1
    first = output.spectra.intensity[output.spectra.sample_index == 0]
    second = output.spectra.intensity[output.spectra.sample_index == 1]
    assert first.shape == second.shape and not torch.equal(first, second)


def test_candidate_mask_requires_mutual_reaction_center_retention():
    engine = ReactionCompatibilityEngine(max_reaction_count=3)
    invalidation = torch.zeros((1, 4, 4), dtype=torch.bool)
    invalidation[0, 0, 2] = True
    invalidation[0, 3, 0] = True
    expansion = engine.expand(
        state_reaction_index=torch.tensor([[0, -1, -1]]),
        state_sample_index=torch.tensor([0]),
        pool_valid=torch.ones((1, 4), dtype=torch.bool),
        pool_conflict=torch.zeros((1, 4, 4), dtype=torch.bool),
        pool_invalidation=invalidation,
        pool_retained=torch.full((1, 4, 1), 15, dtype=torch.long),
        source_atom_valid=torch.full((1, 1), 15, dtype=torch.long),
    )
    assert expansion.valid.tolist() == [False, True, False, False]


def test_fragment_node_budget_is_shared_across_collision_energies():
    value = config()
    value["generation_params"]["max_fragment_nodes"] = 2
    generator = create_spectrum_generator(value).eval()
    source = Compound.from_smiles("CC(O)N")
    adduct = Adduct.parse("[M+H]+")
    outputs = list(
        generator.predict_batches(
            [source, source], [adduct, adduct], [10.0, 40.0], max_samples=1
        )
    )
    assert len(outputs) == 2
    for output in outputs:
        core = output.spectrum_batch.canonical_data
        assert core.execution_graph.compound_index.unique().numel() <= 2


def test_condition_states_keep_equivalent_formulas_separate_until_output():
    generator, data = prepared_two_ce()
    batch = data.spectrum_batch
    rows = (
        ((batch.state_sample_index == 0) & (batch.state_node_index == 0))
        .nonzero()
        .flatten()
    )
    conditions = generator.spectrum_model.main_adduct_conditions["[M+H]+"]
    pairs = set(
        zip(
            (batch.ion_type_strs[int(i)] for i in batch.state_adduct_index[rows]),
            batch.state_hydrogen_shift[rows].tolist(),
        )
    )
    assert pairs == set(conditions)
    assert torch.unique(batch.state_formula_tensor[rows], dim=0).shape[0] < rows.numel()


def test_shared_hydrogen_shift_range_is_used_for_every_adduct():
    generator, _ = prepared_two_ce()
    rules = generator.fragmenter.adduct_rule_set
    tree = generator.fragmenter.build_fragment_ion_tree(Compound.from_smiles("CCO"))
    for adduct in rules.adduct_types:
        assert set(
            tree.get_formula_candidate_group(adduct).hydrogen_shifts.tolist()
        ) == set(range(rules.hydrogen_shift_min, rules.hydrogen_shift_max + 1))


def test_prepared_payload_has_schema_without_version(tmp_path):
    _, data = prepared_two_ce()
    path = tmp_path / "data.preft.pt"
    data.save(path)
    payload = torch.load(path, weights_only=False)
    assert set(payload) == {"schema", "structure"}
    assert payload["schema"] == "fredis.annotated-spectrum-training"
    assert SourceReactionStructure.load(path).num_branch_groups == 1
