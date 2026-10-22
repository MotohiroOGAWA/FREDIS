"""Independent set identity, reference remapping, and context separation."""

from dataclasses import fields

import pytest
import torch

from fredis.ml.input import (
    ReactionSet,
    SourceCompound,
    ReactionContextData,
    SourceReactionStructure,
)
from fredis.ml.input.reaction_batching import select_samples
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.libs.mmkit.mmkit import Compound, Adduct
from .TestReactionPipeline import config


def test_set_membership_is_independent_and_order_invariant():
    table, mapping = ReactionSet.from_rows([[], [1, 0, 1], [0, 1], [1], []])
    assert [f.name for f in fields(table)] == ["reaction_ptr", "reaction_index"]
    assert table.rows() == [[], [0, 1], [1]]
    assert mapping.tolist() == [0, 1, 1, 2, 0]
    table.validate(torch.tensor([0, 0]))
    ptr, values = table.gather(torch.tensor([1, 0, 1]))
    assert ptr.tolist() == [0, 2, 2, 4]
    assert values.tolist() == [0, 1, 0, 1]
    assert set(f.name for f in fields(ReactionContextData)).isdisjoint(
        {"candidate_reaction_ptr", "candidate_reaction_index", "compound_index"}
    )


def test_collation_offsets_reaction_ids_then_deduplicates_empty_sets():
    table, _ = ReactionSet.from_rows([[], [0], [1], [0, 1]])
    merged, maps = ReactionSet.collate([table, table], [0, 2])
    assert merged.rows() == [[], [0], [1], [0, 1], [2], [3], [2, 3]]
    assert maps[0].tolist() == [0, 1, 2, 3]
    assert maps[1].tolist() == [0, 4, 5, 6]
    merged.validate(torch.tensor([0, 0, 1, 1]))
    # One registry reused by two groups: every set, not only {}, is shared.
    same, maps = ReactionSet.collate([table, table], [0, 0])
    assert same.rows() == table.rows()
    assert torch.equal(maps[0], maps[1])


def test_validation_rejects_invalid_membership_and_cross_source_sets():
    table, _ = ReactionSet.from_rows([[0, 1]])
    with pytest.raises(ValueError, match="different SourceCompounds"):
        table.validate(torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="missing concrete reaction"):
        table.validate(torch.tensor([0]))
    duplicate = ReactionSet(torch.tensor([0, 1, 2]), torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="unique sorted sets"):
        duplicate.validate(torch.tensor([0]))
    with pytest.raises(ValueError, match="nonnegative"):
        ReactionSet.from_rows([[-1]])


def test_prepared_groups_share_sets_and_compound_inverse_ids(tmp_path):
    generator = create_spectrum_generator(config()).eval()
    source = Compound.from_smiles("CC(O)N")
    adducts = [Adduct.parse("[M+H]+"), Adduct.parse("[M+Na]+")]
    data, _ = ReactionStructureBuilder(generator).build(
        source,
        adducts,
        [10.0, 30.0],
        [[18.033826], [18.033826]],
        [[1.0], [1.0]],
    )
    data.validate()
    assert isinstance(data.source_compounds, SourceCompound)
    assert data.source_compounds.symbols == generator.mol_encoder.graph_builder.symbols
    contexts = data.states
    assert data.num_branch_groups == 2
    first = contexts.candidate_reaction_set_index[
        contexts.candidate_branch_group_index == 0
    ]
    second = contexts.candidate_reaction_set_index[
        contexts.candidate_branch_group_index == 1
    ]
    assert set(first.tolist()) == set(second.tolist())
    assert len(data.reaction_sets) < contexts.candidate_reaction_set_index.numel()
    assert data.reaction_sets.rows().count([]) == 1
    compounds = data.annotations.compounds
    assert compounds.graph.num_graphs == len(set(compounds.smiles))
    for c in range(len(compounds.smiles)):
        ids = compounds.generating_reaction_sets_index[
            compounds.generating_reaction_sets_ptr[c] : compounds.generating_reaction_sets_ptr[c + 1]
        ]
        assert ids.tolist() == sorted(set(ids.tolist()))
    # Selecting a different adduct keeps the immutable registry/set tables and
    # remaps context, observation, and topology rows, not canonical memberships.
    selected = select_samples(data, [1])
    selected.validate()
    assert selected.reaction_sets is data.reaction_sets
    assert selected.source_compounds is data.source_compounds
    path = tmp_path / "shared.preft.pt"
    selected.save(path)
    loaded = SourceReactionStructure.load(path)
    loaded.validate()
    assert loaded.reaction_sets.rows() == data.reaction_sets.rows()
    assert loaded.states.candidate_reaction_set_index.tolist() == second.tolist()


def test_shared_empty_set_preserves_distinct_source_compounds():
    from fredis.ml.input.source_reaction_structure import prepare_source_reactions

    generator = create_spectrum_generator(config())
    structures = [
        prepare_source_reactions(
            source=Compound.from_smiles(smiles),
            reactions=(),
            graph_builder=generator.mol_encoder.graph_builder,
            collision_energy_eV=torch.tensor([10.0]),
            branch_group_adduct_index=torch.tensor([0]),
            max_reaction_count=1,
        )
        for smiles in ("CCO", "CC(=O)O")
    ]
    combined = SourceReactionStructure.from_structures(structures)
    combined.validate()
    assert combined.reaction_sets.rows() == [[]]
    assert combined.states.candidate_reaction_set_index.tolist() == [0, 0]
    assert combined.branch_group_tree_index.tolist() == [0, 1]
    assert combined.source_compounds.source_smiles == ("CCO", "CC(=O)O")


def test_old_prepared_module_has_an_explicit_regeneration_error():
    from unittest.mock import patch

    error = ModuleNotFoundError(name="fredis.ml.input.reaction_state_data")
    with patch("torch.load", side_effect=error):
        with pytest.raises(ValueError, match="regenerate the dataset"):
            SourceReactionStructure.load("old.preft.pt")
