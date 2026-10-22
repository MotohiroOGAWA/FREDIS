"""Normalized identity, dependencies and trainable views of spectrum annotations."""

from unittest.mock import patch
import torch
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.input.reaction_batching import select_samples
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.libs.mmkit.mmkit import Compound, Adduct
from .TestReactionPipeline import config
from .test_fragment_tree_model import prepared_two_ce


def assert_csr(ptr, index, rows):
    assert ptr.shape == (rows + 1,)
    assert ptr[0] == 0 and ptr[-1] == len(index)
    assert (ptr[1:] >= ptr[:-1]).all()


def assert_tables(data):
    a = data.annotations
    states = data.states
    keys = []
    for group, start, stop in zip(
        states.candidate_branch_group_index,
        states.candidate_reaction_ptr[:-1], states.candidate_reaction_ptr[1:],
    ):
        ids = states.candidate_reaction_index[start:stop]
        assert ids.dtype == torch.int64
        assert (ids[1:] > ids[:-1]).all()
        keys.append((int(group), tuple(ids.tolist())))
    assert len(keys) == len(set(keys))
    c, f, ions, s, matches = a.compounds, a.formulas, a.ions, a.spectra, a.assignments
    assert len(set(c.smiles)) == len(c.smiles) == c.graph.num_graphs
    assert_csr(s.peak_ptr, s.mz, data.num_samples)
    assert_csr(f.ion_ptr, f.ion_index, len(f.composition))
    assert_csr(matches.formula_ptr, matches.formula_index, len(s.mz))
    assert_csr(matches.ion_ptr, matches.ion_index, len(s.mz))
    assert_csr(c.reaction_set_ptr, c.reaction_set_index, len(c.smiles))
    assert c.state_index.shape == data.states.candidate_branch_group_index.shape
    assert torch.equal(c.state_index, data.states.compound_index)
    for compound in range(len(c.smiles)):
        rows = c.reaction_set_index[c.reaction_set_ptr[compound] : c.reaction_set_ptr[compound + 1]]
        assert c.state_index[rows].eq(compound).all()
    for formula in range(len(f.composition)):
        rows = f.ion_index[f.ion_ptr[formula] : f.ion_ptr[formula + 1]]
        assert ions.formula_index[rows].eq(formula).all()
    for peak in range(len(s.mz)):
        rows = matches.ion_index[matches.ion_ptr[peak] : matches.ion_ptr[peak + 1]]
        expected = matches.formula_index[matches.formula_ptr[peak] : matches.formula_ptr[peak + 1]]
        assert set(ions.formula_index[rows].tolist()) == set(expected.tolist())
    if data.teacher.peak_index.numel():
        assert data.teacher.peak_index.max() < len(s.mz)
    formula = ions.formula_index[ions.runtime_index]
    torch.testing.assert_close(f.composition[formula], data.spectrum_batch.state_formula_tensor)
    torch.testing.assert_close(f.mz[formula], data.spectrum_batch.state_mz)
    torch.testing.assert_close(
        data.teacher.spectrum_targets(a), data.spectrum_batch.target_intensity
    )


def test_normalized_tables_share_identities_across_ce_and_molecule_batches():
    _, data = prepared_two_ce()
    assert_tables(data)
    a = data.annotations
    # Two CE spectra reuse ion identities, formulas and compound structures.
    runtime = a.ions.runtime_index
    assert runtime.numel() == 2 * a.ions.compound_index.numel()
    n = runtime.numel() // 2
    assert torch.equal(runtime[:n], runtime[n:])
    assert a.spectra.peak_ptr.tolist() == [0, 2, 4]
    batch = SourceReactionStructure.from_structures([data, data])
    assert_tables(batch)
    assert batch.annotations.compounds.smiles == a.compounds.smiles
    assert batch.annotations.ions.compound_index.numel() == a.ions.compound_index.numel()
    assert batch.annotations.spectra.peak_ptr.tolist() == [0, 2, 4, 6, 8]
    assert batch.teacher.peak_index.tolist() == list(range(8))


def test_same_structure_merges_nodes_but_keeps_distinct_reaction_sets():
    generator = create_spectrum_generator(config())
    source = Compound.from_smiles("CC(C)C")
    adduct = Adduct.parse("[M+H]+")
    data, _ = ReactionStructureBuilder(generator).build(
        source, [adduct], [20.0], [[source.formula.exact_mass + 1.007276466]], [[1.0]]
    )
    assert_tables(data)
    tree = data.spectrum_batch.decoded.trees[0]
    assert len({tree.get_node(i).smiles for i in range(tree.num_nodes)}) == tree.num_nodes
    # Symmetry creates several concrete reaction sets yielding an identical compound.
    ptr = data.annotations.compounds.reaction_set_ptr
    assert (ptr[1:] - ptr[:-1]).max() > 1


def test_select_samples_remaps_observed_peaks_formulas_and_teacher_paths():
    _, data = prepared_two_ce()
    batch = SourceReactionStructure.from_structures([data, data])
    selected = select_samples(batch, [3, 0])
    assert_tables(selected)
    assert selected.annotations.spectra.collision_energy.tolist() == [40.0, 10.0]
    assert selected.annotations.spectra.peak_ptr.tolist() == [0, 2, 4]
    assert set(selected.teacher.peak_index.tolist()) == {0, 1, 2, 3}


def test_new_serialization_omits_expanded_caches_and_restores_without_chemistry(tmp_path):
    _, data = prepared_two_ce()
    compact = data.__getstate__()
    assert compact["spectrum_batch"].decoded.node_graph is None
    assert compact["spectrum_batch"].decoded.compounds == ()
    assert compact["spectrum_batch"].state_formula_tensor is None
    assert compact["spectrum_batch"].target_intensity is None
    assert "sample_annotations" not in compact
    path = tmp_path / "normalized.preft.pt"
    data.save(path)
    with patch.object(Compound, "from_smiles", side_effect=AssertionError("chemistry during load")):
        loaded = SourceReactionStructure.load(path)
    assert_tables(loaded)
    # Chemical annotations can be retained without supervision.
    unlabelled = loaded.without_teacher()
    assert unlabelled.annotations is loaded.annotations
    assert unlabelled.teacher.observed.numel() == 0

    combined = SourceReactionStructure.from_structures([unlabelled, unlabelled])
    assert combined.teacher.observed.numel() == 0
    assert combined.annotations.compounds.smiles == loaded.annotations.compounds.smiles
    assert combined.annotations.spectra.peak_ptr.numel() == 5
    selected = select_samples(combined, [3])
    assert selected.teacher.observed.numel() == 0
    assert selected.annotations.spectra.peak_ptr.tolist() == [0, 2]


def test_compound_reaction_set_csr_keeps_sets_without_duplicating_nodes():
    from dataclasses import replace
    import pytest
    from fredis.domain.fragment.reaction import ReactionInstance, ReactionSequence
    from fredis.ml.input.source_reaction_structure import prepare_source_reactions
    from fredis.ml.input.annotated_spectrum_data import AnnotatedSpectrumData
    from fredis.ml.specgen.components.reaction.reaction_decoder import (
        BranchingReactionDecoder, build_reaction_pool,
    )
    from fredis.ml.specgen.materialization import materialize_reaction_states
    from fredis.ml.specgen.fragment_spectrum_model import prepare_spectrum_model_batch

    generator = create_spectrum_generator(config()).eval()
    source = Compound.from_smiles("CCO")
    maps = [atom.GetAtomMapNum() for atom in source.mapped_mol.GetAtoms()]
    universe = frozenset(maps)
    # Distinct no-edit concrete matches: {}, {A}, {B}, {A,B} yield one Compound.
    reactions = tuple(
        ReactionInstance(0, 0, (atom,), universe, frozenset(), frozenset(),
                         product_molecule_id=0)
        for atom in maps[:2]
    )
    a, b = reactions
    ab, ba = ReactionSequence((a, b)), ReactionSequence((b, a))
    assert ab == ba
    paths = [
        ((None, None), (ReactionSequence((a,)), a), (ab, b)),
        ((None, None), (ReactionSequence((b,)), b), (ba, a)),
    ]
    data = prepare_source_reactions(
        source=source, reactions=reactions,
        graph_builder=generator.mol_encoder.graph_builder,
        collision_energy_eV=torch.tensor([20.]),
        branch_group_adduct_index=torch.tensor([0]), max_reaction_count=2,
        teacher_paths=[[paths]],
    )
    assert data.states.candidate_reaction_ptr.tolist() == [0, 0, 1, 3, 4]
    assert data.states.candidate_reaction_index.tolist() == [0, 0, 1, 1]
    full = (data.states.reaction_count == 2).nonzero().item()
    assert (data.states.transition_child_index == full).sum() == 2
    assert data.teacher.peak_path_ptr.tolist() == [0, 2]

    pool = build_reaction_pool(torch.zeros((2, 8)), data)
    decoder = BranchingReactionDecoder(8, 4, 2, num_heads=2).eval()
    output = decoder(pool, torch.zeros((1, 4)))
    assert output.state_reaction_index.tolist() == [[-1, -1], [0, -1], [1, -1], [0, 1]]
    decoded = materialize_reaction_states(
        output, (source,), (reactions,), torch.tensor([0]), generator.mol_encoder.graph_builder,
    )
    assert decoded.trees[0].num_nodes == 1
    assert decoded.node_graph.num_graphs == 1
    assert decoded.materialized_node_index.tolist() == [0, 0, 0, 0]
    batch = prepare_spectrum_model_batch(
        decoded, generator.fragmenter, [Adduct.parse("[M+H]+")],
        generator.tensorizer, sample_branch_group_index=torch.tensor([0]),
        ion_type_strs=generator.ion_type_strs,
    )
    annotation = AnnotatedSpectrumData.from_runtime(
        batch, ({"precursorMz": 47.0491, "collisionEnergy": 20.,
                 "adduct": "[M+H]+", "peaks": []},),
        torch.tensor([0]), torch.tensor([0]), generator.tensorizer.element_order,
        torch.tensor([0, 0, 0, 0]),
    )
    annotation.validate()
    compounds = annotation.compounds
    assert compounds.graph.num_graphs == 1
    assert compounds.reaction_set_ptr.tolist() == [0, 4]  # C + 1, not V + 1
    assert compounds.reaction_set_index.tolist() == [0, 1, 2, 3]
    assert compounds.state_index.tolist() == [0, 0, 0, 0]
    # A duplicate/missing reaction-set ID must not silently pass validation.
    invalid = replace(annotation, compounds=replace(
        compounds, reaction_set_index=torch.tensor([0, 1, 2, 2]),
    ))
    with pytest.raises(ValueError, match="each materialized set once"):
        invalid.validate()


def test_group_local_reaction_sets_survive_collation_selection_and_save(tmp_path):
    generator, data = prepared_two_ce()
    batch = SourceReactionStructure.from_structures([data, data])
    count = data.states.candidate_reaction_index.numel()
    assert torch.equal(batch.states.candidate_reaction_index[:count],
                       batch.states.candidate_reaction_index[count:])
    # Registry IDs shift for the second source; the stored local IDs do not.
    global_ids = batch.candidate_reaction_registry_index
    assert torch.equal(global_ids[count:], global_ids[:count] + len(data.reactions.reaction_type))
    selected = select_samples(batch, [3])
    assert_tables(selected)
    torch.testing.assert_close(selected.states.candidate_reaction_index,
                               data.states.candidate_reaction_index)
    compound = selected.annotations.compounds
    assert compound.reaction_set_ptr.shape == (len(compound.smiles) + 1,)
    path = tmp_path / "local-reaction-sets.preft.pt"
    selected.save(path)
    loaded = SourceReactionStructure.load(path)
    assert_tables(loaded)
    torch.testing.assert_close(loaded.states.candidate_reaction_index,
                               selected.states.candidate_reaction_index)
    torch.testing.assert_close(loaded.annotations.compounds.reaction_set_ptr,
                               compound.reaction_set_ptr)
    torch.testing.assert_close(loaded.annotations.compounds.reaction_set_index,
                               compound.reaction_set_index)


def test_training_uses_group_local_sets_after_source_registry_offsets():
    from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel

    generator, data = prepared_two_ce()
    generator.eval()
    model = ReactionFragmentTreeTrainingModel(generator.feature_model).eval()
    batch = SourceReactionStructure.from_structures([data, data])
    with torch.no_grad():
        single = model(data).branch_logits
        combined = model(batch).branch_logits
    q = single.shape[0]
    torch.testing.assert_close(combined[:q], single)
    torch.testing.assert_close(combined[q:], single)
