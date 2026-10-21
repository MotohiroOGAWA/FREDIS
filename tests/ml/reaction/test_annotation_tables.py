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
    c, f, ions, s, matches = a.compounds, a.formulas, a.ions, a.spectra, a.assignments
    assert len(set(c.smiles)) == len(c.smiles) == c.graph.num_graphs
    assert_csr(s.peak_ptr, s.mz, data.num_samples)
    assert_csr(f.ion_ptr, f.ion_index, len(f.composition))
    assert_csr(matches.formula_ptr, matches.formula_index, len(s.mz))
    assert_csr(matches.ion_ptr, matches.ion_index, len(s.mz))
    assert_csr(c.state_ptr, c.state_rows, len(c.smiles))
    assert c.state_index.shape == data.states.candidate_branch_group_index.shape
    assert torch.equal(c.state_index, data.states.compound_index)
    for compound in range(len(c.smiles)):
        rows = c.state_rows[c.state_ptr[compound] : c.state_ptr[compound + 1]]
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
    ptr = data.annotations.compounds.state_ptr
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
