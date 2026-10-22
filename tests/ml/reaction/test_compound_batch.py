"""Compound storage follows the canonical structure/generating-set schema."""

from dataclasses import fields
from unittest.mock import patch

import torch

from fredis.ml.input import CompoundBatch, SourceReactionStructure
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.mol.selected_descriptors import selected_descriptor_values
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.domain.molecule.descriptors import compute_descriptor_values
from fredis.libs.mmkit.mmkit import Compound, Adduct
from .TestReactionPipeline import config


def test_compound_batch_stores_no_duplicate_identity_or_reference_side_columns(
    tmp_path,
):
    generator = create_spectrum_generator(config())
    source = Compound.from_smiles("CCO")
    data, _ = ReactionStructureBuilder(generator).build(
        source,
        [Adduct.parse("[M+H]+")],
        [20.0],
        [[47.049141]],
        [[1.0]],
    )
    table = data.compounds
    assert isinstance(table, CompoundBatch)
    assert {f.name for f in fields(table)} == {
        "compounds",
        "graph",
        "generating_reaction_sets_ptr",
        "generating_reaction_sets_index",
    }
    assert isinstance(table.compounds, tuple)
    assert all(isinstance(compound, Compound) for compound in table.compounds)
    assert "smiles" not in vars(table)
    assert table.smiles == tuple(compound.smiles for compound in table.compounds)
    assert table.graph.num_graphs == len(set(table.smiles))
    assert table.generating_reaction_sets_ptr.shape == (len(table.compounds) + 1,)
    for graph in (
        data.source_graph,
        table.graph,
        data.spectrum_batch.decoded.node_graph,
    ):
        assert "descriptors" not in graph
    assert "occurrences" not in vars(data.annotations)
    assert data.formula_explanations.compound_index.max() < len(table.compounds)
    path = tmp_path / "compound-batch.preft.pt"
    data.save(path)
    loaded = SourceReactionStructure.load(path)
    loaded.validate()
    assert isinstance(loaded.compounds, CompoundBatch)
    assert "smiles" not in vars(loaded.compounds)
    assert loaded.compounds.smiles == table.smiles
    assert "descriptors" not in loaded.spectrum_batch.decoded.node_graph


def test_descriptors_are_computed_only_after_model_selection_and_on_demand():
    selected_descriptor_values.cache_clear()
    params = config()
    params["spectrum_model_params"]["descriptors"] = ["TPSA", "MolLogP"]
    generator = create_spectrum_generator(params).eval()
    source = Compound.from_smiles("CCO")
    with patch(
        "fredis.ml.mol.selected_descriptors.compute_descriptor_values",
        wraps=compute_descriptor_values,
    ) as compute:
        data, _ = ReactionStructureBuilder(generator).build(
            source,
            [Adduct.parse("[M+H]+")],
            [20.0],
            [[47.049141]],
            [[1.0]],
        )
        assert compute.call_count == 0
        with torch.no_grad():
            first = generator.spectrum_model._encode_molecules(
                data.spectrum_batch, None, None
            )
        assert compute.call_count == len(data.compounds.compounds)
        assert all(
            call.args[1] == ("TPSA", "MolLogP") for call in compute.call_args_list
        )
        count = compute.call_count
        with torch.no_grad():
            second = generator.spectrum_model._encode_molecules(
                data.spectrum_batch, None, None
            )
        assert compute.call_count == count
        torch.testing.assert_close(first, second)
    selected_descriptor_values.cache_clear()
