"""Incremental inference, pruning and reuse of saved chemistry without teachers."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch, PropertyMock
import math

import pytest
import torch
from torch import nn

from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.incremental_generation import _Chemistry, _saved_registry
from fredis.domain.fragment.reaction import ReactionSequence
from fredis.ml.specgen.generation import GenerationSettings
from .TestReactionPipeline import config


def test_full_budget_replaces_low_probability_nodes_and_continues_to_max_depth():
    class Head(nn.Module):
        def forward(self, x):
            # Feature columns are parent, child, reaction, main adduct.
            probabilities = x.new_tensor([1.0, 0.9, 0.1, 0.05, 0.9, 0.9])
            return torch.logit(probabilities[x[:, 1].long()])[:, None]

    class Zero(nn.Module):
        def forward(self, x):
            return x.new_zeros((len(x), 1), dtype=torch.float32)

    search = object.__new__(_Chemistry)
    feature = SimpleNamespace(
        main_adduct_projection=nn.Identity(), adduct_encoder=Zero(), edge_head=Head()
    )
    search.generator = SimpleNamespace(
        feature_model=feature,
        max_samples=2,
        fragmenter=SimpleNamespace(tree_max_reaction_count=3),
        generation=GenerationSettings(max_fragment_nodes=4),
    )
    search.device = "cpu"
    search.static = (torch.zeros((5, 1)), torch.zeros((1, 1)))
    search.reactions = [None] * 5
    search.relations = SimpleNamespace(rejection=lambda _: None)
    search.embeddings = {i: torch.tensor([float(i)]) for i in range(6)}
    search.encode = lambda _: None
    targets = {(0,): 1, (1,): 2, (2,): 3, (0, 3): 4, (0, 3, 4): 5}
    attempted = []

    def materialize(members):
        attempted.append(members)
        return targets.get(members)

    search.materialize = materialize
    states, edges = search.search(0)
    assert (0, 3, 4) in attempted  # The budget was already full at depth one.
    assert {cid for cid, _ in states.values()} == {0, 1, 4, 5}
    assert set(states) == {(), (0,), (0, 3), (0, 3, 4)}
    assert states[(0, 3, 4)][1] == pytest.approx(3 * math.log(0.9))
    assert all(parent in states and child in states for parent, child, _ in edges)


def make_prepared(smiles="CC(O)N", *, depth=3, save_all_edges=False):
    torch.manual_seed(42)
    cfg = config()
    cfg["fragmenter_params"]["fragment_ion_tree_builder"]["max_reaction_count"] = depth
    model = create_spectrum_generator(cfg).eval()
    source = Compound.from_smiles(smiles)
    adduct = Adduct.parse("[M+H]+")
    data, _ = ReactionStructureBuilder(model).build(
        source,
        [adduct, adduct],
        [10.0, 40.0],
        [[18.033826], [44.049476]],
        [[1.0], [1.0]],
        save_all_edges=save_all_edges,
    )
    return model, source, adduct, data


def test_saved_matches_restore_exact_reactions_and_relations():
    model, source, _, data = make_prepared()
    original = model.fragmenter.fragment_ion_tree_builder.create_reaction_instances(
        source
    )
    _, _, restored, relations, registry, _ = _saved_registry(model, data, 0)
    assert restored == original
    for field in registry.field_names():
        torch.testing.assert_close(
            getattr(registry, field), getattr(data.reactions, field)
        )
    from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations

    assert relations == ReactionRelations.from_reactions(original)


def test_saved_chemistry_avoids_rematching_materializing_and_rebuilding_known_graphs():
    model, _, _, data = make_prepared(depth=1, save_all_edges=True)
    tree_builder = type(model.fragmenter.fragment_ion_tree_builder)
    graph_builder = type(model.mol_encoder.graph_builder)
    with (
        patch.object(
            Compound,
            "smiles",
            new_callable=PropertyMock,
            side_effect=AssertionError("known structure canonicalized again"),
        ),
        patch.object(
            tree_builder,
            "create_reaction_instances",
            side_effect=AssertionError("rematching"),
        ),
        patch.object(
            tree_builder, "build", side_effect=AssertionError("exhaustive search")
        ),
        patch.object(
            ReactionSequence,
            "compile",
            side_effect=AssertionError("known chemistry rebuilt"),
        ),
        patch.object(
            graph_builder, "build", side_effect=AssertionError("known graph rebuilt")
        ),
    ):
        outputs = list(model.predict_prepared_batches(data))
    assert len(outputs) == 1
    outputs[0].spectrum_batch.canonical_data.validate()
    assert torch.isfinite(outputs[0].spectra.intensity).all()


def test_saved_teacher_pruning_does_not_limit_prediction_and_labels_do_not_leak():
    model, source, adduct, data = make_prepared()
    reference = model.predict([source, source], [adduct, adduct], [10.0, 40.0])
    changed = replace(
        data,
        reaction_contexts=replace(
            data.reaction_contexts, edge_label=1 - data.reaction_contexts.edge_label
        ),
        targets=replace(
            data.targets, intensity=torch.zeros_like(data.targets.intensity)
        ),
    )
    # The prepared graph is teacher-pruned. Only its chemistry cache is reusable.
    result = list(model.predict_prepared_batches(changed))[0]
    torch.testing.assert_close(
        reference.spectra.formula_tensor, result.spectra.formula_tensor
    )
    torch.testing.assert_close(
        reference.spectra.intensity, result.spectra.intensity, rtol=1e-4, atol=1e-6
    )
    core = result.spectrum_batch.canonical_data
    assert not hasattr(core, "targets") and core.reaction_contexts.edge_label is None


def test_saved_multisource_registry_offsets_and_reordered_conditions():
    from fredis.ml.input.source_reaction_structure import SourceReactionStructure

    model, _, _, first = make_prepared("CCCO", depth=2)
    _, _, _, second = make_prepared("CCO", depth=2)
    combined = SourceReactionStructure.from_structures([first, second]).select_samples(
        [3, 0, 2, 1]
    )
    expected = {}
    for output in model.predict_prepared_batches(combined, max_samples=2):
        for local, original in enumerate(output.sample_input_index):
            mask = output.spectra.sample_index == local
            expected[original] = (
                output.spectra.formula_tensor[mask],
                output.spectra.intensity[mask],
            )
    assert set(expected) == set(range(4))
    for index in range(4):
        single = list(model.predict_prepared_batches(combined.select_samples([index])))[
            0
        ]
        torch.testing.assert_close(expected[index][0], single.spectra.formula_tensor)
        torch.testing.assert_close(
            expected[index][1], single.spectra.intensity, rtol=1e-4, atol=1e-6
        )
