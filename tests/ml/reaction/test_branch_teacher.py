"""Prepared peak-specific branch reaction history graphs and runtime balancing."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from fredis.ml.input.branch_teacher import BranchTeacher
from fredis.ml.input.compound_reaction_graph import ReactionExecutionGraph
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.training.fragment_tree_training.branch_mil import branch_path_mil_loss
from fredis.ml.training.fragment_tree_training.model import (
    ReactionFragmentTreeTrainingModel,
)
from .test_canonical_workflow import prepared


def graph_fixture():
    graph = ReactionExecutionGraph(
        torch.arange(7),
        torch.zeros(7, dtype=torch.long),
        torch.tensor([0, 0, 1, 2, 1, 2]),
        torch.tensor([1, 2, 3, 4, 5, 6]),
        torch.arange(6),
        torch.tensor([0]),
        torch.arange(7),
    )
    return graph, torch.tensor([0, 1, 1, 2, 2, 2, 2])


def test_peak_specific_paths_and_class_balance_match_explicit_objective():
    graph, depth = graph_fixture()
    teacher = BranchTeacher.from_peak_edges(
        graph, depth, [(0, 0, [0, 2], [3]), (1, 0, [1, 3], [4])]
    )
    # A positive path for another observed peak is never a negative alternative.
    assert teacher.edge_label.tolist() == [1, 1, 1, 1, 0, 0]
    assert teacher.negative_edge_index.tolist() == [4, 5]
    data = SimpleNamespace(branch_teacher=teacher, num_branch_groups=1)
    for weight in (0.0, 0.25, 2.0):
        logits = torch.tensor([-3.0, 2.0, 1.0, -2.0, 0.5, -0.5], requires_grad=True)
        actual = branch_path_mil_loss(logits, data, negative_weight=weight)[0]
        scores = -F.logsigmoid(logits)
        expected = torch.stack(
            [
                (
                    scores[0]
                    + (scores[0] + scores[2] + weight * F.softplus(logits[4]))
                    / (1 + weight)
                )
                / 2,
                (
                    scores[1]
                    + (scores[1] + scores[3] + weight * F.softplus(logits[5]))
                    / (1 + weight)
                )
                / 2,
            ]
        ).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits)[0],
            torch.autograd.grad(expected, logits)[0],
        )


def test_one_ambiguous_peak_pools_paths_instead_of_supervising_each_target():
    graph, depth = graph_fixture()
    logits = torch.tensor([-4.0, 4.0, -4.0, 4.0, 0.0, 0.0], requires_grad=True)
    ambiguous = BranchTeacher.from_peak_edges(
        graph, depth, [(0, 0, [0, 1, 2, 3], [3, 4])]
    )
    separate = BranchTeacher.from_peak_edges(
        graph, depth, [(0, 0, [0, 2], [3]), (1, 0, [1, 3], [4])]
    )
    first = branch_path_mil_loss(
        logits,
        SimpleNamespace(branch_teacher=ambiguous, num_branch_groups=1),
        negative_weight=0,
    )[0]
    second = branch_path_mil_loss(
        logits,
        SimpleNamespace(branch_teacher=separate, num_branch_groups=1),
        negative_weight=0,
    )[0]
    assert first < second / 10
    first.backward()
    assert torch.isfinite(logits.grad).all()


def test_saved_teachers_survive_selection_collation_and_training_without_rebuild(
    tmp_path,
):
    generator, data = prepared()
    path = tmp_path / "prepared.preft.pt"
    data.save(path)
    payload = torch.load(path, weights_only=False)
    assert payload["structure"].branch_teacher is not None
    assert "execution_graph" not in payload["structure"].__dict__
    with (
        patch.object(
            BranchTeacher,
            "from_preparation",
            side_effect=AssertionError("teacher rebuild in training"),
        ),
        patch.object(
            BranchTeacher,
            "from_peak_edges",
            side_effect=AssertionError("teacher regrouping in training"),
        ),
    ):
        loaded = SourceReactionStructure.load(path)
        selected = loaded.select_samples([1, 0, 1])
        selected.validate()
        batch = SourceReactionStructure.from_structures([selected, loaded])
        batch.validate()
        assert (
            batch.branch_teacher.peak_index.numel()
            == selected.branch_teacher.peak_index.numel()
            + loaded.branch_teacher.peak_index.numel()
        )
        expected = batch.branch_teacher.graph
        assert batch.execution_graph is expected
        training = ReactionFragmentTreeTrainingModel(
            generator.feature_model, spectrum_model=generator.spectrum_model
        )
        output = training(batch)
        output.loss.backward()
        assert torch.isfinite(output.loss)
        assert not hasattr(batch, "_branch_mil_topology")
    prediction = data.for_prediction()
    assert not hasattr(prediction, "branch_teacher")


def test_changing_branch_peak_units_keeps_spectrum_loss_and_outputs_identical():
    generator, data = prepared()
    generator.eval()
    groups = torch.arange(data.num_branch_groups)
    first_spectrum_peaks = torch.arange(int(data.annotations.spectra.peak_ptr[1]))
    changed = replace(
        data,
        branch_teacher=data.branch_teacher.select(first_spectrum_peaks, groups, groups),
    )
    changed.validate()
    assert (
        changed.branch_teacher.peak_index.numel()
        < data.branch_teacher.peak_index.numel()
    )
    training = ReactionFragmentTreeTrainingModel(
        generator.feature_model, spectrum_model=generator.spectrum_model
    ).eval()
    before, after = training(data), training(changed)
    torch.testing.assert_close(
        before.spectrum_output.loss, after.spectrum_output.loss, rtol=0, atol=0
    )
    torch.testing.assert_close(
        before.spectrum_output.intensity,
        after.spectrum_output.intensity,
        rtol=0,
        atol=0,
    )
    assert changed.targets is data.targets and changed.annotations is data.annotations


def test_preparation_retains_exact_peak_paths_instead_of_compound_edge_labels():
    original = BranchTeacher.from_preparation
    captured = []

    def capture(data, raw):
        captured.append(raw)
        return original(data, raw)

    with patch.object(BranchTeacher, "from_preparation", side_effect=capture):
        _, data = prepared()
    # Compound-level diagnostics cannot change which raw peak paths are correct.
    contexts = replace(
        data.reaction_contexts,
        edge_label=torch.zeros_like(data.reaction_contexts.edge_label),
    )
    changed = replace(data, reaction_contexts=contexts, branch_teacher=None)
    rebuilt = original(changed, captured[-1])
    for name in (
        "edge_label",
        "peak_index",
        "node_state_index",
        "source_index",
        "target_index",
        "edge_index",
        "bag_peak_index",
        "bag_depth",
        "negative_edge_index",
    ):
        torch.testing.assert_close(
            getattr(data.branch_teacher, name), getattr(rebuilt, name)
        )
    assert (rebuilt.edge_label == 1).any()


def test_missing_prepared_branch_teacher_is_rejected_without_runtime_fallback(tmp_path):
    import pytest
    from fredis.ml.input.source_reaction_structure import SCHEMA

    _, data = prepared()
    missing = replace(data, branch_teacher=None)
    with pytest.raises(ValueError, match="regenerate prepared datasets"):
        missing.save(tmp_path / "missing.preft.pt")
    path = tmp_path / "missing.preft.pt"
    torch.save(dict(schema=SCHEMA, structure=missing), path)
    with pytest.raises(ValueError, match="regenerate prepared datasets"):
        SourceReactionStructure.load(path)


def test_shorter_target_is_preserved_when_it_is_also_an_intermediate():
    graph, depth = graph_fixture()
    teacher = BranchTeacher.from_peak_edges(graph, depth, [(0, 0, [0, 2], [1, 3])])
    assert teacher.node_state_index[teacher.target_node_index].tolist() == [1, 3]
