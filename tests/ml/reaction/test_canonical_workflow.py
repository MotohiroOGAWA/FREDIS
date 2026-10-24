"""Shared graph training, teacher-free prediction, batching and Workbench regressions."""

from dataclasses import fields, replace
from importlib.util import spec_from_file_location, module_from_spec
from unittest.mock import patch, PropertyMock
import json
import torch
import pytest
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.domain.fragment.reaction import ReactionSequence
from fredis.ml.input.source_reaction_structure import (
    SourceReactionStructure,
    SpectrumPredictionData,
)
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.training.fragment_tree_training.model import (
    ReactionFragmentTreeTrainingModel,
    branch_edge_loss,
)
from .TestReactionPipeline import config


def prepared():
    torch.manual_seed(17)
    model = create_spectrum_generator(config())
    source = Compound.from_smiles("CC(O)N")
    adduct = Adduct.parse("[M+H]+")
    data, _ = ReactionStructureBuilder(model).build(
        source,
        [adduct, adduct],
        [10.0, 40.0],
        [[18.033826, 44.049476], [18.033826, 44.049476]],
        [[0.5, 1.0], [1.0, 0.5]],
    )
    return model, data


def test_canonical_indices_labels_and_shared_prediction_components():
    _, data = prepared()
    data.validate()
    assert "states" not in vars(data) and "teacher" not in vars(data)
    assert "occurrences" not in vars(data.annotations)
    assert set(vars(data.compound_graph)) == {
        "source_compound_index",
        "target_compound_index",
        "reaction_index",
    }
    context = data.reaction_contexts
    assert context.edge_ptr.shape == (data.num_branch_groups + 1,)
    assert context.edge_index.shape == context.edge_label.shape
    assert context.edge_label.dtype == torch.int8 and (context.edge_label == 1).any()
    assert data.source_compounds.compound_index.tolist() == [
        data.compounds.smiles.index(data.source_smiles[0])
    ]
    prediction = data.for_prediction()
    prediction.validate()
    assert type(prediction) is SpectrumPredictionData
    assert not hasattr(prediction, "annotations") and not hasattr(prediction, "targets")
    assert prediction.reaction_contexts.edge_label is None
    for field in [
        "compounds",
        "compound_graph",
        "formulas",
        "formula_explanations",
        "source_compounds",
    ]:
        assert getattr(prediction, field) is getattr(data, field)


def test_branch_weighting_and_ignored_gradient():
    logits = torch.tensor([0.2, -0.7, 1.2], requires_grad=True)
    labels = torch.tensor([1, 0, -1], dtype=torch.int8)
    loss, _, _ = branch_edge_loss(
        logits,
        labels,
        0.25,
        source_index=torch.zeros(3, dtype=torch.long),
        branch_group_index=torch.zeros(3, dtype=torch.long),
    )
    expected = (
        torch.nn.functional.softplus(-logits[0])
        + 0.25 * torch.nn.functional.softplus(logits[1])
    ) / 1.25
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[2] == 0
    ignored = torch.tensor([1.0, -1.0], requires_grad=True)
    loss, _, _ = branch_edge_loss(
        ignored,
        torch.full((2,), -1, dtype=torch.int8),
        source_index=torch.zeros(2, dtype=torch.long),
        branch_group_index=torch.zeros(2, dtype=torch.long),
    )
    loss.backward()
    assert loss == 0 and torch.equal(ignored.grad, torch.zeros(2))


def test_training_and_loading_do_not_run_chemistry(tmp_path):
    model, data = prepared()
    path = tmp_path / "data.preft.pt"
    _ = data.execution_graph
    _ = data.spectrum_batch
    data.save(path)
    assert (
        "execution_graph" not in data.__getstate__()
        and "spectrum_batch" not in data.__getstate__()
    )
    with patch.object(
        Compound, "from_smiles", side_effect=AssertionError("chemistry on load")
    ):
        loaded = SourceReactionStructure.load(path)
    trainer = ReactionFragmentTreeTrainingModel(
        model.feature_model, spectrum_model=model.spectrum_model
    )
    optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-4)
    with patch.object(
        ReactionSequence, "compile", side_effect=AssertionError("chemistry in forward")
    ):
        optimizer.zero_grad()
        out = trainer(loaded)
        out.loss.backward()
        optimizer.step()
    assert torch.isfinite(out.loss)
    labels = loaded.reaction_contexts.edge_label
    predicted = out.branch_logits.detach() >= 0
    for name, mask in {
        "tp": predicted & (labels == 1),
        "tn": ~predicted & (labels == 0),
        "fp": predicted & (labels == 0),
        "fn": ~predicted & (labels == 1),
    }.items():
        assert out.metrics["branch_count_" + name] == mask.sum()
    assert (
        sum(out.metrics["branch_count_" + name] for name in ("tp", "tn", "fp", "fn"))
        == (labels >= 0).sum()
    )
    assert model.feature_model.edge_head[-1].weight.grad.abs().sum() > 0
    assert out.spectrum_output.metrics["probability_mass_error"] < 1e-5
    assert loaded.targets.intensity.tolist() != []
    assert loaded.annotations.spectra.collision_energy.dtype == torch.float64


def test_branch_is_ce_independent_and_intensity_changes_with_ce():
    model, data = prepared()
    model.eval()
    first = model.feature_model(data)
    logits, _ = model.feature_model.score_edges(data, first)
    changed = replace(
        data,
        conditions=replace(
            data.conditions, collision_energy_eV=torch.tensor([70.0, 90.0])
        ),
    )
    second = model.feature_model(changed)
    other, _ = model.feature_model.score_edges(changed, second)
    torch.testing.assert_close(logits, other)
    out = ReactionFragmentTreeTrainingModel(
        model.feature_model, spectrum_model=model.spectrum_model
    ).eval()(data)
    values = out.spectrum_output.intensity
    n = (out.spectrum_output.sample_index == 0).sum()
    assert values[:n].shape == values[n:].shape and not torch.equal(
        values[:n], values[n:]
    )


def test_collation_and_reordered_conditions_preserve_targets_and_compounds():
    _, a = prepared()
    batch = SourceReactionStructure.from_structures([a, a])
    batch.validate()
    assert len(batch.compounds.compounds) == len(a.compounds.compounds)
    assert (
        batch.source_compounds.compound_index.tolist()
        == a.source_compounds.compound_index.tolist() * 2
    )
    subset = batch.select_samples([3, 0, 3])
    subset.validate()
    assert subset.collision_energy_eV.tolist() == [40.0, 10.0, 40.0]
    assert [s["collisionEnergy"] for s in subset.sample_annotations] == [
        40.0,
        10.0,
        40.0,
    ]
    assert subset.annotations.spectra.peak_ptr.tolist() == [0, 2, 4, 6]
    for i, source_sample in enumerate([1, 0, 1]):
        assert (
            subset.sample_annotations[i]["peaks"]
            == a.sample_annotations[source_sample]["peaks"]
        )


def test_batch_prediction_equals_individual_predictions_and_keeps_input_order():
    model, _ = prepared()
    model.eval()
    sources = [Compound.from_smiles(s) for s in ["CCO", "CC(N)O", "CCO", "CC(N)O"]]
    adducts = [Adduct.parse(s) for s in ["[M+H]+", "[M+H]+", "[M+Na]+", "[M+H]+"]]
    ce = [10.0, 40.0, 20.0, 10.0]
    chunks = list(model.predict_batches(sources, adducts, ce, max_samples=2))
    found = {}
    for output in chunks:
        assert not hasattr(output.spectrum_batch.canonical_data, "targets")
        for internal, original in enumerate(output.sample_input_index):
            mask = output.spectra.sample_index == internal
            found[original] = (
                output.spectra.formula_tensor[mask],
                output.spectra.intensity[mask],
            )
    assert set(found) == set(range(4))
    for i in range(4):
        single = model.predict([sources[i]], [adducts[i]], [ce[i]])
        torch.testing.assert_close(found[i][0], single.spectra.formula_tensor)
        torch.testing.assert_close(
            found[i][1], single.spectra.intensity, rtol=1e-4, atol=1e-6
        )


def test_workbench_shows_group_labels_and_ce_specific_formula_assignments(tmp_path):
    _, data = prepared()
    path = tmp_path / "view.preft.pt"
    data.save(path)
    spec = spec_from_file_location(
        "canonical_workbench",
        "fredis_workbench/src/features/fragment-tree-result/backend.py",
    )
    backend = module_from_spec(spec)
    spec.loader.exec_module(backend)
    output = backend.inspect_structure(path)
    json.dumps(output)
    assert len(output["branchGroups"]) == 1 and len(output["samples"]) == 2
    assert all(
        {"retainedAtomMaps", "discardedAtomMaps", "cutBondMaps", "changedBondMaps"}
        <= row.keys()
        for row in output["reactions"]
    )
    labels = {e["id"]: e["label"] for e in output["branchGroups"][0]["edges"]}
    assert set(labels.values()) <= {-1, 0, 1} and 1 in labels.values()
    assert (
        output["samples"][0]["collisionEnergy"]
        != output["samples"][1]["collisionEnergy"]
    )
    assert (
        output["samples"][0]["peaks"][0]["intensity"]
        != output["samples"][1]["peaks"][0]["intensity"]
    )
    for sample in output["samples"]:
        assert {e["id"]: e["label"] for e in sample["edges"]} == labels
        for peak in sample["peaks"]:
            for match in peak["matches"]:
                assert match["nodeIds"] == [match["compoundIndex"]]
                assert "formulaExplanationIndex" in match


def test_empty_edge_graph_and_all_ignored_branch_labels():
    value = config()
    value["fragmenter_params"]["fragment_ion_tree_builder"]["max_reaction_count"] = 1
    model = create_spectrum_generator(value)
    data, _ = ReactionStructureBuilder(model).build(
        Compound.from_smiles("C"),
        [Adduct.parse("[M+H]+")],
        [20.0],
        [[17.03858]],
        [[1.0]],
    )
    data.validate()
    assert data.reaction_contexts.edge_ptr.tolist() == [0, 0]
    out = ReactionFragmentTreeTrainingModel(
        model.feature_model, spectrum_model=model.spectrum_model
    )(data)
    assert out.branch_loss == 0 and torch.isfinite(out.loss)


def test_merged_compound_self_loop_does_not_repeat_reactions_or_invent_histories():
    from fredis.ml.input import (
        CompoundReactionGraph,
        ReactionSet,
        ReactionContextData,
        SpectrumExplanationData,
    )
    from fredis.ml.input.source_reaction_structure import csr

    _, training = prepared()
    data = training.for_prediction()
    root = int(data.source_compounds.compound_index[0])
    b, c = [i for i in range(len(data.compounds.compounds)) if i != root][:2]
    sets, _ = ReactionSet.from_rows([[], [0], [1], [0, 1]])
    generating = [[] for _ in data.compounds.compounds]
    generating[root] = [0, 1]
    generating[b] = [2]
    generating[c] = [3]
    ptr, ids = csr(generating)
    graph = CompoundReactionGraph(
        torch.tensor([root, root, b]),
        torch.tensor([root, b, c]),
        torch.tensor([0, 1, 0]),
    )
    data = replace(
        data,
        reaction_sets=sets,
        reactions=replace(
            data.reactions,
            reaction_conflict_index=torch.empty((2, 0), dtype=torch.long),
            reaction_invalidation_index=torch.empty((2, 0), dtype=torch.long),
        ),
        compounds=replace(
            data.compounds,
            generating_reaction_sets_ptr=ptr,
            generating_reaction_sets_index=ids,
        ),
        compound_graph=graph,
        reaction_contexts=ReactionContextData(
            torch.tensor([0]), torch.tensor([0, 3]), torch.arange(3)
        ),
        spectrum_explanations=SpectrumExplanationData(
            torch.zeros(3, dtype=torch.long), torch.empty(0, dtype=torch.long)
        ),
    )
    data.validate()
    execution = data.execution_graph
    histories = {
        (int(cid), tuple(sets.rows()[sid]))
        for cid, sid in zip(execution.compound_index, execution.reaction_set_index)
    }
    assert histories == {(root, ()), (root, (0,)), (b, (1,)), (c, (0, 1))}
    for src, dst, reaction in zip(
        execution.source_index, execution.target_index, execution.reaction_index
    ):
        before = sets.rows()[execution.reaction_set_index[src]]
        after = sets.rows()[execution.reaction_set_index[dst]]
        assert int(reaction) not in before and after == sorted(before + [int(reaction)])


def test_intensity_supervision_mask_excludes_unavailable_formula_targets():
    model, data = prepared()
    model.eval()
    from fredis.ml.specgen.fragment_spectrum_model import spectrum_batch_from_input

    target = data.targets
    selected = (target.supervision_mask & (target.intensity > 0)).nonzero().flatten()[0]
    mask = target.supervision_mask.clone()
    mask[selected] = False
    first = replace(data, targets=replace(target, supervision_mask=mask))
    intensities = target.intensity.clone()
    intensities[selected] = 100000
    second = replace(first, targets=replace(first.targets, intensity=intensities))
    trainer = ReactionFragmentTreeTrainingModel(
        model.feature_model, spectrum_model=model.spectrum_model
    ).eval()
    torch.testing.assert_close(trainer(first).loss, trainer(second).loss)
    batch = spectrum_batch_from_input(first, first.targets)
    assert batch.target_mask.dtype == torch.bool and (~batch.target_mask).any()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_training_epoch_checkpoint_reload_and_teacher_free_prediction(
    tmp_path, device, monkeypatch
):
    from fredis.ml.training.fragment_tree_training.training import train_reactions
    from fredis.ml.specgen.predict_spectrum import load_generator
    from .TestReactionPipeline import chemistry

    _, data = prepared()
    directories = [tmp_path / "train", tmp_path / "validation"]
    for directory in directories:
        directory.mkdir()
        data.save(directory / "sample.preft.pt")
        (directory / "manifest.tsv").write_text(
            "file\tnum_valid_samples\tstatus\nsample.preft.pt\t2\tcompleted\n"
        )
        (directory / "reaction_statistics.json").write_text(
            json.dumps({"chemistry_config": chemistry()})
        )
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        def unexpected_cuda(*args, **kwargs):
            raise AssertionError("CPU training invoked a CUDA-only operation")

        for name in (
            "reset_peak_memory_stats",
            "max_memory_allocated",
            "get_rng_state_all",
            "set_rng_state_all",
        ):
            monkeypatch.setattr(torch.cuda, name, unexpected_cuda)
    output = tmp_path / "model"
    train_reactions(
        model_config=config(),
        train_dir=directories[0],
        val_dir=directories[1],
        output_dir=output,
        epochs=1,
        batch_size=1,
        device=device,
        max_samples=2,
        warmup_steps=0,
        validation_interval_steps=1,
        validate_before_training=device == "cpu",
    )
    subset = json.loads((output / "intermediate_validation_subset.json").read_text())
    assert subset["selection_unit"] == "compound"
    assert subset["total_compounds"] == subset["selected_compounds"] == 1
    assert subset["selected_samples"] == 2
    assert subset["sample_indices"] == [0, 1]
    assert subset["compound_smiles"] == ["CC(N)O"]
    checkpoint = output / "last.pt"
    assert checkpoint.exists()
    if device == "cuda":
        assert not (output / "spectrum_validation" / "epoch_0.json").exists()
        assert (
            torch.load(checkpoint, map_location="cpu", weights_only=False)[
                "initial_validation"
            ]
            is None
        )
    if device == "cpu":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["initial_validation"]["epoch"] == 0
        assert saved["initial_validation"]["global_step"] == 0
        assert len(saved["history"]) == 1 and saved["epoch"] == 1
        initial_file = output / "spectrum_validation" / "epoch_0.json"
        initial_mtime = initial_file.stat().st_mtime_ns
        saved_validation = json.loads(initial_file.read_text())
        assert saved_validation["summary"]["samples"] == 2
        assert len(saved_validation["spectra"]) <= 20
        assert saved_validation["summary"]["saved_spectra"] == len(
            saved_validation["spectra"]
        )
        assert set(saved_validation["representatives"]) == {
            "cosine_similarity",
            "cosine_similarity_without_precursor",
            "assignment_score",
            "assignment_score_without_precursor",
        }
        assert not initial_file.with_suffix(".summary.json").exists()
        from fredis.ml.training.fragment_tree_training.validation_storage import (
            read_validation_msdataset,
        )

        preview = read_validation_msdataset(initial_file.with_suffix(".msds"))
        assert preview["summary"]["samples"] == 2
        assert preview["representatives"]["cosine_similarity"]["median"]["spectrum"][
            "generated_peaks"
        ]

        from fredis.ml.training.fragment_tree_training.training import (
            _read_metric_events,
        )

        assert [
            r["epoch"]
            for r in _read_metric_events(output / "metrics.tsv")
            if r["event"] == "epoch_end"
        ] == [0, 1]
        assert saved["cuda_rng_state"] is None
        assert "cuda_peak_memory_mb" not in saved["history"][-1]
        assert all(v.device.type == "cpu" for v in saved["model_state_dict"].values())
        # A CUDA RNG entry in an older checkpoint must be ignored on CPU resume.
        saved["cuda_rng_state"] = [torch.get_rng_state()]
        torch.save(saved, checkpoint)
        train_reactions(
            model_config=config(),
            train_dir=directories[0],
            val_dir=directories[1],
            output_dir=output,
            epochs=1,
            device="cpu",
            resume=str(checkpoint),
            validate_before_training=True,
        )
        resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert resumed["epoch"] == 2 and resumed["cuda_rng_state"] is None
        assert len(resumed["history"]) == 2
        assert initial_file.stat().st_mtime_ns == initial_mtime
        assert [
            r["epoch"]
            for r in _read_metric_events(output / "metrics.tsv")
            if r["event"] == "epoch_end"
        ] == [0, 1, 2]
    generator = load_generator(
        model_path=str(checkpoint),
        params_path=None,
        device=torch.device("cpu"),
        strict=True,
    )
    predicted = generator.predict(
        [data.compounds.compounds[int(data.source_compounds.compound_index[0])]],
        [Adduct.parse("[M+H]+")],
        [20.0],
    )
    assert torch.isfinite(predicted.spectra.intensity).all()
    assert not hasattr(predicted.spectrum_batch.canonical_data, "targets")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer check")
def test_device_transfer_preserves_cpu_prepared_components_and_supports_training():
    model, data = prepared()
    moved = data.to("cuda")
    assert data.source_graph.x.device.type == "cpu"
    assert data.compounds.graph.x.device.type == "cpu"
    assert moved.source_graph.x.device.type == "cuda"
    moved.validate()
    trainer = ReactionFragmentTreeTrainingModel(
        model.feature_model, spectrum_model=model.spectrum_model
    ).cuda()
    result = trainer(moved)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    data.validate()


def test_collation_requires_explicit_conversion_when_mixing_teacher_free_inputs():
    _, data = prepared()
    with pytest.raises(ValueError, match="mix training and prediction"):
        SpectrumPredictionData.from_structures([data, data.for_prediction()])


def test_workbench_single_prediction_backend_uses_shared_compound_ids(tmp_path):
    model, _ = prepared()
    model.eval()
    spec = spec_from_file_location(
        "canonical_prediction_backend",
        "fredis_workbench/src/features/spectrum-prediction/backend.py",
    )
    backend = module_from_spec(spec)
    spec.loader.exec_module(backend)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    with patch.object(backend, "load_generator", return_value=model):
        result = backend.predict(
            dict(
                smiles="CCO",
                adductType="[M+H]+",
                ce="20",
                modelPath=str(checkpoint),
                device="cpu",
            )
        )
    json.dumps(result)
    nodes = {row["id"] for row in result["tree"]["nodes"]}
    assert len(nodes) > 1 and result["tree"]["edges"]
    assert all(
        edge["transitions"] and edge["source"] in nodes and edge["target"] in nodes
        for edge in result["tree"]["edges"]
    )
    assert all(peak["nodeId"] in nodes for peak in result["peaks"])


def test_annotation_export_computes_compound_smiles_once_and_summary_reuses_it():
    from fredis.ml.input.compound_batch import CompoundBatch
    from fredis.ml.data_preparation.fragment_tree.manifest_summary import (
        structure_manifest_fields,
    )

    _, data = prepared()
    original = CompoundBatch.smiles.func
    calls = []

    def counted(batch):
        calls.append(batch)
        return original(batch)

    with patch.object(CompoundBatch, "smiles", property(counted)):
        annotations = data.sample_annotations
    assert len(calls) == 1 and calls[0] is data.compounds
    assert any(peak["matches"] for sample in annotations for peak in sample["peaks"])

    def unexpected(_):
        raise AssertionError("Manifest must reuse the exported annotations")

    with patch.object(
        SourceReactionStructure, "sample_annotations", property(unexpected)
    ):
        summary = structure_manifest_fields(data, annotations)
    assert summary["num_nodes"] == len(data.compounds.compounds)


def test_branch_frontier_storage_preserves_supervised_edges(tmp_path):
    from fredis.ml.input.structure_builder import _ChemistryPreparationBuilder
    from fredis.ml.input.source_reaction_structure import rows

    model = create_spectrum_generator(config())
    raw, _ = _ChemistryPreparationBuilder(model).build(
        Compound.from_smiles("CC(O)N"),
        [Adduct.parse("[M+H]+")],
        [20.0],
        [[18.033826, 44.049476]],
        [[0.5, 1.0]],
    )
    compact = SourceReactionStructure.from_preparation(raw)
    full = SourceReactionStructure.from_preparation(raw, save_all_edges=True)
    compact.validate()
    full.validate()

    def entries(data):
        graph = data.compound_graph
        result = {}
        labels = rows(
            data.reaction_contexts.edge_ptr, data.reaction_contexts.edge_label
        )
        for group, edges in enumerate(
            rows(data.reaction_contexts.edge_ptr, data.reaction_contexts.edge_index)
        ):
            for edge, label in zip(edges, labels[group]):
                key = (
                    group,
                    data.compounds.smiles[int(graph.source_compound_index[edge])],
                    data.compounds.smiles[int(graph.target_compound_index[edge])],
                    int(graph.reaction_index[edge]),
                )
                result[key] = label
        return result

    all_entries, selected = entries(full), entries(compact)
    supported_nodes = {
        (key[0], node)
        for key, label in all_entries.items()
        if label == 1
        for node in key[1:3]
    }
    supported_nodes.update(
        (g, full.source_smiles[int(source)])
        for g, source in enumerate(full.branch_groups.source_compound_index)
    )
    assert supported_nodes
    assert any(label == -1 for label in all_entries.values())
    for key, label in all_entries.items():
        if label != 1:
            assert label == (0 if (key[0], key[1]) in supported_nodes else -1)
    assert selected == {key: label for key, label in all_entries.items() if label >= 0}
    positive_sources = {(k[0], k[1]) for k, label in all_entries.items() if label == 1}
    terminal_nodes = {
        (k[0], k[2]) for k, label in all_entries.items() if label == 1
    } - positive_sources
    terminal_edges = {
        k: label for k, label in all_entries.items() if (k[0], k[1]) in terminal_nodes
    }
    assert terminal_edges, "Fixture must exercise positive terminal continuations"
    assert all(label == 0 for label in terminal_edges.values())
    assert all(k in selected for k in terminal_edges)
    endpoints = set(compact.source_compounds.compound_index.tolist())
    endpoints.update(compact.compound_graph.source_compound_index.tolist())
    endpoints.update(compact.compound_graph.target_compound_index.tolist())
    assert endpoints == set(range(len(compact.compounds.compounds)))
    assert len(compact.compounds.compounds) < len(full.compounds.compounds)
    for data in (compact, full):
        path = tmp_path / ("full.pt" if data is full else "compact.pt")
        data.save(path)
        assert entries(SourceReactionStructure.load(path)) == entries(data)
    full_logits = torch.full((len(all_entries),), 0.3, requires_grad=True)
    compact_logits = torch.full((len(selected),), 0.3, requires_grad=True)
    full_labels = torch.tensor(list(all_entries.values()), dtype=torch.int8)
    compact_labels = torch.tensor(list(selected.values()), dtype=torch.int8)
    full_loss = branch_edge_loss(
        full_logits,
        full_labels,
        source_index=torch.tensor(
            [full.compounds.smiles.index(k[1]) for k in all_entries]
        ),
        branch_group_index=torch.tensor([k[0] for k in all_entries]),
    )[0]
    compact_loss = branch_edge_loss(
        compact_logits,
        compact_labels,
        source_index=torch.tensor(
            [compact.compounds.smiles.index(k[1]) for k in selected]
        ),
        branch_group_index=torch.tensor([k[0] for k in selected]),
    )[0]
    torch.testing.assert_close(full_loss, compact_loss)
    full_loss.backward()
    assert (full_logits.grad[full_labels == -1] == 0).all()


def test_branch_node_balancing_and_sum():
    def loss(values, labels, sources, groups, weight=0.5):
        z = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        output = branch_edge_loss(
            z,
            torch.tensor(labels, dtype=torch.int8),
            weight,
            source_index=torch.tensor(sources, dtype=torch.long),
            branch_group_index=torch.tensor(groups, dtype=torch.long),
        )
        return z, output

    z, (actual, positive, negative) = loss(
        [0.2, -0.4] + [0.7] * 10, [1, 1] + [0] * 10, [0] * 12, [0] * 12
    )
    pl = torch.nn.functional.softplus(-z[:2]).mean()
    nl = torch.nn.functional.softplus(z[2:]).mean()
    torch.testing.assert_close(actual, (pl + 0.5 * nl) / 1.5)
    torch.testing.assert_close(positive, pl)
    torch.testing.assert_close(negative, nl)
    # Replicating either class leaves this node's loss unchanged.
    _, repeated = loss(
        [0.2, -0.4] * 3 + [0.7] * 20, [1] * 6 + [0] * 20, [0] * 26, [0] * 26
    )
    torch.testing.assert_close(actual, repeated[0])
    # A second group with the same Compound ID is an independent source node.
    _, doubled = loss(
        ([0.2, -0.4] + [0.7] * 10) * 2,
        ([1, 1] + [0] * 10) * 2,
        [0] * 24,
        [0] * 12 + [1] * 12,
    )
    torch.testing.assert_close(doubled[0], 2 * actual)
    # Positive-only and negative-only terminal nodes use their class mean.
    z, output = loss([0.2, 0.7, 4.0], [1, 0, -1], [0, 1, 0], [0, 0, 0])
    torch.testing.assert_close(
        output[0],
        torch.nn.functional.softplus(-z[0]) + torch.nn.functional.softplus(z[1]),
    )
    output[0].backward()
    assert z.grad[1] > 0 and z.grad[2] == 0
    # Zero negative weight and empty batches remain differentiable.
    z, output = loss([0.2, 0.7], [1, 0], [0, 0], [0, 0], weight=0)
    torch.testing.assert_close(output[0], torch.nn.functional.softplus(-z[0]))
    output[0].backward()
    assert z.grad[1] == 0
    z, terminal = loss([0.7, -0.2], [0, 0], [0, 0], [0, 0])
    torch.testing.assert_close(terminal[0], torch.nn.functional.softplus(z).mean())
    terminal[0].backward()
    torch.testing.assert_close(z.grad, z.detach().sigmoid() / 2)
    _, replicated_terminal = loss([0.7, -0.2] * 5, [0] * 10, [0] * 10, [0] * 10)
    torch.testing.assert_close(terminal[0], replicated_terminal[0])
    # Terminal-only sources also honor zero negative weight.
    z, output = loss([0.7, -0.2], [0, 0], [0, 0], [0, 0], weight=0)
    output[0].backward()
    assert output[0] == 0 and (z.grad == 0).all()
    z, output = loss([], [], [], [])
    output[0].backward()
    assert output[0] == 0 and z.grad.numel() == 0


def test_prepared_compound_smiles_reused_on_transfer_and_reload(tmp_path):
    _, data = prepared()
    expected = data.compounds.smiles
    # Prepared identity metadata survives both device copies and serialization.
    with patch.object(
        Compound,
        "smiles",
        new_callable=PropertyMock,
        side_effect=AssertionError("Unexpected RDKit SMILES regeneration"),
    ):
        copied = data.to("cpu")
        assert copied.compounds.smiles == expected
        file = tmp_path / "prepared.preft.pt"
        copied.save(file)
        restored = SourceReactionStructure.load(file)
        assert restored.compounds.smiles == expected
