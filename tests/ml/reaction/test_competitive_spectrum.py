import pytest
import torch
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.competitive_spectrum_model import propagate_probability
from fredis.ml.common.torch_utils.segment_ops import segment_softmax
from fredis.ml.specgen.components.condition.collision_energy_feature import (
    CollisionEnergyFeatureLayer,
)
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.input.reaction_batching import select_samples
from .TestReactionPipeline import config, chemistry


def competitive_config():
    value = config()
    value["spectrum_model_params"].update(
        ce_fc_dims=[12, 8],
        collision_energy_dim=8,
        ce_scale=100.0,
        ce_center=0.0,
        fragment_fc_dims=[16, 8],
        state_fc_dims=[16, 8],
    )
    return value


def test_flow_competes_and_adds_diamond_paths():
    # Scrambled IDs: 3 -> 2/0 -> 1, including terminal Self and sink.
    source = torch.tensor([3, 3, 3, 2, 2, 0, 0, 1, 1])
    target = torch.tensor([3, 2, 0, 2, 1, 0, 1, 1, -1])
    logits = torch.zeros(9, requires_grad=True)
    probs = segment_softmax(logits, source, 4)
    initial = torch.tensor([0.0, 0.0, 0.0, 1.0])
    final, sink = propagate_probability(initial, source, target, probs)
    torch.testing.assert_close(final, torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 3]))
    torch.testing.assert_close(sink, torch.tensor([0.0, 1 / 6, 0.0, 0.0]))
    torch.testing.assert_close(final.sum() + sink.sum(), initial.sum())
    (-final[1].log()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_ce_is_normalized_scalar_mlp_and_roundtrips():
    layer = CollisionEnergyFeatureLayer(3, fc_dims=(), center=10.0, scale=100.0)
    with torch.no_grad():
        layer.proj[0].weight.fill_(1)
        layer.proj[0].bias.zero_()
    torch.testing.assert_close(
        layer(torch.tensor([20.0, 21.0])), torch.tensor([[0.1] * 3, [0.11] * 3])
    )
    assert "frequencies" not in layer.state_dict()
    with pytest.raises(ValueError):
        CollisionEnergyFeatureLayer(8, scale=0.0)


def test_competitive_training_prediction_and_checkpoint():
    torch.manual_seed(9)
    params = competitive_config()
    generator = create_spectrum_generator(params).eval()
    source = Compound.from_smiles("CC(O)N")
    adduct = Adduct.parse("[M+H]+")
    data, _ = ReactionStructureBuilder(generator).build(
        source,
        [adduct, adduct],
        [20.0, 40.0],
        [[18.033826, 44.049476]] * 2,
        [[0.5, 1.0]] * 2,
    )
    assert data.spectrum_batch.decoded.reaction_state_graph
    features = generator.feature_model(data, training_pool=True)
    output = generator.spectrum_model(
        data.spectrum_batch,
        reaction_h=features.reaction_h,
        collision_energy_h=features.collision_energy_h,
        source_embeddings=features.source_embeddings,
    )
    assert (
        torch.isfinite(output.loss) and output.metrics["probability_mass_error"] < 1e-5
    )
    output.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in generator.spectrum_model.state_heads.parameters()
    )
    retained = output.ion_probability.new_zeros(2).scatter_add_(
        0, output.sample_index, output.ion_probability
    )
    torch.testing.assert_close(retained + output.sink_probability, torch.ones(2))
    sink_events = output.transition_target_index == -1
    assert int(sink_events.sum()) == output.fragment_probability.numel()
    has_child = torch.zeros_like(output.fragment_probability, dtype=torch.bool)
    has_child[output.transition_source_index[output.transition_reaction_index >= 0]] = (
        True
    )
    leaves = (~has_child).nonzero().flatten()
    self_events = output.transition_source_index == output.transition_target_index
    self_prob = output.transition_probability[self_events]
    sink_prob = output.transition_probability[sink_events]
    assert leaves.numel() > 0
    assert (self_prob > 0).all() and (self_prob < 1).all()
    torch.testing.assert_close(
        self_prob[leaves] + sink_prob[leaves], torch.ones(leaves.numel())
    )
    assert ((self_prob + sink_prob)[has_child] < 1).all()
    state_totals = output.state_probability.new_zeros(
        output.fragment_probability.numel()
    ).scatter_add_(0, output.state_fragment_index, output.state_probability)
    torch.testing.assert_close(state_totals, torch.ones_like(state_totals))
    torch.testing.assert_close(
        output.ion_probability,
        output.fragment_probability[output.state_fragment_index]
        * output.state_probability,
    )
    fragment_totals = torch.zeros_like(output.fragment_probability).scatter_add_(
        0, output.state_fragment_index, output.ion_probability
    )
    torch.testing.assert_close(fragment_totals, output.fragment_probability)
    assert (
        next(iter(generator.spectrum_model.fragment_heads.values()))
        .layers[0]
        .weight.grad.abs()
        .sum()
        > 0
    )
    assert (
        generator.feature_model.collision_energy_encoder.proj[0].weight.grad.abs().sum()
        > 0
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in generator.feature_model.reaction_instance_encoder.parameters()
    )
    restored = create_spectrum_generator(params).eval()
    restored.load_state_dict(generator.state_dict(), strict=True)
    with torch.no_grad():
        pred = generator.predict([source], [adduct], [20.0])
        again = restored.predict([source], [adduct], [20.0])
    torch.testing.assert_close(pred.spectra.intensity, again.spectra.intensity)
    assert pred.spectra.metrics["probability_mass_error"] < 1e-5
    from fredis.ml.specgen.competitive_spectrum_model import merge_candidate_peaks

    merged = merge_candidate_peaks(pred.spectra)
    valid = (pred.spectra.formula_tensor[:, :-1] >= 0).all(1) & (pred.spectra.mz > 0)
    torch.testing.assert_close(
        merged.intensity.sum(), pred.spectra.intensity[valid].sum()
    )
    torch.testing.assert_close(merged.intensity.max(), torch.tensor(1.0))
    torch.testing.assert_close(
        merged.ion_probability.sum(), pred.spectra.ion_probability[valid].sum()
    )
    probabilities = pred.spectra.transition_probability
    sources = pred.spectra.transition_source_index
    totals = probabilities.new_zeros(
        pred.spectra.fragment_probability.numel()
    ).scatter_add_(0, sources, probabilities)
    torch.testing.assert_close(totals, torch.ones_like(totals))

    assert not any("tree_encoder" in k for k in generator.spectrum_model.state_dict())
    subset = select_samples(data, [1])
    assert subset.spectrum_batch.precursor_adducts == ("[M+H]+",)
    from fredis.ml.input.source_reaction_structure import SourceReactionStructure

    batched = SourceReactionStructure.from_structures([subset, subset])
    f = generator.feature_model(batched, training_pool=True)
    b = generator.spectrum_model(
        batched.spectrum_batch,
        reaction_h=f.reaction_h,
        collision_energy_h=f.collision_energy_h,
        source_embeddings=f.source_embeddings,
    )
    assert b.metrics["probability_mass_error"] < 1e-5
    assert len(batched.spectrum_batch.precursor_adducts) == 2


def test_preparation_context_keeps_state_graph():
    from fredis.ml.data_preparation.fragment_tree.context import (
        create_preparation_context,
    )

    data, _ = ReactionStructureBuilder(create_preparation_context(chemistry())).build(
        Compound.from_smiles("CC(O)N"),
        [Adduct.parse("[M+H]+")],
        [20.0],
        [[44.049476]],
        [[1.0]],
    )
    assert data.spectrum_batch.decoded.reaction_state_graph


def test_mlp_cli_forwarding_and_configuration():
    from fredis.ml.training.fragment_tree_training.training import (
        build_arg_parser,
        training_model_config,
    )
    from fredis.cli.base import namespace_argv

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--dataset-dir",
            "dataset",
            "--output-dir",
            "out",
            "--ce-fc-dims",
            "24,12",
            "--ce-scale",
            "50",
            "--collision-energy-feature-dim",
            "7",
            "--fragment-fc-dims",
            "48,24",
            "--state-fc-dims",
            "32,16",
            "--dropout",
            "0.2",
        ]
    )
    forwarded = parser.parse_args(namespace_argv(parser, args))
    cfg = training_model_config(forwarded)["spectrum_model_params"]
    assert (
        cfg["ce_fc_dims"] == [24, 12]
        and cfg["collision_energy_dim"] == 7
        and cfg["ce_scale"] == 50
    )
    assert cfg["fragment_fc_dims"] == [48, 24]
    assert cfg["state_fc_dims"] == [32, 16] and training_model_config(forwarded)["dropout"] == 0.2
    assert "propagation_steps" not in cfg
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset-dir",
                "dataset",
                "--output-dir",
                "o",
                "--propagation-steps",
                "6",
            ]
        )
    assert (
        parser.parse_args(
            namespace_argv(
                parser,
                parser.parse_args(
                    [
                        "--dataset-dir",
                        "dataset",
                        "--output-dir",
                        "o",
                        "--fragment-fc-dims",
                        "",
                    ]
                ),
            )
        ).fragment_fc_dims
        == []
    )


def test_all_parents_survive_reaction_set_deduplication():
    from fredis.ml.specgen.components.reaction.reaction_decoder import (
        ReactionPool,
        ReactionDecoderOutput,
        with_all_transitions,
    )
    from fredis.ml.specgen.components.reaction.reaction_compatibility import (
        ReactionCompatibilityEngine,
    )

    masks = torch.zeros((1, 2, 2), dtype=torch.bool)
    pool = ReactionPool(
        torch.tensor([[0, 1]]),
        torch.ones((1, 2), dtype=torch.bool),
        torch.randn(1, 2, 4),
        masks,
        masks,
        torch.ones((1, 2, 1), dtype=torch.bool),
        torch.ones((1, 1), dtype=torch.bool),
    )
    decoded = ReactionDecoderOutput(
        torch.zeros(4, dtype=torch.long),
        torch.tensor([[-1, -1], [0, -1], [1, -1], [0, 1]]),
        torch.zeros(4),
        torch.tensor([False, False, False, True]),
        torch.tensor([-1, 0, 0, 1]),
        torch.tensor([-1, 0, 1, 1]),
        pool,
    )
    result = with_all_transitions(decoded, ReactionCompatibilityEngine(2))
    assert set(
        result.transition_parent_index[result.transition_child_index == 3].tolist()
    ) == {1, 2}


def test_neutral_loss_precursor_initialization():
    generator = create_spectrum_generator(competitive_config()).eval()
    output = generator.predict(
        [Compound.from_smiles("CCO")], [Adduct.parse("[M+H-H2O]+")], [20.0]
    )
    assert output.spectra.metrics["probability_mass_error"] < 1e-5


def test_configurable_adduct_embedding_and_legacy_ce():
    params = competitive_config()
    params["spectrum_model_params"]["adduct_embedding_dim"] = 7
    generator = create_spectrum_generator(params)
    features = generator.feature_model
    assert features.adduct_encoder.embedding.embedding_dim == 7
    assert features.main_adduct_projection[0].in_features == 7
    ce = features.collision_energy_encoder
    assert not any(isinstance(module, torch.nn.Dropout) for module in ce.modules())
    energy = torch.tensor([10.0, 20.0])
    torch.testing.assert_close(ce(energy), ce(energy), rtol=0, atol=0)
    for width in (0, -1, True):
        params["spectrum_model_params"]["adduct_embedding_dim"] = width
        with pytest.raises(ValueError, match="adduct_embedding_dim"):
            create_spectrum_generator(params)


def test_main_adduct_condition_banks_have_independent_weights():
    params = config()
    rules = params["fragmenter_params"]["fragment_ion_tree_builder"][
        "fragment_ion_adduct_rule_set"
    ]
    rules["hydrogen_shift_min"], rules["hydrogen_shift_max"] = -5, 0
    sodium = next(
        rule for rule in rules["adduct_rules"] if rule["adduct_type"] == "[M+Na]+"
    )
    sodium["fragment_ion_adducts"] = ["[M+Na]+", "[M+H]+"]
    generator = create_spectrum_generator(params).eval()
    model = generator.spectrum_model
    na = model.state_heads[model.head_index["[M+Na]+"]]
    proton = model.state_heads[model.head_index["[M+H]+"]]
    assert na.layers[0].weight.shape == (12, 16, 48)
    assert na.layers[0].bias.shape == (12, 16)
    assert na.layers[-1].weight.shape == (12, 1, 16)
    assert isinstance(na.activation, torch.nn.GELU)
    x = torch.randn(2, 48)
    ids = torch.tensor([0, 1])
    before = proton(x, ids).detach().clone()
    with torch.no_grad():
        na.layers[-1].bias[0].add_(5)
    torch.testing.assert_close(proton(x, ids), before, rtol=0, atol=0)
    loss = na(x, ids).sum()
    loss.backward()
    assert na.layers[0].weight.grad[:2].abs().sum() > 0
    assert na.layers[0].weight.grad[2:].abs().sum() == 0
    assert proton.layers[0].weight.grad is None


def test_formula_and_mz_do_not_control_transitions_or_probability_flow():
    from dataclasses import replace

    generator = create_spectrum_generator(config()).eval()
    source, adduct = Compound.from_smiles("CC(O)N"), Adduct.parse("[M+H]+")
    data, _ = ReactionStructureBuilder(generator).build(
        source, [adduct], [20.0], [[44.049476]], [[1.0]]
    )
    features = generator.feature_model(data)
    kwargs = dict(
        reaction_h=features.reaction_h,
        collision_energy_h=features.collision_energy_h,
        source_embeddings=features.source_embeddings,
    )
    original = generator.spectrum_model(data.spectrum_batch, **kwargs)
    modified = replace(
        data.spectrum_batch,
        state_formula_tensor=data.spectrum_batch.state_formula_tensor + 100,
        state_mz=data.spectrum_batch.state_mz + 1000,
    )
    changed = generator.spectrum_model(modified, **kwargs)
    for field in (
        "transition_source_index",
        "transition_target_index",
        "transition_probability",
        "ion_probability",
    ):
        torch.testing.assert_close(
            getattr(original, field), getattr(changed, field), rtol=0, atol=0
        )


def test_inference_prepares_shared_full_condition_grid():
    generator = create_spectrum_generator(config()).eval()
    model = generator.spectrum_model
    result = generator.predict(
        [Compound.from_smiles("CCO")], [Adduct.parse("[M+H]+")], [20.0]
    )
    data = result.spectrum_batch
    per_node = len(model.main_adduct_conditions["[M+H]+"])
    assert data.state_node_index.numel() == len(result.fragments.compounds) * per_node
    index = 0
    node = int(data.state_node_index[index])
    ion = Adduct.parse(data.ion_type_strs[int(data.state_adduct_index[index])])
    shift = Adduct.from_dict({"H": int(data.state_hydrogen_shift[index])})
    expected = ion.apply_to_formula(
        shift.apply_to_formula(result.fragments.compounds[node].formula)
    ).normalized
    torch.testing.assert_close(
        result.spectra.formula_tensor[index],
        generator.tensorizer.formula_to_tensor(expected),
    )
    assert float(result.spectra.mz[index]) == pytest.approx(
        expected.exact_mass / abs(ion.charge)
    )


def test_mixed_main_adduct_batch_routes_to_separate_banks():
    params = config()
    rules = params["fragmenter_params"]["fragment_ion_tree_builder"][
        "fragment_ion_adduct_rule_set"
    ]
    sodium = next(
        rule for rule in rules["adduct_rules"] if rule["adduct_type"] == "[M+Na]+"
    )
    sodium["fragment_ion_adducts"] = ["[M+Na]+", "[M+H]+"]
    generator = create_spectrum_generator(params).eval()
    called = []
    hooks = [
        generator.spectrum_model.fragment_heads[
            generator.spectrum_model.head_index[main]
        ].register_forward_hook(
            lambda module, args, output, main=main: called.append(main)
        )
        for main in ("[M+H]+", "[M+Na]+")
    ]
    source = Compound.from_smiles("CCO")
    result = generator.predict(
        [source, source],
        [Adduct.parse("[M+H]+"), Adduct.parse("[M+Na]+")],
        [20.0, 40.0],
    )
    for hook in hooks:
        hook.remove()
    assert set(called) == {"[M+H]+", "[M+Na]+"}
    assert len(result.fragments.trees) == 2
    probabilities = result.spectra.ion_probability
    totals = probabilities.new_zeros(2).scatter_add_(
        0, result.spectra.sample_index, probabilities
    )
    torch.testing.assert_close(totals + result.spectra.sink_probability, torch.ones(2))
    invalid = (result.spectra.formula_tensor[:, :-1] < 0).any(1)
    assert (
        invalid.any()
    ), "The neural grid must not be pruned by output formula validity"
    from fredis.ml.specgen.competitive_spectrum_model import merge_candidate_peaks

    merged = merge_candidate_peaks(result.spectra)
    assert (merged.formula_tensor[:, :-1] >= 0).all()
    assert (merged.mz > 0).all()


@pytest.mark.parametrize("adduct_type", ["[M+2H]2+", "[M-2H]2-"])
def test_annotation_mz_uses_adduct_charge(adduct_type):
    from types import SimpleNamespace
    from fredis.ml.specgen.fragment_spectrum_model import (
        SpectrumModelBatch,
        annotate_ion_conditions,
    )

    generator = create_spectrum_generator(config())
    compound = Compound.from_smiles("CCO")
    data = SpectrumModelBatch(
        decoded=SimpleNamespace(compounds=[compound]),
        state_node_index=torch.tensor([0]),
        state_sample_index=torch.tensor([0]),
        state_formula_tensor=torch.empty(0),
        state_mz=torch.empty(0),
        state_adduct_index=torch.tensor([0]),
        state_hydrogen_shift=torch.tensor([0]),
        ion_type_strs=(adduct_type,),
    )
    annotated = annotate_ion_conditions(data, generator.tensorizer)
    ion = Adduct.parse(adduct_type)
    formula = ion.apply_to_formula(compound.formula).normalized
    assert annotated.state_mz.item() == pytest.approx(
        formula.exact_mass / abs(ion.charge)
    )


@pytest.mark.parametrize("precursor_adduct", ["[M+H]+", "[M+H-H2O]+"])
@pytest.mark.parametrize("exponent", [0.5, 1.0])
def test_spectrum_losses_include_and_exclude_declared_precursor(precursor_adduct, exponent):
    from types import SimpleNamespace
    from fredis.ml.specgen.fragment_spectrum_model import SpectrumLossSettings

    model = create_spectrum_generator(competitive_config()).spectrum_model
    adduct = str(Adduct.parse(precursor_adduct))
    data = SimpleNamespace(
        decoded=SimpleNamespace(node_sample_index=torch.tensor([0, 0, 0, 0])),
        state_node_index=torch.tensor([0, 1, 2, 3]),
        state_mz=torch.tensor([100.0, 50.0, 40.0, 100.0]),
        ion_type_strs=(adduct,),
        state_adduct_index=torch.zeros(4, dtype=torch.long),
        precursor_adducts=(adduct,),
        target_intensity=torch.tensor([100.0, 1.0, 1.0, 100.0]),
    )
    # The last candidate has the precursor formula at a different structural node.
    formulas = torch.tensor([[10.0, 1.0], [5.0, 1.0], [4.0, 1.0], [10.0, 1.0]])
    probability = torch.tensor([0.45, 0.09, 0.01, 0.45], requires_grad=True)
    sample = torch.zeros(4, dtype=torch.long)
    _, loss, metrics = model._spectrum_loss(
        data,
        probability,
        sample,
        formulas,
        1,
        SpectrumLossSettings(cosine_loss_weight=0, intensity_exponent=exponent),
    )
    full_expected = (
        -(
            100 ** exponent * torch.log(torch.tensor(0.9))
            + torch.log(torch.tensor(0.09))
            + torch.log(torch.tensor(0.01))
        )
        / (100 ** exponent + 2)
    )
    fragment_expected = (
        -(torch.log(torch.tensor(0.9)) + torch.log(torch.tensor(0.1))) / 2
    )
    torch.testing.assert_close(metrics["spectrum_loss_with_precursor"], full_expected)
    torch.testing.assert_close(
        metrics["spectrum_loss_without_precursor"], fragment_expected
    )
    torch.testing.assert_close(loss, full_expected + fragment_expected)
    (gradient,) = torch.autograd.grad(
        metrics["spectrum_loss_without_precursor"], probability
    )
    assert gradient[0] == 0 and gradient[3] == 0
    assert gradient[1] > 0 and gradient[2] < 0
    # Precursor amplitude cannot change the conditional fragment objective.
    changed = torch.tensor([0.05, 0.81, 0.09, 0.05])
    _, _, changed_metrics = model._spectrum_loss(
        data, changed, sample, formulas, 1, SpectrumLossSettings(cosine_loss_weight=0, intensity_exponent=exponent)
    )
    torch.testing.assert_close(
        changed_metrics["spectrum_loss_without_precursor"], fragment_expected
    )
    _, _, cosine_metrics = model._spectrum_loss(
        data, probability, sample, formulas, 1,
        SpectrumLossSettings(intensity_exponent=exponent),
    )
    for suffix in ("", "_with_precursor", "_without_precursor"):
        torch.testing.assert_close(
            cosine_metrics["spectrum_loss" + suffix],
            cosine_metrics["cross_entropy_loss" + suffix]
            + cosine_metrics["cosine_loss" + suffix],
        )
    for name in ("spectrum_loss", "cross_entropy_loss", "cosine_loss"):
        torch.testing.assert_close(
            cosine_metrics[name],
            cosine_metrics[name + "_with_precursor"]
            + cosine_metrics[name + "_without_precursor"],
        )
    pred = torch.tensor([0.9, 0.09, 0.01]).pow(exponent)
    truth = torch.tensor([100.0, 1.0, 1.0]).pow(exponent)
    expected_cosine = torch.nn.functional.cosine_similarity(pred, truth, dim=0)
    torch.testing.assert_close(cosine_metrics["full_spectrum_cosine"], expected_cosine)
    data.target_intensity = torch.tensor([100.0, 0.0, 0.0, 100.0])
    _, only_precursor_loss, metrics = model._spectrum_loss(
        data, probability, sample, formulas, 1, SpectrumLossSettings()
    )
    assert metrics["spectrum_loss_without_precursor"] == 0
    only_precursor_loss.backward()
    assert torch.isfinite(probability.grad).all()


@pytest.mark.parametrize("precursor_adduct", ["[M+H]+", "[M+H-H2O]+"])
def test_precursor_loss_excludes_hydrogen_shifts_but_keeps_other_ions(precursor_adduct):
    from types import SimpleNamespace
    from fredis.ml.specgen.fragment_spectrum_model import SpectrumLossSettings

    model = create_spectrum_generator(competitive_config()).spectrum_model
    precursor = Adduct.parse(precursor_adduct)
    plus = str(precursor.add_prefer_self(Adduct.from_dict({"H": 1})))
    minus = str(precursor.add_prefer_self(Adduct.from_dict({"H": -1})))
    # Na addition and water loss must not count as hydrogen-only changes.
    sodium = str(precursor.add_prefer_self(Adduct.from_dict({"Na": 1})))
    water_loss = str(precursor.add_prefer_self(Adduct.parse("[M-H2O]")))
    data = SimpleNamespace(
        decoded=SimpleNamespace(node_sample_index=torch.zeros(4, dtype=torch.long)),
        state_node_index=torch.tensor([0, 0, 0, 1, 2, 3, 0, 0]),
        state_mz=torch.ones(8),
        ion_type_strs=(
            str(precursor),
            plus,
            minus,
            plus,
            minus,
            plus,
            sodium,
            water_loss,
        ),
        state_adduct_index=torch.arange(8),
        precursor_adducts=(str(precursor),),
        target_intensity=torch.tensor([100.0, 50.0, 50.0, 1.0, 1.0, 50.0, 1.0, 1.0]),
    )
    formulas = torch.tensor(
        [
            [20.0, 1.0],
            [21.0, 1.0],
            [19.0, 1.0],
            [5.0, 1.0],
            [4.0, 1.0],
            [21.0, 1.0],
            [30.0, 1.0],
            [18.0, 1.0],
        ]
    )
    probability = torch.tensor(
        [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2], requires_grad=True
    )
    _, loss, metrics = model._spectrum_loss(
        data,
        probability,
        torch.zeros(8, dtype=torch.long),
        formulas,
        1,
        SpectrumLossSettings(cosine_loss_weight=0, intensity_exponent=1),
    )
    expected = -torch.tensor([0.2, 0.2, 0.2, 0.4]).log().mean()
    torch.testing.assert_close(metrics["spectrum_loss_without_precursor"], expected)
    (gradient,) = torch.autograd.grad(
        metrics["spectrum_loss_without_precursor"], probability
    )
    assert (gradient[[0, 1, 2, 5]] == 0).all()
    assert (gradient[[3, 4, 6, 7]] != 0).all()
    assert torch.isfinite(loss)


def test_terminal_self_and_sink_are_learnable_and_conserve_initial_mass():
    logits = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    final, sink = propagate_probability(
        torch.tensor([0.7], dtype=torch.float64),
        torch.tensor([0, 0]),
        torch.tensor([0, -1]),
        logits.softmax(0),
    )
    torch.testing.assert_close(final, torch.tensor([0.35], dtype=torch.float64))
    torch.testing.assert_close(sink, torch.tensor([0.35], dtype=torch.float64))
    (-final.log().sum()).backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0


def test_dag_flow_reaches_beyond_six_breaks_without_repropagating_self():
    count = 10
    source = torch.tensor([i for i in range(count) for _ in range(2)])
    target = torch.tensor(
        [v for i in range(count) for v in (i, i + 1 if i + 1 < count else -1)]
    )
    probability = torch.tensor(
        [0.25, 0.75] * (count - 1) + [0.5, 0.5], requires_grad=True
    )
    initial = torch.zeros(count)
    initial[0] = 1
    final, sink = propagate_probability(initial, source, target, probability)
    expected = torch.tensor(
        [0.25 * 0.75**i for i in range(count - 1)] + [0.5 * 0.75 ** (count - 1)]
    )
    torch.testing.assert_close(final, expected)
    assert final[-1] > 0
    torch.testing.assert_close(final.sum() + sink.sum(), torch.tensor(1.0))
    final[-1].backward()
    assert torch.isfinite(probability.grad).all()
    assert (probability.grad[1 : 2 * (count - 1) : 2] > 0).all()


def test_probability_flow_rejects_break_cycles():
    with pytest.raises(ValueError, match="cycles"):
        propagate_probability(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            torch.ones(2),
        )


def test_disconnected_samples_and_multiple_initial_states_conserve_mass():
    # Two independent DAGs; the first has mass initially at both its root and child.
    initial = torch.tensor([0.6, 0.4, 1.0, 0.0])
    source = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    target = torch.tensor([0, 1, 1, -1, 2, 3, 3, -1])
    final, sink = propagate_probability(initial, source, target, torch.full((8,), 0.5))
    sample = torch.tensor([0, 0, 1, 1])
    total = torch.zeros(2).scatter_add_(0, sample, final + sink)
    torch.testing.assert_close(total, torch.ones(2))
    torch.testing.assert_close(final, torch.tensor([0.3, 0.35, 0.5, 0.25]))
    torch.testing.assert_close(sink, torch.tensor([0.0, 0.35, 0.0, 0.25]))


def test_state_head_is_shared_normalized_and_cannot_change_fragment_flow():
    generator = create_spectrum_generator(competitive_config()).eval()
    model = generator.spectrum_model
    with torch.no_grad():
        for head in (*model.fragment_heads.values(), *model.state_heads.values()):
            for parameter in head.parameters():
                parameter.zero_()
    source, adduct = Compound.from_smiles("CCO"), Adduct.parse("[M+H]+")
    before = generator.predict([source], [adduct], [20.0]).spectra
    condition_count = len(model.main_adduct_conditions["[M+H]+"])
    torch.testing.assert_close(
        before.state_probability,
        torch.full_like(before.state_probability, 1 / condition_count),
    )
    event_count = torch.bincount(before.transition_source_index)
    torch.testing.assert_close(
        before.transition_probability,
        1 / event_count[before.transition_source_index].float(),
    )
    with torch.no_grad():
        model.state_heads[model.head_index["[M+H]+"]].layers[-1].bias[0].add_(2)
    after = generator.predict([source], [adduct], [20.0]).spectra
    torch.testing.assert_close(
        before.fragment_probability, after.fragment_probability, rtol=0, atol=0
    )
    torch.testing.assert_close(
        before.transition_probability, after.transition_probability, rtol=0, atol=0
    )
    torch.testing.assert_close(
        before.sink_probability, after.sink_probability, rtol=0, atol=0
    )
    assert not torch.equal(before.state_probability, after.state_probability)
    assert not torch.equal(before.ion_probability, after.ion_probability)
    totals = torch.zeros_like(after.fragment_probability).scatter_add_(
        0, after.state_fragment_index, after.state_probability
    )
    torch.testing.assert_close(totals, torch.ones_like(totals))


def test_fragment_flow_has_no_autograd_dependency_on_state_head():
    generator = create_spectrum_generator(competitive_config()).eval()
    data, _ = ReactionStructureBuilder(generator).build(
        Compound.from_smiles("CC(O)N"),
        [Adduct.parse("[M+H]+")],
        [20.0],
        [[18.033826, 44.049476]],
        [[0.5, 1.0]],
    )
    features = generator.feature_model(data, training_pool=True)
    output = generator.spectrum_model(
        data.spectrum_batch,
        reaction_h=features.reaction_h,
        collision_energy_h=features.collision_energy_h,
        source_embeddings=features.source_embeddings,
    )
    state_parameters = tuple(generator.spectrum_model.state_heads.parameters())
    gradients = torch.autograd.grad(
        output.fragment_probability.sum(),
        state_parameters,
        allow_unused=True,
        retain_graph=True,
    )
    assert all(gradient is None for gradient in gradients)
    output.loss.backward()
    for heads in (
        generator.spectrum_model.fragment_heads,
        generator.spectrum_model.state_heads,
    ):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in heads.parameters()
        )
    # Sink is a separate learned event even for nonterminal fragments.
    head = generator.spectrum_model.fragment_heads[
        generator.spectrum_model.head_index["[M+H]+"]
    ]
    assert head.layers[-1].bias.grad[2].abs().sum() > 0


def test_fragment_diamond_then_state_softmax_has_known_joint_probabilities():
    from types import SimpleNamespace
    from fredis.ml.specgen.competitive_spectrum_model import CompetitiveSpectrumModel
    from fredis.ml.mol.formula_encoder import FormulaTensorizer

    encoder = torch.nn.Identity()
    encoder.graph_dim = 2
    model = CompetitiveSpectrumModel(
        encoder,
        1,
        main_adduct_conditions={"[M+H]+": (("[M+H]+", -1), ("[M+H]+", 0))},
        precursor_main_adducts={"[M+H]+": "[M+H]+"},
        formula_tensorizer=FormulaTensorizer(("C",)),
        collision_energy_dim=1,
        fragment_fc_dims=(),
        state_fc_dims=(),
    )
    with torch.no_grad():
        for heads in (model.fragment_heads, model.state_heads):
            for parameter in heads.parameters():
                parameter.zero_()
    data = SimpleNamespace(
        decoded=SimpleNamespace(
            reaction_state_graph=True,
            node_sample_index=torch.zeros(4, dtype=torch.long),
            edge_index=torch.tensor([[2, 0, 1, 0], [3, 2, 3, 1]]),
            edge_added_reaction_index=torch.zeros(4, dtype=torch.long),
        ),
        state_node_index=torch.arange(4).repeat_interleave(2),
        state_sample_index=torch.zeros(8, dtype=torch.long),
        state_adduct_index=torch.zeros(8, dtype=torch.long),
        state_hydrogen_shift=torch.tensor([-1, 0] * 4),
        state_formula_tensor=torch.tensor([[1.0, 1.0]] * 8),
        state_mz=torch.ones(8),
        ion_type_strs=("[M+H]+",),
        precursor_adducts=("[M+H]+",),
        target_intensity=torch.zeros(8),
    )
    output = model(
        data,
        reaction_h=torch.ones(1, 1),
        collision_energy_h=torch.ones(1, 1),
        molecular_embeddings=torch.ones(4, 2),
    )
    # Root has four events; middle nodes have three; leaf has Self and Sink.
    expected = torch.tensor([1 / 4, 1 / 12, 1 / 12, 1 / 12])
    torch.testing.assert_close(output.fragment_probability, expected)
    torch.testing.assert_close(output.state_probability, torch.full((8,), 0.5))
    torch.testing.assert_close(
        output.ion_probability, expected.repeat_interleave(2) * 0.5
    )
    torch.testing.assert_close(output.sink_probability, torch.tensor([0.5]))
    torch.testing.assert_close(
        output.ion_probability.sum() + output.sink_probability.sum(), torch.tensor(1.0)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_spectrum_loss_matches_cpu_and_backpropagates():
    """Formula validity and precursor exclusion reductions work on CUDA too."""
    from copy import deepcopy

    torch.manual_seed(9)
    cpu = create_spectrum_generator(competitive_config()).eval()
    data, _ = ReactionStructureBuilder(cpu).build(
        Compound.from_smiles("CC(O)N"),
        [Adduct.parse("[M+H]+")] * 2,
        [20.0, 40.0],
        [[18.033826, 44.049476]] * 2,
        [[0.5, 1.0]] * 2,
    )

    def forward(model, batch):
        features = model.feature_model(batch, training_pool=True)
        return model.spectrum_model(
            batch.spectrum_batch,
            reaction_h=features.reaction_h,
            collision_energy_h=features.collision_energy_h,
            source_embeddings=features.source_embeddings,
        )

    gpu = create_spectrum_generator(competitive_config()).eval().cuda()
    gpu.load_state_dict(cpu.state_dict())
    gpu_data = deepcopy(data).to("cuda")
    expected = forward(cpu, data)
    actual = forward(gpu, gpu_data)
    torch.testing.assert_close(actual.loss.cpu(), expected.loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        actual.intensity.cpu(), expected.intensity, rtol=1e-4, atol=1e-5
    )
    actual.loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in gpu.spectrum_model.state_heads.parameters()
    )


@pytest.mark.parametrize("exponent", [0, -0.5, float("nan"), float("inf")])
def test_invalid_intensity_exponent(exponent):
    from fredis.ml.specgen.fragment_spectrum_model import SpectrumLossSettings
    with pytest.raises(ValueError, match="intensity_exponent"):
        SpectrumLossSettings(intensity_exponent=exponent)


def test_log_path_probabilities_survive_underflow_and_merge_gradients():
    from fredis.ml.specgen.competitive_spectrum_model import (
        propagate_log_probability, segment_log_softmax,
    )
    # Two extremely unlikely branches join at the same fragment.
    source = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3])
    target = torch.tensor([1, 2, -1, 3, -1, 3, -1, 3, -1])
    logits = torch.tensor([-1000., -1001., 0., -400., 0., -400., 0., 0., 0.], requires_grad=True)
    log_rates = segment_log_softmax(logits, source, 4)
    retained, sink = propagate_log_probability(
        torch.tensor([0., -torch.inf, -torch.inf, -torch.inf]), source, target, log_rates
    )
    expected = torch.logsumexp(torch.stack((log_rates[0] + log_rates[3],
                                           log_rates[1] + log_rates[5])), 0) + log_rates[7]
    torch.testing.assert_close(retained[3], expected)
    assert retained[3] < -1400 and retained[3].exp() == 0
    (-retained[3]).backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[[0, 1, 3, 5]] < 0).all()
    torch.testing.assert_close(torch.cat((retained, sink)).exp().sum(), torch.tensor(1.))


def test_log_formula_merge_loss_does_not_clamp_small_probabilities():
    from types import SimpleNamespace
    from fredis.ml.specgen.competitive_spectrum_model import segment_logsumexp
    from fredis.ml.specgen.fragment_spectrum_model import SpectrumLossSettings
    model = create_spectrum_generator(competitive_config()).spectrum_model
    data = SimpleNamespace(
        decoded=SimpleNamespace(node_sample_index=torch.tensor([0, 0, 0])),
        state_node_index=torch.tensor([0, 1, 2]), state_mz=torch.tensor([100., 50., 50.]),
        ion_type_strs=("[M+H]+",), state_adduct_index=torch.zeros(3, dtype=torch.long),
        precursor_adducts=("[M+H]+",), target_intensity=torch.tensor([0., 1., 1.]),
    )
    log_probabilities = torch.tensor([0., -1400., -1401.], requires_grad=True)
    _, loss, metrics = model._spectrum_loss(
        data, log_probabilities.exp(), torch.zeros(3, dtype=torch.long),
        torch.tensor([[10., 1.], [5., 1.], [5., 1.]]), 1,
        SpectrumLossSettings(cosine_loss_weight=0),
        final_log_probability=log_probabilities,
    )
    expected = -torch.logsumexp(log_probabilities[1:], 0)
    torch.testing.assert_close(metrics['transition_cross_entropy'], expected)
    torch.testing.assert_close(metrics['transition_cross_entropy_without_precursor'], torch.tensor(0.))
    loss.backward()
    assert torch.isfinite(log_probabilities.grad).all()
    assert (log_probabilities.grad[1:] < 0).all()
    # Empty groups and entirely unreachable inputs must have finite gradients.
    values = torch.tensor([-torch.inf, -torch.inf, -1000.], requires_grad=True)
    merged = segment_logsumexp(values, torch.tensor([0, 0, 1]), 3)
    assert torch.isneginf(merged[[0, 2]]).all()
    merged[1].backward()
    assert torch.isfinite(values.grad).all()


def test_validation_persists_ion_assignment_facets(tmp_path):
    import csv
    import json
    from fredis.ml.training.fragment_tree_training.spectrum_validation import (
        validate_spectra,
    )

    torch.manual_seed(9)
    generator = create_spectrum_generator(competitive_config()).eval()
    data, _ = ReactionStructureBuilder(generator).build(
        Compound.from_smiles("CC(O)N"),
        [Adduct.parse("[M+H]+")],
        [20.0],
        [[18.033826, 44.049476]],
        [[0.5, 1.0]],
    )
    validate_spectra(generator, [data], tmp_path, "epoch_1", global_step=1, epoch=1)
    payload = json.loads((tmp_path / "spectrum_validation/epoch_1.json").read_text())
    assert payload["spectra"]
    assert "fragment_ion_adduct_accuracy" in payload["spectra"][0]
    catalog = json.loads((tmp_path / "metrics/index.json").read_text())
    metrics = {entry["name"].split("/", 2)[2] for entry in catalog["series"]}
    assert "ion_assignment_coverage_mean" in metrics
    assert any(
        name.startswith("ion_assignment_coverage@main_adduct:") for name in metrics
    )
    assert any(
        name.startswith("ion_assignment_coverage@collision_energy:") for name in metrics
    )


def test_validation_intensity_filter_matches_inference_export(tmp_path):
    import json
    from dataclasses import replace
    from fredis.ml.specgen.competitive_spectrum_model import merge_candidate_peaks
    from fredis.ml.training.fragment_tree_training.spectrum_validation import (
        validate_spectra,
    )

    torch.manual_seed(9)
    generator = create_spectrum_generator(competitive_config()).eval()
    data, _ = ReactionStructureBuilder(generator).build(
        Compound.from_smiles("CC(O)N"),
        [Adduct.parse("[M+H]+")],
        [20.0],
        [[18.033826, 44.049476]],
        [[0.5, 1.0]],
    )
    generator.set_generation(
        replace(generator.generation, peak_intensity_threshold=0.5)
    )
    expected = []
    unfiltered_count = 0
    for output in generator.predict_prepared_batches(data):
        merged = merge_candidate_peaks(output.spectra)
        unfiltered_count += len(merged.mz)
        expected.extend(
            (float(mz), float(value))
            for mz, value in zip(merged.mz, merged.intensity)
            if float(value) > 0.5
        )
    assert len(expected) < unfiltered_count
    validate_spectra(generator, [data], tmp_path, "epoch_1", global_step=1, epoch=1)
    payload = json.loads((tmp_path / "spectrum_validation/epoch_1.json").read_text())
    record = payload["spectra"][0]
    assert "hydrogen_shift_mae" not in record
    assert [p["intensity"] for p in record["original_peaks"]] == [1.0]
    # The validation artifact must retain exactly the peaks exported by inference.
    actual = [(p["mz"], p["intensity"]) for p in record["generated_peaks"]]
    torch.testing.assert_close(
        torch.tensor(sorted(actual)), torch.tensor(sorted(expected))
    )
