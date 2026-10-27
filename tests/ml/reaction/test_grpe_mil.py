"""GRPE symmetries, exact reaction history graph MIL, and shared training/inference scores."""

from types import SimpleNamespace
from dataclasses import replace
import math
import pytest
import torch
from torch.nn import functional as F
from fredis.ml.specgen.components.reaction.reaction_transformer import (
    ReactionMemoryEncoder,
    ReactionStateDecoder,
    reaction_geometry,
)
from fredis.ml.training.fragment_tree_training.branch_mil import (
    softmax_mil_log_probability,
    branch_path_mil_loss,
)
from fredis.ml.training.fragment_tree_training.model import (
    ReactionFragmentTreeTrainingModel,
)
from .test_canonical_workflow import prepared
from fredis.ml.input.branch_teacher import BranchTeacher


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mil_extremes(dtype):
    good = torch.tensor([0.95, 0.05], dtype=dtype).log().requires_grad_()
    bad = torch.tensor([0.05, 0.05], dtype=dtype).log()
    both = torch.tensor([0.95, 0.95], dtype=dtype).log()
    loss = -softmax_mil_log_probability(good)
    assert loss < 0.06
    assert -softmax_mil_log_probability(bad) > 2.9
    torch.testing.assert_close(
        loss, -softmax_mil_log_probability(both), atol=0.003, rtol=0.01
    )
    loss.backward()
    assert torch.isfinite(good.grad).all()
    single = torch.tensor([-10000.0], requires_grad=True)
    torch.testing.assert_close(softmax_mil_log_probability(single), single[0])
    torch.testing.assert_close(
        softmax_mil_log_probability(torch.tensor([-torch.inf, -2.0])),
        torch.tensor(-2.0),
    )
    for scores in (torch.empty(0), torch.tensor([-torch.inf])):
        with pytest.raises(ValueError):
            softmax_mil_log_probability(scores)


def test_grpe_permutation_padding_and_gradient():
    torch.manual_seed(4)
    model = ReactionMemoryEncoder(16, heads=4).eval()
    x = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    distance = torch.randint(0, 6, (2, 5, 5))
    relations = torch.randint(0, 2, (2, 5, 5, 4)).float()
    out = model(x, distance, relations, valid)
    p = torch.tensor([3, 1, 0, 4, 2])
    permuted = model(
        x[:, p],
        distance[:, p][:, :, p],
        relations[:, p][:, :, p],
        valid[:, p],
    )
    torch.testing.assert_close(permuted, out[:, p], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(out).all() and not out[~valid].any()
    out.square().sum().backward()
    for layer in model.encoder.layers:
        layer = layer.self_attn.relative_encoding
        for encoding in (*layer.spatial, *layer.relations):
            assert (
                encoding.weight.grad is not None
                and torch.isfinite(encoding.weight.grad).all()
            )


def test_decoder_set_invariance_mapping_and_no_leakage():
    torch.manual_seed(12)
    model = ReactionStateDecoder(16).eval()
    memory = torch.randn(2, 6, 16, requires_grad=True)
    states = torch.tensor([[0, 3, -1], [3, 0, -1], [-1, -1, -1]])
    owners = torch.tensor([1, 1, 0])
    adduct = torch.randn(1, 16).expand(3, -1)
    root = torch.randn(2, 16, requires_grad=True)
    out = model(
        memory, states, owners, adduct, root[owners], torch.ones(2, 6, dtype=torch.bool)
    )
    torch.testing.assert_close(out[0], out[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        out[2:], model(memory[:1], states[2:], torch.tensor([0]), adduct[2:], root[:1])
    )
    assert out.shape == (3, 16)
    out.square().sum().backward()
    assert memory.grad.abs().sum() > 0
    assert root.grad.abs().sum() > 0


def history_data(shared=False, unequal=False):
    # Alternative target paths, with one shared prefix when requested.
    if shared:
        source, target, compounds = [0, 1, 1, 2, 3], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 4]
        depth = [0, 1, 2, 2, 3, 3]
    else:
        source, target, compounds = [0, 0, 1, 2], [1, 2, 3, 4], [0, 1, 2, 3, 3]
        depth = [0, 1, 1, 2, 2]
        if unequal:
            compounds[1] = 3
    source, target = torch.tensor(source), torch.tensor(target)
    n = len(compounds)
    e = SimpleNamespace(
        source_index=source,
        target_index=target,
        reaction_index=torch.arange(len(source)),
        compound_index=torch.tensor(compounds),
        branch_group_index=torch.zeros(n, dtype=torch.long),
        root_index=torch.tensor([0]),
    )
    target_compound = max(compounds)
    data = SimpleNamespace(
        execution_graph=e,
        device=torch.device("cpu"),
        num_branch_groups=1,
        state_reaction_count=torch.tensor(depth),
        compound_graph=SimpleNamespace(
            source_compound_index=e.compound_index[source],
            target_compound_index=e.compound_index[target],
            reaction_index=e.reaction_index,
        ),
        reaction_contexts=SimpleNamespace(
            edge_ptr=torch.tensor([0, len(source)]),
            edge_index=torch.arange(len(source)),
            edge_label=torch.ones(len(source), dtype=torch.int8),
        ),
        annotations=SimpleNamespace(
            spectra=SimpleNamespace(
                branch_group_index=torch.tensor([0]), peak_ptr=torch.tensor([0, 1])
            ),
            assignments=SimpleNamespace(
                formula_explanation_ptr=torch.tensor([0, 1]),
                formula_explanation_index=torch.tensor([0]),
            ),
        ),
        formula_explanations=SimpleNamespace(
            compound_index=torch.tensor([target_compound])
        ),
    )
    prepare_branch_teacher(data)
    return data


def prepare_branch_teacher(data):
    positive = (data.reaction_contexts.edge_label == 1).nonzero().flatten().tolist()
    negative = (data.reaction_contexts.edge_label == 0).nonzero().flatten().tolist()
    data.branch_teacher = BranchTeacher.from_peak_edges(
        data.execution_graph,
        data.state_reaction_count,
        (
            [
                (
                    0,
                    0,
                    positive,
                    sorted(
                        set(data.execution_graph.target_index[positive].tolist())
                        - set(data.execution_graph.source_index[positive].tolist())
                    ),
                )
            ]
            if positive
            else []
        ),
        negative_candidates=negative,
    )


@pytest.mark.parametrize(
    "shared,unequal", [(False, False), (True, False), (False, True)]
)
def test_history_mil_matches_explicit_unique_prefixes_and_gradients(shared, unequal):
    data = history_data(shared, unequal)
    logits = torch.linspace(
        -1, 2, data.execution_graph.source_index.numel(), requires_grad=True
    )
    scores = F.logsigmoid(logits)
    actual = branch_path_mil_loss(logits, data)[0]
    if shared:
        bags = [
            scores[:1],
            scores[0] + scores[1:3],
            scores[0] + torch.stack((scores[1] + scores[3], scores[2] + scores[4])),
        ]
    else:
        bags = [scores[:2], torch.stack((scores[0] + scores[2], scores[1] + scores[3]))]
    expected = -torch.stack([softmax_mil_log_probability(bag) for bag in bags]).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=2e-6,
        rtol=2e-5,
    )


def test_negative_and_ignore_loss():
    data = history_data()
    data.reaction_contexts.edge_label[:] = 0
    data.annotations.assignments.formula_explanation_index = torch.empty(
        0, dtype=torch.long
    )
    data.annotations.assignments.formula_explanation_ptr = torch.tensor([0, 0])
    prepare_branch_teacher(data)
    logits = torch.ones(4, requires_grad=True)
    high = branch_path_mil_loss(logits, data)[0]
    low = branch_path_mil_loss(logits - 3, data)[0]
    assert high > low
    high.backward()
    assert (logits.grad[:2] > 0).all()
    assert (logits.grad[2:] == 0).all()
    data.reaction_contexts.edge_label[:] = -1
    prepare_branch_teacher(data)
    assert branch_path_mil_loss(logits, data)[0] == 0


def test_training_memory_shared_and_spectrum_conserves_mass():
    model, data = prepared()
    model.eval()
    features = model.feature_model(data)
    static = model.feature_model.encode_static(data)
    cached = model.feature_model(data, static_features=static)
    torch.testing.assert_close(features.reaction_h, cached.reaction_h)
    torch.testing.assert_close(features.state_h, cached.state_h)
    logits = model.feature_model.score_execution(data, features)
    log_probs = F.logsigmoid(logits)
    assert (log_probs <= 0).all()
    trainer = ReactionFragmentTreeTrainingModel(
        model.feature_model, spectrum_model=model.spectrum_model
    )
    output = trainer(data)
    assert output.spectrum_output.metrics["probability_mass_error"] < 1e-5
    output.loss.backward()
    assert (
        model.feature_model.memory_encoder.encoder.layers[0]
        .self_attn.q_proj.weight.grad.abs()
        .sum()
        > 0
    )
    assert model.feature_model.state_decoder.bos.grad.abs().sum() > 0
    assert model.spectrum_model.fragment_gat.layers[0].gat.node_projection.weight.grad.abs().sum() > 0


def test_geometry_atom_relabeling_invariance():
    model, data = prepared()
    registry, graph = data.reactions, data.source_graph
    ids, distance, relations = reaction_geometry(data)[:3]
    n = graph.num_nodes
    # Permute within each molecule; all atom-referencing registry tensors follow.
    permutation = torch.arange(n)
    for a, b in zip(graph.ptr[:-1], graph.ptr[1:]):
        permutation[a:b] = torch.arange(int(b) - 1, int(a) - 1, -1)
    changed_graph = graph.clone()
    changed_graph.edge_index = permutation[graph.edge_index]
    changed_registry = replace(
        registry,
        reaction_source_atom_index=permutation[registry.reaction_source_atom_index],
        reaction_retained_atom_index=permutation[registry.reaction_retained_atom_index],
    )
    changed = SimpleNamespace(reactions=changed_registry, source_graph=changed_graph)
    other = reaction_geometry(changed)
    for a, b in zip((ids, distance, relations), other):
        torch.testing.assert_close(a, b)


def test_memory_is_encoded_once_then_reused_for_states_and_conditions():
    from unittest.mock import patch

    model, data = prepared()
    model.eval()
    with patch.object(
        model.feature_model.memory_encoder,
        "forward",
        wraps=model.feature_model.memory_encoder.forward,
    ) as encode:
        static = model.feature_model.encode_static(data)
        first = model.feature_model(data, static_features=static)
        second = model.feature_model(data, static_features=static)
        assert encode.call_count == 1
        torch.testing.assert_close(first.state_h, second.state_h)
    assert "_grpe_geometry" not in data.__getstate__()


def test_specification_configuration():
    from .TestReactionPipeline import config
    from fredis.ml.specgen.spectrum_generator import create_spectrum_generator

    cfg = config()
    cfg["reaction_transformer"] = dict(
        architecture="grpe_encoder_decoder",
        hidden_dim=16,
        encoder_layers=2,
        decoder_layers=2,
        num_heads=4,
        dropout=0.0,
        ffn_mult=2,
        norm_first=True,
        grpe=dict(
            enabled=True,
            max_spatial_distance=8,
            use_spatial_encoding=True,
            use_edge_encoding=True,
            use_graph_encoded_value=True,
            use_reaction_relations=True,
        ),
        decoder=dict(
            state_representation="permutation_invariant",
            use_bos=True,
            use_absolute_positional_encoding=False,
        ),
        branch=dict(output="log_probability", activation="logsigmoid"),
        spectrum=dict(
            activation="log_softmax", include_self_event=True, include_sink_event=True
        ),
    )
    cfg["branch_mil"] = dict(
        enabled=True,
        pooling="softmax_weighted",
        temperature=0.2,
        depth_reduction="mean",
        peak_reduction="mean",
        negative_weight=0.5,
    )
    model = create_spectrum_generator(cfg)
    assert model.feature_model.max_reaction_distance == 8
    assert model.branch_mil["temperature"] == 0.2


def test_prepared_teacher_rejects_negative_edges_in_positive_dag():
    _, data = prepared()
    teacher = data.branch_teacher
    labels = teacher.edge_label.clone()
    labels[teacher.edge_index[0]] = 0
    with pytest.raises(ValueError, match="memberships disagree"):
        replace(teacher, edge_label=labels).validate(data)


def test_history_mil_small_temperature_and_extreme_scores():
    data = history_data()
    logits = torch.full((4,), -1000.0, requires_grad=True)
    loss = branch_path_mil_loss(logits, data, temperature=1e-7)[0]
    torch.testing.assert_close(loss, torch.tensor(1500.0))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def add_negative_edges(data, parents):
    e = data.execution_graph
    first_node = e.compound_index.numel()
    first_edge = e.source_index.numel()
    parent = torch.tensor(parents, dtype=torch.long)
    e.source_index = torch.cat((e.source_index, parent))
    e.target_index = torch.cat(
        (e.target_index, torch.arange(first_node, first_node + len(parents)))
    )
    e.reaction_index = torch.arange(first_edge + len(parents))
    e.compound_index = torch.cat(
        (e.compound_index, torch.arange(first_node, first_node + len(parents)))
    )
    e.branch_group_index = torch.zeros_like(e.compound_index)
    data.state_reaction_count = torch.cat(
        (data.state_reaction_count, data.state_reaction_count[parent] + 1)
    )
    graph = data.compound_graph
    graph.source_compound_index = e.compound_index[e.source_index]
    graph.target_compound_index = e.compound_index[e.target_index]
    graph.reaction_index = e.reaction_index
    context = data.reaction_contexts
    context.edge_ptr = torch.tensor([0, first_edge + len(parents)])
    context.edge_index = e.reaction_index
    context.edge_label = torch.cat(
        (context.edge_label, torch.zeros(len(parents), dtype=torch.int8))
    )
    prepare_branch_teacher(data)


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 2.0])
def test_mil_balances_negatives_within_alternative_parent_groups(weight):
    data = history_data()
    # Depth one uses root 0; depth two jointly uses alternative parents 1 and 2.
    add_negative_edges(data, [0, 1, 1, 1, 2])
    logits = torch.tensor([0.0] * 4 + [-2.0, 2.0, -1.0, 3.0, 3.0], requires_grad=True)
    actual = branch_path_mil_loss(logits, data, negative_weight=weight)[0]
    positive = -F.logsigmoid(logits[0])
    root_negative = F.softplus(logits[4])
    alternative_negative = F.softplus(logits[5:]).mean()
    expected = (
        (positive + weight * root_negative) / (1 + weight)
        + (2 * positive + weight * alternative_negative) / (1 + weight)
    ) / 2
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    # Positive MIL distributes gradients across equally likely paths.
    torch.testing.assert_close(actual_gradient[4:], expected_gradient[4:])


def test_terminal_negatives_are_balanced_as_next_depth_of_same_peak():
    data = history_data()
    # States 3 and 4 are alternative histories of the same target Compound.
    add_negative_edges(data, [3, 4])
    logits = torch.tensor([0.0] * 4 + [-2.0, 2.0], requires_grad=True)
    actual = branch_path_mil_loss(logits, data)[0]
    positive = -F.logsigmoid(logits[0]) * 1.5
    expected = (positive * 2 + F.softplus(logits[4:]).mean()) / 3
    torch.testing.assert_close(actual, expected)
    disabled = branch_path_mil_loss(logits, data, negative_weight=0)[0]
    torch.testing.assert_close(disabled, positive)


def test_reaction_features_do_not_depend_on_molecular_encoder_weights():
    model, data = prepared()
    model.eval()
    feature = model.feature_model
    captured = []
    hook = feature.reaction_instance_encoder.register_forward_hook(
        lambda module, args, output: captured.append(output.detach().clone())
    )
    with torch.no_grad():
        before = feature(data)
        for parameter in model.mol_encoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
        after = feature(data)
    hook.remove()
    torch.testing.assert_close(captured[0], captured[1])
    torch.testing.assert_close(before.reaction_h, after.reaction_h)
    assert not torch.allclose(before.state_h, after.state_h)
    assert not torch.allclose(before.source_embeddings, after.source_embeddings)
    model.zero_grad(set_to_none=True)
    feature(data).reaction_h[:, 0].sum().backward()
    assert all(parameter.grad is None for parameter in model.mol_encoder.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in feature.reaction_instance_encoder.parameters()
    )


def test_fine_tuning_updates_molecular_encoder_but_preserves_local_reactions():
    from fredis.ml.specgen.fine_tuning import configure_pattern_training

    model, data = prepared()
    feature = model.feature_model
    reaction = feature.reaction_instance_encoder
    # Reconfiguration must also clear a previously requested molecular freeze.
    feature.freeze_mol_encoder()
    configure_pattern_training(
        model, {"frozen_pattern_ids": list(reaction.reactant_encoders)}
    )
    trainer = ReactionFragmentTreeTrainingModel(
        feature, spectrum_model=model.spectrum_model
    )
    trainer.train()
    assert model.mol_encoder.training
    assert all(p.requires_grad for p in model.mol_encoder.parameters())
    assert all(not p.requires_grad for p in reaction.parameters())
    captured = []
    hook = reaction.register_forward_hook(
        lambda module, args, output: captured.append(output.detach().clone())
    )
    optimizer = torch.optim.SGD(model.mol_encoder.parameters(), lr=0.01)
    before = [p.detach().clone() for p in model.mol_encoder.parameters()]
    trainer(data).loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.mol_encoder.parameters()
    )
    assert all(p.grad is None for p in reaction.parameters())
    optimizer.step()
    with torch.no_grad():
        feature(data)
    hook.remove()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.mol_encoder.parameters())
    )
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)


def test_decoder_conditions_change_empty_state_and_receive_gradients():
    torch.manual_seed(31)
    decoder = ReactionStateDecoder(16).eval()
    memory = torch.randn(1, 3, 16)
    states = torch.empty((2, 0), dtype=torch.long)
    owners = torch.zeros(2, dtype=torch.long)
    adduct = torch.randn(2, 16, requires_grad=True)
    root = torch.randn(2, 16, requires_grad=True)
    baseline = decoder(
        memory, states, owners, adduct[:1].expand(2, -1), root[:1].expand(2, -1)
    )
    changed_adduct = decoder(memory, states, owners, adduct, root[:1].expand(2, -1))
    changed_root = decoder(memory, states, owners, adduct[:1].expand(2, -1), root)
    torch.testing.assert_close(baseline[0], baseline[1])
    assert not torch.allclose(changed_adduct[0], changed_adduct[1])
    assert not torch.allclose(changed_root[0], changed_root[1])
    (changed_adduct[:, 0].sum() + changed_root[:, 0].sum()).backward()
    assert adduct.grad.abs().sum() > 0
    assert root.grad.abs().sum() > 0


def test_state_conditions_train_root_encoder_and_condition_projections():
    model, data = prepared()
    features = model.feature_model(data)
    features.state_h[:, 0].sum().backward()
    feature = model.feature_model
    for module in (
        feature.decoder_adduct_projection,
        feature.decoder_root_projection,
        feature.mol_encoder,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )


def test_negative_history_of_positive_compound_does_not_create_positive_bag():
    data = history_data()
    add_negative_edges(data, [3])
    logits = torch.linspace(-1, 1, 5, requires_grad=True)
    expected = branch_path_mil_loss(logits, data)
    data.execution_graph.compound_index[-1] = 3
    data.compound_graph.target_compound_index[-1] = 3
    actual = branch_path_mil_loss(logits, data)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after)
    actual[0].backward()
    assert torch.isfinite(logits.grad).all()


def test_spectrum_gat_uses_local_reactions_without_branch_transformer_gradients():
    from fredis.ml.specgen.fragment_spectrum_model import spectrum_batch_from_input

    model, data = prepared()
    model.eval()
    features = model.feature_model(data)
    cached = model.feature_model(data, static_features=model.feature_model.encode_static(data))
    torch.testing.assert_close(features.local_reaction_h, cached.local_reaction_h)
    batch = spectrum_batch_from_input(data, data.targets)
    spectrum = model.spectrum_model(
        batch,
        reaction_h=features.local_reaction_h,
        collision_energy_h=features.collision_energy_h,
        source_embeddings=features.source_embeddings,
    )
    spectrum.loss.backward()
    assert all(p.grad is None for p in model.feature_model.memory_encoder.parameters())
    assert all(p.grad is None for p in model.feature_model.state_decoder.parameters())
    for component in (model.spectrum_model.state_set_encoder,
                      model.spectrum_model.fragment_gat,
                      model.spectrum_model.fragment_projection,
                      model.spectrum_model.main_adduct_embedding,
                      model.mol_encoder,
                      model.feature_model.reaction_instance_encoder):
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0 for p in component.parameters())
    before = model.feature_model.score_execution(data, features).detach()
    with torch.no_grad():
        model.spectrum_model.fragment_projection.weight.add_(1)
    after_features = model.feature_model(data)
    after = model.feature_model.score_execution(data, after_features).detach()
    torch.testing.assert_close(before, after, atol=0, rtol=0)
