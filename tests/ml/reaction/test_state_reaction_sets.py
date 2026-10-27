"""Applied reaction sets condition ion/shift predictions without changing flow."""

import math

import pytest
import torch

from fredis.ml.input.reaction_set import ReactionSet
from fredis.ml.specgen.competitive_spectrum_model import segment_logsumexp
from fredis.libs.mmkit.mmkit import Adduct, Compound
from .test_canonical_workflow import prepared

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_state_set_encoding_is_order_invariant_deduplicated_and_differentiable(device):
    generator, _ = prepared()
    model = generator.spectrum_model.to(device).eval()
    reaction_h = torch.randn(3, model.reaction_dim, device=device, requires_grad=True)
    sets, _ = ReactionSet.from_rows([(), (0, 1), (2,)], device=device)
    requested = torch.tensor([0, 1, 2, 1], device=device)
    calls = []
    handle = model.state_set_encoder.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].shape[0])
    )
    encoded = model._encode_reaction_sets(reaction_h, sets, requested)
    handle.remove()
    assert calls == [3]
    assert encoded.shape == (4, model.reaction_dim)
    assert torch.isfinite(encoded).all()  # Includes the empty root set.
    torch.testing.assert_close(encoded[1], encoded[3])
    permuted_index = sets.reaction_index.clone()
    permuted_index[:2] = permuted_index[:2].flip(0)
    permuted = ReactionSet(sets.reaction_ptr, permuted_index)
    reordered = model._encode_reaction_sets(reaction_h, permuted, requested)
    torch.testing.assert_close(encoded, reordered, rtol=1e-5, atol=1e-5)
    (encoded * torch.randn_like(encoded)).sum().backward()
    assert reaction_h.grad.abs().sum() > 0
    assert torch.isfinite(reaction_h.grad).all()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.state_set_encoder.parameters()
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["finite", "unreachable_set", "unreachable_fragment"])
def test_same_sets_share_states_and_distinct_sets_sum_joint_probabilities(
    device, monkeypatch, case
):
    from types import SimpleNamespace

    generator, _ = prepared()
    model = generator.spectrum_model.to(device).double().eval()
    # A->B and B->A have one canonical set; C->D has a distinct set.
    sets, rows = ReactionSet.from_rows([(0, 1), (1, 0), (2, 3)], device=device)
    assert rows.tolist() == [0, 0, 1]
    log_mass = [-1000.0, -1000.0 + math.log(2), -1000.0 + math.log(4)]
    if case == "unreachable_set":
        log_mass[2] = -torch.inf
    elif case == "unreachable_fragment":
        log_mass = [-torch.inf] * 3
    retained = torch.tensor(
        log_mass,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    inverse = torch.zeros(3, dtype=torch.long, device=device)
    log_fragment = segment_logsumexp(retained, inverse, 1)
    seen = []

    def encode(_reaction_h, _sets, set_index):
        seen.append(set_index.tolist())
        return set_index[:, None].double().expand(-1, model.reaction_dim)

    def score(
        _data,
        _molecular,
        _ce,
        _fragments,
        state_group,
        roots,
        *,
        reaction_set_h,
        state_rows,
        root_nodes,
        log,
    ):
        assert reaction_set_h.shape[0] == 2  # Not three history evaluations.
        table = reaction_set_h.new_tensor([[0.8, 0.2], [0.1, 0.9]])
        return table[reaction_set_h[state_group, 0].long(), state_rows].log()

    monkeypatch.setattr(model, "_encode_reaction_sets", encode)
    monkeypatch.setattr(model, "_score_states", score)
    data = SimpleNamespace()
    molecular = torch.zeros(
        (1, generator.mol_encoder.graph_dim), device=device, dtype=torch.float64
    )
    fragments = torch.tensor([[0, 0]], device=device)
    final, conditional = model._score_reaction_set_states(
        data,
        molecular,
        molecular,
        molecular,
        sets,
        rows,
        retained,
        inverse,
        log_fragment,
        fragments,
        torch.zeros(2, dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    )
    assert seen == [[0, 1]]
    if case == "unreachable_fragment":
        assert torch.isneginf(final).all()
        # Reporting remains normalized; zero mass contributes no ion intensity.
        torch.testing.assert_close(conditional.exp(), retained.new_tensor([0.45, 0.55]))
        return
    if case == "unreachable_set":
        expected_mass, expected_state, expected_grad = (
            [2.4, 0.6],
            [0.8, 0.2],
            [-1 / 3, -2 / 3, 0],
        )
    else:
        # (1 + 2) * State(AB) + 4 * State(CD), scaled by exp(-1000).
        expected_mass, expected_state, expected_grad = (
            [2.8, 4.2],
            [0.4, 0.6],
            [-2 / 7, -4 / 7, -1 / 7],
        )
    torch.testing.assert_close(final, retained.new_tensor(expected_mass).log() - 1000)
    torch.testing.assert_close(conditional.exp(), retained.new_tensor(expected_state))
    (-final[0]).backward()
    torch.testing.assert_close(retained.grad, retained.new_tensor(expected_grad))


def test_reaction_set_state_features_change_ion_states_without_changing_fragment_flow():
    generator, _ = prepared()
    generator.eval()
    source, adduct = Compound.from_smiles("CC(O)N"), Adduct.parse("[M+H]+")
    with torch.no_grad():
        before = generator.predict([source], [adduct], [20.0]).spectra

        def replace_context(_module, _args, result):
            return result + torch.arange(result.shape[1], device=result.device) * 2

        handle = generator.spectrum_model.state_set_encoder.register_forward_hook(
            replace_context
        )
        try:
            after = generator.predict([source], [adduct], [20.0]).spectra
        finally:
            handle.remove()
    for name in ["fragment_probability", "transition_probability", "sink_probability"]:
        torch.testing.assert_close(
            getattr(before, name), getattr(after, name), rtol=0, atol=0
        )
    assert not torch.allclose(before.state_probability, after.state_probability)
    totals = after.state_probability.new_zeros(
        after.fragment_probability.numel()
    ).scatter_add_(0, after.state_fragment_index, after.state_probability)
    torch.testing.assert_close(totals, torch.ones_like(totals))
