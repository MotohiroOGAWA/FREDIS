"""Current reaction-pattern routing, input features, and parameter inheritance."""

import pytest
import torch
from fredis.ml.specgen.components.reaction.reaction_instance_encoder import ReactionInstanceEncoder

@pytest.mark.parametrize("node_dim", [8, 4])
def test_training_encoder_routes_mixed_patterns_and_restores_reaction_order(
    node_dim
):
    torch.manual_seed(5)
    model = ReactionInstanceEncoder(
        4,
        8,
        (2, 1, 1),
        num_heads=2,
        pattern_smarts={0: "CC", 1: "CCC"},
        pattern_model_params={str(pid): {"hidden_dim": node_dim} for pid in (0, 1)},
    )
    assert all(config["max_degree"] == 4 for config in model.pattern_model_params.values())
    assert all(encoder.centrality.max_degree == 4 for encoder in model.reactant_encoders.values())
    atoms = torch.randn(7, 4, requires_grad=True)
    kwargs = dict(
        reaction_type=torch.tensor([[1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        reaction_source_atom_ptr=torch.tensor([0, 3, 5, 8]),
        reaction_source_atom_index=torch.tensor([0, 1, 2, 3, 4, 4, 5, 6]),
        source_atom_features=atoms,
        source_edge_index=torch.empty((2, 0), dtype=torch.long),
        source_edge_attr=torch.empty((0, 0)),
    )
    output = model(**kwargs)
    assert output.shape == (3, 8)
    # Each match is equivalent to processing it alone, even in a mixed-pattern batch.
    for row in range(3):
        start, end = kwargs["reaction_source_atom_ptr"][row : row + 2]
        one = {
            **kwargs,
            "reaction_type": kwargs["reaction_type"][row : row + 1],
            "reaction_source_atom_ptr": torch.tensor([0, end - start]),
            "reaction_source_atom_index": kwargs["reaction_source_atom_index"][
                start:end
            ],
        }
        torch.testing.assert_close(output[row : row + 1], model(**one))
    output.square().sum().backward()
    for encoder in model.reactant_encoders.values():
        assert encoder.reactant_projection.weight.grad is not None
        assert encoder.layers[0].q_proj.weight.grad is not None
    assert atoms.grad is not None


def test_fine_tuning_remaps_existing_pattern_encoder_and_trains_new_pattern(tmp_path):
    from torch import nn
    from fredis.ml.specgen.fine_tuning import configure_pattern_training
    from fredis.ml.training.fragment_tree_training.fine_tuning import (
        initialize_from_base,
    )

    class Feature(nn.Module):
        def __init__(self, smarts):
            super().__init__()
            self.reaction_instance_encoder = ReactionInstanceEncoder(
                4, 8, (len(smarts), 1, 1), num_heads=2, pattern_smarts=smarts
            )

        def freeze_mol_encoder(self):
            pass

    class Generator(nn.Module):
        def __init__(self, smarts):
            super().__init__()
            self.mol_encoder = nn.Linear(4, 4)
            self.feature_model = Feature(smarts)

    old = Generator({0: "CC"})
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_state_dict": old.state_dict()}, checkpoint)
    new = Generator({0: "CCC", 1: "CC"})
    config = {
        "frozen_pattern_ids": [1],
        "pattern_mapping": [[0, 1]],
        "source_checkpoint": str(checkpoint),
    }
    configure_pattern_training(new, config)
    initialize_from_base(new, {"fine_tuning": config})
    before = old.feature_model.reaction_instance_encoder.reactant_encoders["0"]
    inherited = new.feature_model.reaction_instance_encoder.reactant_encoders["1"]
    added = new.feature_model.reaction_instance_encoder.reactant_encoders["0"]
    torch.testing.assert_close(before.readout.weight, inherited.readout.weight)
    assert not inherited.readout.weight.requires_grad
    before_position = old.feature_model.reaction_instance_encoder.endpoint_projections['0']
    inherited_position = new.feature_model.reaction_instance_encoder.endpoint_projections['1']
    torch.testing.assert_close(before_position.weight, inherited_position.weight)
    assert not inherited_position.weight.requires_grad
    assert new.feature_model.reaction_instance_encoder.endpoint_projections['0'].weight.requires_grad
    assert all(parameter.requires_grad for parameter in added.parameters())
    from fredis.ml.specgen.components.reaction.reactant_atom_attention import build_molecular_batch
    atoms = torch.randn(3, 4)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    batch = build_molecular_batch(atoms, edges)
    sites = torch.tensor([[0, 1]])
    torch.testing.assert_close(before(batch, sites), inherited(batch, sites))


def edges(pairs):
    return torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).T.contiguous()


def test_pattern_input_uses_labels_positions_and_configurable_cross_attention():
    model = ReactionInstanceEncoder(
        3, 8, (2, 1, 1), num_heads=2, pattern_smarts={0: "CC", 1: "CC"},
        pattern_model_params={
            "0": dict(hidden_dim=8, num_heads=2, num_layers=2),
            "1": dict(hidden_dim=12, num_heads=3, num_layers=3),
        },
    )
    chemical = torch.randn(6, 3, requires_grad=True)
    args = dict(source_atom_features=chemical,
                source_edge_index=edges([(0, 1), (0, 2), (2, 3), (3, 4), (4, 5)]),
                reaction_type=torch.tensor([[0, 0, 0], [1, 0, 0]]),
                reaction_source_atom_ptr=torch.tensor([0, 2, 4]),
                reaction_source_atom_index=torch.tensor([0, 1, 0, 1]))
    captured = []
    handle = model.reactant_encoders["0"].reactant_projection.register_forward_pre_hook(
        lambda module, values: captured.append(values[0]))
    output = model(**args)
    handle.remove()
    assert output.shape == (2, 8)
    torch.testing.assert_close(captured[0][..., :3], chemical[:2].unsqueeze(0))
    torch.testing.assert_close(captured[0][..., 3:], torch.eye(2).unsqueeze(0))
    assert len(model.reactant_encoders["1"].layers) == 3
    output[:, 0].sum().backward()
    assert chemical.grad[5].abs().sum() > 0


