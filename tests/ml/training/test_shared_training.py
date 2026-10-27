"""Check shared CLI expansion against the detailed training path and real modules."""

from copy import deepcopy
import json

import pytest
import torch

from fredis.domain.fragment import Fragmenter
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.training.fragment_tree_training import shared_training, training
from fredis.ml.training.fragment_tree_training.architecture_options import fc_widths
from fredis.ml.training.fragment_tree_training.pattern_models import (
    apply_pattern_parameters,
    parse_pattern_parameters,
)
from fredis.ml.training.fragment_tree_training.sources import inherit_model_config
from ..reaction.TestReactionPipeline import chemistry, config


@pytest.fixture
def prepared(tmp_path):
    for split in ("train_structures", "validation_structures"):
        (tmp_path / split).mkdir()
    (tmp_path / "fragment-tree.training-data-config.pft").write_text(
        json.dumps({"chemistry_config": chemistry()})
    )
    return tmp_path


def shared_args(prepared, *extra):
    return shared_training.build_arg_parser().parse_args(
        ["--dataset-dir", str(prepared), "--output-dir", str(prepared / "run"), *extra]
    )


def expanded_config(prepared, *extra):
    args = shared_args(prepared, *extra)
    detailed = training.build_arg_parser().parse_args(
        shared_training.expand_arguments(args)
    )
    params = inherit_model_config(
        training.training_model_config(detailed),
        prepared / "train_structures",
        prepared / "validation_structures",
    )
    params = apply_pattern_parameters(params, parse_pattern_parameters(detailed))
    return detailed, params


@pytest.mark.parametrize(
    "layers,expected",
    [
        (0, []),
        (1, [256]),
        (2, [256, 256]),
        (3, [256, 512, 256]),
        (4, [256, 512, 512, 256]),
        (5, [256, 512, 1024, 512, 256]),
    ],
)
def test_symmetric_fc_widths(layers, expected):
    assert fc_widths(256, layers, 2) == expected


@pytest.mark.parametrize("values", [(0, 3, 2), (8, -1, 2), (8, 3, 0), (8, True, 2)])
def test_invalid_fc_widths(values):
    with pytest.raises(ValueError):
        fc_widths(*values)


def test_shared_arguments_expand_to_all_detailed_architectures(prepared):
    detailed, params = expanded_config(
        prepared,
        "--hidden-dim",
        "12",
        "--num-heads",
        "3",
        "--num-layers",
        "1",
        "--ffn-mult",
        "3",
        "--reaction-encoder-layers",
        "2",
        "--reaction-decoder-layers",
        "3",
        "--reaction-instance-max-distance",
        "7",
        "--max-spatial-dist",
        "5",
        "--max-edge-dist",
        "6",
        "--fc-layers",
        "3",
        "--fc-mult",
        "2",
        "--epochs",
        "7",
        "--ce-center",
        "20",
        "--dropout",
        "0.2",
    )
    mol, reaction, spectrum = (
        params[key]
        for key in (
            "mol_encoder_params",
            "reaction_model_params",
            "spectrum_model_params",
        )
    )
    assert mol["node_dim"] == mol["graph_dim"] == reaction["hidden_dim"] == 12
    assert (
        reaction["branch_main_adduct_dim"]
        == spectrum["adduct_embedding_dim"]
        == spectrum["collision_energy_dim"]
        == 12
    )
    assert (
        mol["num_heads"]
        == reaction["num_heads"]
        == spectrum["state_set_num_heads"]
        == 3
    )
    assert (
        mol["ffn_mult"]
        == spectrum["gat_ffn_mult"]
        == spectrum["state_set_ffn_mult"]
        == 3
    )
    assert (
        mol["num_layers"]
        == spectrum["gat_num_layers"]
        == spectrum["state_set_num_layers"]
        == 1
    )
    assert reaction["reaction_transformer"] == dict(
        encoder_layers=2,
        decoder_layers=3,
        num_heads=3,
        ffn_mult=3,
        max_reaction_distance=7,
    )
    for widths in ("ce_fc_dims", "fragment_fc_dims", "state_fc_dims"):
        assert spectrum[widths] == [12, 24, 12]
    patterns = Fragmenter.from_dict(
        params["fragmenter_params"]
    ).reaction_pattern_set.patterns
    assert set(reaction["pattern_model_params"]) == {
        str(p.pattern_id) for p in patterns
    }
    for pattern in reaction["pattern_model_params"].values():
        assert pattern == dict(
            hidden_dim=12,
            num_heads=3,
            num_layers=1,
            ffn_mult=3,
            max_degree=4,
            max_spatial_dist=5,
            max_edge_dist=6,
        )
    assert (
        detailed.epochs == 7
        and spectrum["ce_center"] == 20
        and params["dropout"] == 0.2
    )

    generator = create_spectrum_generator(params)
    assert generator.mol_encoder.encoder.ffn_mult == 3
    assert generator.feature_model.max_reaction_distance == 7
    assert len(generator.feature_model.memory_encoder.encoder.layers) == 2
    assert len(generator.feature_model.state_decoder.decoder.layers) == 3
    assert len(generator.spectrum_model.state_set_encoder.blocks) == 1
    for module in generator.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            assert module.embed_dim == 12 and module.num_heads == 3
    # All FFNs must receive the common multiplier, including previously fixed ones.
    ffns = [
        (name, module)
        for name, module in generator.named_modules()
        if name.endswith((".ffn", ".ff")) and isinstance(module, torch.nn.Sequential)
    ]
    assert ffns
    for name, module in ffns:
        linears = [
            layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)
        ]
        assert linears[0].out_features == 3 * linears[0].in_features, name
    from fredis.ml.common.layers.feed_forward import FeedForwardBlock

    blocks = [m for m in generator.modules() if isinstance(m, FeedForwardBlock)]
    assert blocks
    for block in blocks:
        assert block.linear_1.out_features == 3 * block.linear_1.in_features


def test_shared_entrypoint_forwards_ordinary_detailed_arguments(prepared, monkeypatch):
    calls = []
    monkeypatch.setattr(shared_training, "run_detailed_training", calls.append)
    shared_training.main(
        [
            "--dataset-dir",
            str(prepared),
            "--output-dir",
            str(prepared / "run"),
            "--hidden-dim",
            "256",
            "--num-heads",
            "8",
            "--fc-layers",
            "3",
        ]
    )
    args = training.build_arg_parser().parse_args(calls[0])
    assert args.node_dim == args.graph_dim == args.reaction_hidden_dim == 256
    assert args.fragment_fc_dims == [256, 512, 256]
    assert args.pattern_hidden_dim and all(
        value.endswith("=256") for value in args.pattern_hidden_dim
    )


def test_resume_does_not_reopen_imports_or_override_patterns(prepared, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail(
            "Resume must use saved architecture without reopening imported assets"
        )

    monkeypatch.setattr(shared_training, "inherit_model_config", fail)
    args = shared_args(prepared, "--resume", str(prepared / "last.pt"))
    detailed = training.build_arg_parser().parse_args(
        shared_training.expand_arguments(args)
    )
    assert detailed.resume == str(prepared / "last.pt")
    assert detailed.pattern_hidden_dim is None


def test_frozen_patterns_are_preserved_and_incompatible_shared_width_rejected(
    prepared, monkeypatch
):
    _, original = expanded_config(prepared, "--hidden-dim", "12", "--num-heads", "3")
    pid = next(iter(original["reaction_model_params"]["pattern_model_params"]))
    original["fine_tuning"] = {"frozen_pattern_ids": [int(pid)]}
    original["reaction_model_params"]["pattern_model_params"][pid]["hidden_dim"] = 24
    monkeypatch.setattr(
        shared_training, "inherit_model_config", lambda *args: deepcopy(original)
    )
    args = shared_args(prepared, "--hidden-dim", "12", "--num-heads", "3")
    detailed = training.build_arg_parser().parse_args(
        shared_training.expand_arguments(args)
    )
    assert all(
        not value.startswith(pid + "=") for value in (detailed.pattern_hidden_dim or [])
    )
    with pytest.raises(ValueError, match="fragment-tree-detailed"):
        shared_training.expand_arguments(
            shared_args(prepared, "--hidden-dim", "24", "--num-heads", "3")
        )


def test_detailed_fc_generation_remains_independent(prepared):
    args = training.build_arg_parser().parse_args(
        [
            "--dataset-dir",
            str(prepared),
            "--output-dir",
            str(prepared / "run"),
            "--node-dim",
            "8",
            "--graph-dim",
            "17",
            "--reaction-hidden-dim",
            "12",
            "--ce-fc-layers",
            "0",
            "--fragment-fc-layers",
            "5",
            "--fragment-fc-mult",
            "2",
            "--state-fc-dims",
            "7,9",
        ]
    )
    params = training.training_model_config(args)
    assert params["mol_encoder_params"]["graph_dim"] == 17
    assert params["spectrum_model_params"]["ce_fc_dims"] == []
    assert params["spectrum_model_params"]["fragment_fc_dims"] == [12, 24, 48, 24, 12]
    assert params["spectrum_model_params"]["state_fc_dims"] == [7, 9]


def test_invalid_shared_architecture_and_removed_role_option(prepared):
    with pytest.raises(ValueError, match="divisible"):
        shared_training.detailed_namespace(shared_args(prepared, "--hidden-dim", "11"))
    for parser in (shared_training.build_arg_parser(), training.build_arg_parser()):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--dataset-dir",
                    str(prepared),
                    "--output-dir",
                    str(prepared / "run"),
                    "--role-dim",
                    "8",
                ]
            )
    with pytest.raises(SystemExit):
        shared_args(prepared, "--graph-dim", "16")


def test_reaction_distance_legacy_key_and_conflicts():
    params = config()
    params["reaction_model_params"]["reaction_transformer"] = {
        "max_spatial_distance": 7
    }
    assert create_spectrum_generator(params).feature_model.max_reaction_distance == 7
    params["reaction_model_params"]["reaction_transformer"]["max_reaction_distance"] = 5
    with pytest.raises(ValueError, match="Conflicting"):
        create_spectrum_generator(params)


@pytest.mark.parametrize("command", ["fragment-tree", "fragment-tree-detailed"])
def test_public_cli_dispatch_and_tuning_command(prepared, monkeypatch, command):
    from fredis.cli.main import build_parser
    from fredis.cli.train.commands import fragment_tree, fragment_tree_detailed
    from fredis.ml.tuning.fragment_tree import expand_search

    calls = []
    module = fragment_tree if command == "fragment-tree" else fragment_tree_detailed
    monkeypatch.setattr(module, "run_training", calls.append)
    option = "--hidden-dim" if command == "fragment-tree" else "--graph-dim"
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            command,
            "--dataset-dir",
            str(prepared),
            "--output-dir",
            str(prepared / "run"),
            option,
            "24",
        ]
    )
    args.func(args)
    assert calls[0][calls[0].index(option) + 1] == "24"
    plan = expand_search(
        dict(
            trainingCommand=command,
            outputDir=str(prepared / "search"),
            baseArgs=["--dataset-dir", str(prepared)],
            parameters={option: [12, 24]},
        )
    )
    assert len(plan["trials"]) == 2
    assert all(
        trial["args"][:4] == ["-m", "fredis.cli", "train", command]
        for trial in plan["trials"]
    )


def test_zero_fc_layers_survive_shared_argument_serialization(prepared):
    _, params = expanded_config(prepared, "--fc-layers", "0")
    for key in ("ce_fc_dims", "fragment_fc_dims", "state_fc_dims"):
        assert params["spectrum_model_params"][key] == []
