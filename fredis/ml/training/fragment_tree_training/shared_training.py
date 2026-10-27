"""Expand a shared architecture into the existing detailed training command."""

from copy import deepcopy
import argparse

from fredis.cli.base import namespace_argv
from fredis.domain.fragment import Fragmenter
from .architecture_options import (
    REACTION_TRANSFORMER_OPTIONS,
    fc_widths,
    nonnegative_int,
    positive_int,
)
from .training import (
    MODEL_OPTIONS,
    MOL_ENCODER_OPTIONS,
    MLPWidths,
    build_arg_parser as detailed_parser,
    main as run_detailed_training,
    training_model_config,
)
from .pattern_models import PATTERN_PARAMETERS
from .sources import inherit_model_config, prepared_dataset_paths


def architecture_destinations():
    names = {flag.replace("-", "_") for flag in MOL_ENCODER_OPTIONS}
    names.update(
        flag.replace("-", "_")
        for flag, (section, _, _, _) in MODEL_OPTIONS.items()
        if section != "generation_params" and flag not in ("ce-center", "ce-scale")
    )
    names.update("pattern_" + name for name in PATTERN_PARAMETERS)
    names.update(flag.replace("-", "_") for flag in REACTION_TRANSFORMER_OPTIONS)
    names.update(
        name + "_fc_" + suffix
        for name in ("ce", "fragment", "state")
        for suffix in ("layers", "mult")
    )
    names.update(("model_architecture", "model_architecture_json"))
    return names


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    individual = architecture_destinations()
    for action in detailed_parser()._actions:
        if isinstance(action, argparse._HelpAction) or action.dest in individual:
            continue
        parser._add_action(action)
    shared = parser.add_argument_group("Shared model architecture")
    for flag, default, help_text in (
        (
            "hidden-dim",
            128,
            "Common atom, molecule, reaction, adduct and CE feature width.",
        ),
        (
            "num-heads",
            4,
            "Common head count for all multihead-attention modules; divides hidden-dim.",
        ),
        (
            "num-layers",
            2,
            "Common molecular Graphormer, reaction atom, spectrum GAT and set-encoder depth.",
        ),
        ("ffn-mult", 4, "Common FFN intermediate-width multiplier."),
        (
            "max-degree",
            4,
            "Degree clipping for molecular and reaction-atom centrality encodings.",
        ),
        (
            "max-spatial-dist",
            4,
            "Atom shortest-path encoding limit for molecular and reaction-atom attention.",
        ),
        (
            "max-edge-dist",
            4,
            "Atom bond-path encoding limit for molecular and reaction-atom attention.",
        ),
        ("reaction-encoder-layers", 2, "Reaction-memory Transformer encoder layers."),
        (
            "reaction-decoder-layers",
            2,
            "Applied-reaction-set Transformer decoder layers.",
        ),
        (
            "reaction-instance-max-distance",
            4,
            REACTION_TRANSFORMER_OPTIONS["reaction-instance-max-distance"][2],
        ),
        (
            "fc-mult",
            2,
            "Symmetric FC width multiplier; three layers produce H,H*mult,H.",
        ),
    ):
        shared.add_argument(
            "--" + flag, type=positive_int, default=default, help=help_text
        )
    shared.add_argument(
        "--fc-layers",
        type=nonnegative_int,
        default=3,
        help="Number of FC hidden widths for CE, Fragment Head and State Head; zero selects a direct Linear.",
    )
    return parser


def detailed_namespace(args):
    """Expand shared scalars without changing the detailed configuration builder."""
    if args.hidden_dim % args.num_heads:
        raise ValueError("Shared hidden-dim must be divisible by num-heads")
    target = detailed_parser().parse_args(
        [
            "--dataset-dir",
            args.dataset_dir,
            "--output-dir",
            str(args.output_dir),
        ]
    )
    for action in detailed_parser()._actions:
        if hasattr(args, action.dest):
            setattr(target, action.dest, deepcopy(getattr(args, action.dest)))
    for name in (
        "node_dim",
        "graph_dim",
        "reaction_hidden_dim",
        "reaction_main_adduct_dim",
        "adduct_embedding_dim",
        "collision_energy_feature_dim",
    ):
        setattr(target, name, args.hidden_dim)
    for name in (
        "num_heads",
        "reaction_num_heads",
        "reaction_transformer_heads",
        "state_set_heads",
    ):
        setattr(target, name, args.num_heads)
    for name in (
        "ffn_mult",
        "reaction_transformer_ffn_mult",
        "spectrum_gat_ffn_mult",
        "state_set_ffn_mult",
    ):
        setattr(target, name, args.ffn_mult)
    target.num_layers = target.state_set_layers = target.spectrum_gat_layers = (
        args.num_layers
    )
    widths = MLPWidths(fc_widths(args.hidden_dim, args.fc_layers, args.fc_mult))
    for name in ("ce", "fragment", "state"):
        setattr(target, name + "_fc_dims", MLPWidths(widths))
    return target


def expand_arguments(args):
    """Return ordinary detailed CLI flags, including each new pattern's settings."""
    target = detailed_namespace(args)
    if not args.resume:
        config = inherit_model_config(
            training_model_config(target), *prepared_dataset_paths(args.dataset_dir)
        )
        frozen = set(
            map(str, config.get("fine_tuning", {}).get("frozen_pattern_ids", []))
        )
        if frozen:
            molecular = config["mol_encoder_params"]
            defaults = {
                key: default for key, _, default in MOL_ENCODER_OPTIONS.values()
            }
            expected = dict(
                node_dim=args.hidden_dim,
                graph_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ffn_mult=args.ffn_mult,
                max_degree=args.max_degree,
                max_spatial_dist=args.max_spatial_dist,
                max_edge_dist=args.max_edge_dist,
            )
            if (
                any(
                    molecular.get(key, defaults[key]) != value
                    for key, value in expected.items()
                )
                or config["reaction_model_params"].get("hidden_dim", 128)
                != args.hidden_dim
            ):
                raise ValueError(
                    "Shared architecture differs from the imported molecule/reaction architecture. "
                    "Use matching shared settings or fragment-tree-detailed for independent dimensions."
                )
        patterns = Fragmenter.from_dict(
            config["fragmenter_params"]
        ).reaction_pattern_set.patterns
        values = dict(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_mult=args.ffn_mult,
            max_degree=args.max_degree,
            max_spatial_dist=args.max_spatial_dist,
            max_edge_dist=args.max_edge_dist,
        )
        for name, value in values.items():
            setattr(
                target,
                "pattern_" + name,
                [
                    f"{pattern.pattern_id}={value}"
                    for pattern in patterns
                    if str(pattern.pattern_id) not in frozen
                ],
            )
    return namespace_argv(detailed_parser(), target)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_detailed_training(expand_arguments(args))


if __name__ == "__main__":
    main()
