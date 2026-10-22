"""Mol Encoder architecture arguments for fragment-tree training."""

from __future__ import annotations

import argparse

from ....domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
from ...mol.mol_graphormer import (
    DEFAULT_MOL_GRAPHORMER_MAX_DEGREE,
    DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST,
    DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
)

# (argument, config key, type, default)
MOL_ENCODER_OPTIONS = (
    ("node-dim", "node_dim", int, 16),
    ("graph-dim", "graph_dim", int, 64),
    ("num-layers", "num_layers", int, 2),
    ("num-heads", "num_heads", int, 4),
    ("max-degree", "max_degree", int, DEFAULT_MOL_GRAPHORMER_MAX_DEGREE),
    (
        "max-spatial-dist",
        "max_spatial_dist",
        int,
        DEFAULT_MOL_GRAPHORMER_MAX_SPATIAL_DIST,
    ),
    ("max-edge-dist", "max_edge_dist", int, DEFAULT_MOL_GRAPHORMER_MAX_EDGE_DIST),
)


def add_mol_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the Mol Encoder architecture options."""
    for flag, key, kind, default in MOL_ENCODER_OPTIONS:
        parser.add_argument(
            "--" + flag,
            default=default,
            type=kind,
            help=f'Mol Encoder {key.replace("_", " ")} architecture setting.',
        )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Shared dropout for all newly trained neural networks.",
    )
    parser.add_argument(
        "--intensity-descriptors",
        "--descriptors",
        dest="descriptors",
        nargs="*",
        choices=SUPPORTED_DESCRIPTOR_NAMES,
        default=[],
        help="Fragment molecular descriptors used only by the post-branch intensity head "
        "(space-separated; give no names to use none).",
    )
