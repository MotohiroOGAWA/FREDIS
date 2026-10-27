"""Shared architecture expansion used by both training command interfaces."""

import argparse


def positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Use a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return number


def nonnegative_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Use a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("Use a nonnegative integer")
    return number


def fc_widths(hidden_dim, layers, mult):
    """Symmetric hidden widths, including the base width at both ends.

    H=256 and mult=2 produce [256, 512, 256] for three layers,
    [256, 512, 512, 256] for four, and [256, 512, 1024, 512, 256] for five.
    Zero layers selects the existing direct Linear readout.
    """
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("FC hidden_dim must be a positive integer")
    if type(layers) is not int or layers < 0:
        raise ValueError("FC layers must be a nonnegative integer")
    if type(mult) is not int or mult < 1:
        raise ValueError("FC multiplier must be a positive integer")
    return [
        hidden_dim * mult ** min(index, layers - index - 1) for index in range(layers)
    ]


REACTION_TRANSFORMER_OPTIONS = {
    "reaction-encoder-layers": (
        "encoder_layers",
        2,
        "Reaction-memory Transformer encoder layers.",
    ),
    "reaction-decoder-layers": (
        "decoder_layers",
        2,
        "Applied-reaction-set Transformer decoder layers.",
    ),
    "reaction-transformer-heads": (
        "num_heads",
        None,
        "Reaction Transformer heads; defaults to reaction-num-heads.",
    ),
    "reaction-transformer-ffn-mult": (
        "ffn_mult",
        4,
        "Reaction Transformer FFN hidden-width multiplier.",
    ),
    "reaction-instance-max-distance": (
        "max_reaction_distance",
        4,
        "Maximum shortest atom-graph distance between reaction instances' matched atom sets, in bonds. "
        "Overlapping instances have distance zero; farther or disconnected pairs use the overflow bucket.",
    ),
}
