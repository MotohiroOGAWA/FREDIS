import math
import torch.nn as nn


def build_fc_layers(
    input_dim: int, fc_dims: list, output_dim: int, dropout: float = 0.0
) -> nn.Sequential:
    """
    Build a fully-connected (dense) layer stack.

    Args:
        input_dim (int): Input feature dimension.
        fc_dims (list): List of hidden layer dimensions.
        output_dim (int): Final output dimension.
        dropout (float): Dropout rate applied after each hidden layer.

    Returns:
        nn.Sequential: Fully connected layers as a sequential module.
    """
    if any(
        type(dim) is not int or dim < 1 for dim in (input_dim, output_dim, *fc_dims)
    ):
        raise ValueError("MLP dimensions must be positive integers")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("MLP dropout must be finite and in [0, 1)")
    layers = []
    in_dim = input_dim
    for dim in fc_dims:
        layers.append(nn.Linear(in_dim, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)
