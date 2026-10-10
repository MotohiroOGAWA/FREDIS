import torch.nn as nn

def build_fc_layers(input_dim: int, fc_dims: list, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
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
    layers = []
    in_dim = input_dim
    for dim in fc_dims:
        layers.append(nn.Linear(in_dim, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)