"""Independent GELU MLP slices for fragment event types or ion/shift conditions."""

from __future__ import annotations
import math
import torch
from torch import nn


class ConditionLinear(nn.Module):
    def __init__(self, conditions: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(conditions, output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(conditions, output_dim))
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -(input_dim**-0.5), input_dim**-0.5)

    def forward(self, values, condition):
        # Group events rather than copying a full weight matrix per event.
        output = values.new_zeros((values.shape[0], self.bias.shape[1]))
        for index in condition.unique().tolist():
            rows = (condition == index).nonzero().flatten()
            output[rows] = torch.nn.functional.linear(
                values[rows], self.weight[index], self.bias[index]
            )
        return output


class ConditionHead(nn.Module):
    def __init__(self, conditions, input_dim, hidden_dims, dropout):
        super().__init__()
        dims = (input_dim, *hidden_dims, 1)
        if any(type(d) is not int or d < 1 for d in dims):
            raise ValueError("Head MLP widths must be positive integers")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.layers = nn.ModuleList(
            ConditionLinear(conditions, a, b) for a, b in zip(dims, dims[1:])
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, condition):
        for layer in self.layers[:-1]:
            values = self.dropout(self.activation(layer(values, condition)))
        return self.layers[-1](values, condition).squeeze(-1)
