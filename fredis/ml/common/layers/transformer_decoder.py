"""Post-LayerNorm Transformer decoder over padded token sets (batch_first).

Each layer applies self-attention among query tokens, cross-attention from the
query tokens to an encoded memory, then a feed-forward block. No positional
encoding and no causal mask are used: query tokens form an unordered set and
the decoder is permutation-equivariant over queries and permutation-invariant
over memory tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .feed_forward import FeedForwardBlock
from .multihead_attention import MultiheadAttention
from .transformer_encoder import check_attention_rows


@dataclass(frozen=True)
class TransformerDecoderWeights:
    # One tensor per layer.
    self_attention: list[Tensor]  # [B, H, Tq, Tq]
    cross_attention: list[Tensor]  # [B, H, Tq, Tk]


class TransformerDecoderLayer(nn.Module):
    """Post-LN self-attention, cross-attention, and FFN residual sublayers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_drop = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardBlock(dim, ffn_mult * dim, dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,  # [B, Tq, D]
        memory: Tensor,  # [B, Tk, D]
        *,
        query_padding_mask: Optional[Tensor] = None,  # [B, Tq] True = padding
        memory_key_padding_mask: Optional[Tensor] = None,  # [B, Tk] True = padding
        self_attn_mask: Optional[Tensor] = None,  # [B, Tq, Tq] True = allowed
        memory_mask: Optional[Tensor] = None,  # [B, Tq, Tk] True = allowed
        need_weights: bool = False,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        h, self_weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=query_padding_mask,
            attn_mask=self_attn_mask,
            need_weights=need_weights,
            need_head_weights=need_weights,
        )
        x = self.self_norm(x + self.self_drop(h))

        h, cross_weights = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_mask,
            need_weights=need_weights,
            need_head_weights=need_weights,
        )
        x = self.cross_norm(x + self.cross_drop(h))
        x = self.ffn_norm(x + self.ffn_drop(self.ffn(x)))
        return x, self_weights, cross_weights


class TransformerDecoder(nn.Module):
    """Stack of post-LN decoder layers; each residual sublayer ends in LayerNorm.

    Input:
        x: query tokens [B, Tq, D]
        memory: encoder output [B, Tk, D]
        query_padding_mask: [B, Tq] bool, True for padding queries
        memory_key_padding_mask: [B, Tk] bool, True for padding memory tokens
        self_attn_mask: optional [B, Tq, Tq] bool, True where allowed
        memory_mask: optional [B, Tq, Tk] bool, True where a query may attend to
            a memory token (e.g. excluding candidates invalid for this state)

    Output:
        out: [B, Tq, D]; padding queries are zero
        weights: None, or TransformerDecoderWeights when need_weights
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        *,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(
                dim,
                num_heads,
                dropout=dropout,
                ffn_mult=ffn_mult,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        *,
        query_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Optional[TransformerDecoderWeights]]:
        if x.dim() != 3 or x.size(-1) != self.dim:
            raise ValueError(f"x must be [B, Tq, {self.dim}]")
        if memory.dim() != 3 or memory.size(-1) != self.dim:
            raise ValueError(f"memory must be [B, Tk, {self.dim}]")
        B, Tq, _ = x.shape
        Tk = memory.size(1)
        if memory.size(0) != B:
            raise ValueError("x and memory batch sizes differ")
        for name, mask, shape in (
            ("query_padding_mask", query_padding_mask, (B, Tq)),
            ("memory_key_padding_mask", memory_key_padding_mask, (B, Tk)),
            ("self_attn_mask", self_attn_mask, (B, Tq, Tq)),
            ("memory_mask", memory_mask, (B, Tq, Tk)),
        ):
            if mask is not None and mask.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

        allowed_self = torch.ones((B, Tq, Tq), dtype=torch.bool, device=x.device)
        if query_padding_mask is not None:
            allowed_self = allowed_self & ~query_padding_mask[:, None, :]
        if self_attn_mask is not None:
            allowed_self = allowed_self & self_attn_mask
        check_attention_rows(
            allowed_self, query_padding_mask, "TransformerDecoder self-attention"
        )
        allowed_cross = torch.ones((B, Tq, Tk), dtype=torch.bool, device=x.device)
        if memory_key_padding_mask is not None:
            allowed_cross = allowed_cross & ~memory_key_padding_mask[:, None, :]
        if memory_mask is not None:
            allowed_cross = allowed_cross & memory_mask
        check_attention_rows(
            allowed_cross, query_padding_mask, "TransformerDecoder cross-attention"
        )

        self_weights: list[Tensor] = []
        cross_weights: list[Tensor] = []
        for layer in self.layers:
            x, s, c = layer(
                x,
                memory,
                query_padding_mask=query_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                self_attn_mask=self_attn_mask,
                memory_mask=memory_mask,
                need_weights=need_weights,
            )
            if need_weights:
                self_weights.append(s)
                cross_weights.append(c)
        if query_padding_mask is not None:
            x = x.masked_fill(query_padding_mask[..., None], 0)
        weights = (
            TransformerDecoderWeights(self_weights, cross_weights)
            if need_weights
            else None
        )
        return x, weights
