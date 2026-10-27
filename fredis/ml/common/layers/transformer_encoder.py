"""Post-LayerNorm Transformer encoder over padded token sets (batch_first).

No positional encoding is added: tokens are treated as an unordered set, so the
encoder is permutation-equivariant over the valid tokens of each batch row.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .feed_forward import FeedForwardBlock
from .multihead_attention import MultiheadAttention


def check_attention_rows(
    allowed: Tensor, query_padding_mask: Optional[Tensor], name: str
) -> None:
    """Reject valid query rows whose keys are all masked.

    ``allowed`` is bool [B, Tq, Tk] (True = attendable). Callers remove invalid
    candidates beforehand, so a fully masked valid query indicates a data error
    rather than something to silently zero.
    """
    empty = ~allowed.any(dim=-1)  # [B, Tq]
    if query_padding_mask is not None:
        empty = empty & ~query_padding_mask
    if bool(empty.any()):
        batch, query = (int(v) for v in empty.nonzero()[0])
        raise ValueError(
            f"{name}: query {query} of batch row {batch} has no attendable keys"
        )


class TransformerEncoderLayer(nn.Module):
    """x = LN(x + SelfAttn(x)); x = LN(x + FFN(x))."""

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
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardBlock(dim, ffn_mult * dim, dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,  # [B, T, D]
        *,
        key_padding_mask: Optional[Tensor] = None,  # [B, T] bool, True = padding
        attn_mask: Optional[Tensor] = None,  # [B, T, T] bool, True = allowed
        need_weights: bool = False,
        relative_features=None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        h, weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            need_head_weights=need_weights,
            relative_features=relative_features,
        )
        x = self.self_norm(x + self.self_drop(h))
        x = self.ffn_norm(x + self.ffn_drop(self.ffn(x)))
        return x, weights


class TransformerEncoder(nn.Module):
    """Stack of post-LN self-attention layers; each residual sublayer ends in LayerNorm.

    Input:
        x: [B, T, D]
        key_padding_mask: [B, T] bool, True for padding tokens
        attn_mask: optional [B, T, T] bool, True where query may attend to key

    Output:
        out: [B, T, D]; padding positions are zero
        weights: None, or one [B, H, T, T] tensor per layer when need_weights
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
            TransformerEncoderLayer(
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
        *,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        relative_features=None,
    ) -> tuple[Tensor, Optional[list[Tensor]]]:
        if x.dim() != 3 or x.size(-1) != self.dim:
            raise ValueError(f"x must be [B, T, {self.dim}]")
        B, T, _ = x.shape
        if key_padding_mask is not None and key_padding_mask.shape != (B, T):
            raise ValueError(f"key_padding_mask must have shape {(B, T)}")
        if attn_mask is not None and attn_mask.shape != (B, T, T):
            raise ValueError(f"attn_mask must have shape {(B, T, T)}")

        allowed = torch.ones((B, T, T), dtype=torch.bool, device=x.device)
        if key_padding_mask is not None:
            allowed = allowed & ~key_padding_mask[:, None, :]
        if attn_mask is not None:
            allowed = allowed & attn_mask
        check_attention_rows(allowed, key_padding_mask, "TransformerEncoder")

        weights = [] if need_weights else None
        for layer in self.layers:
            x, w = layer(
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
                relative_features=relative_features,
            )
            if weights is not None:
                weights.append(w)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask[..., None], 0)
        return x, weights
