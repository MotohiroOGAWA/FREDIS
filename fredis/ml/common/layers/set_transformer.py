from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================
# Set Transformer (generic, batch_first)
# - Supports variable-sized sets via padding + key_padding_mask
# ============================================================

class MAB(nn.Module):
    """
    Multihead Attention Block (Set Transformer core primitive).

    Args:
        dim_q: query dimension
        dim_kv: key/value dimension
        dim_out: output dimension (also the internal attention dimension)
        num_heads: attention heads
        dropout: dropout probability
        ln: whether to use LayerNorm

    Input:
        Q: [B, Lq, dim_q]
        K: [B, Lk, dim_kv]
        key_padding_mask: [B, Lk] bool, True for PAD positions in K/V

    Output:
        H: [B, Lq, dim_out]
    """
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_out: int,
        *,
        num_heads: int,
        dropout: float = 0.0,
        ln: bool = True,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.dim_out = dim_out

        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_kv, dim_out)
        self.v_proj = nn.Linear(dim_kv, dim_out)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_out,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln0 = nn.LayerNorm(dim_out) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_out) if ln else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim_out, ffn_mult * dim_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim_out, dim_out),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(K)

        if key_padding_mask is None:
            h_attn, _ = self.attn(
                query=q,
                key=k,
                value=v,
                need_weights=False,
            )
        else:
            if key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must be [B, Lk]")

            B = int(Q.size(0))
            if int(key_padding_mask.size(0)) != B or int(key_padding_mask.size(1)) != int(K.size(1)):
                raise ValueError(
                    f"key_padding_mask shape mismatch: got {tuple(key_padding_mask.shape)}, "
                    f"expected ({B}, {int(K.size(1))})"
                )

            all_pad = key_padding_mask.all(dim=1)  # [B]
            h_attn = torch.zeros_like(q)

            valid_rows = ~all_pad
            if valid_rows.any():
                h_valid, _ = self.attn(
                    query=q[valid_rows],
                    key=k[valid_rows],
                    value=v[valid_rows],
                    key_padding_mask=key_padding_mask[valid_rows],
                    need_weights=False,
                )
                h_attn[valid_rows] = h_valid

        h = self.ln0(h_attn + q)
        h2 = self.ffn(h)
        h = self.ln1(h + h2)
        return h


class SAB(nn.Module):
    """
    Self-Attention Block: MAB(X, X)
    """
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        dropout: float = 0.0,
        ln: bool = True,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.mab = MAB(dim, dim, dim, num_heads=num_heads, dropout=dropout, ln=ln, ffn_mult=ffn_mult)

    def forward(self, X: Tensor, *, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.mab(X, X, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block (reduces attention cost).
    Uses m inducing points I.

    Steps:
      H = MAB(I, X)
      Y = MAB(X, H)

    Input:
      X: [B, L, D]
      key_padding_mask: [B, L] bool (True for PAD in X)
    Output:
      Y: [B, L, D]
    """
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        num_inducing_points: int,
        dropout: float = 0.0,
        ln: bool = True,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing_points, dim) * 0.02)
        self.mab0 = MAB(dim, dim, dim, num_heads=num_heads, dropout=dropout, ln=ln, ffn_mult=ffn_mult)
        self.mab1 = MAB(dim, dim, dim, num_heads=num_heads, dropout=dropout, ln=ln, ffn_mult=ffn_mult)

    def forward(self, X: Tensor, *, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B = int(X.size(0))
        I = self.I.expand(B, -1, -1)  # [B, m, D]
        H = self.mab0(I, X, key_padding_mask=key_padding_mask)   # mask applies to K/V=X
        Y = self.mab1(X, H, key_padding_mask=None)               # H has no padding
        return Y


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.

    Uses k seed vectors S to attend over the set X.

    Input:
      X: [B, L, D]
      key_padding_mask: [B, L] bool (True for PAD in X)
    Output:
      P: [B, k, D]
    """
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
        ln: bool = True,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, dim, dim, num_heads=num_heads, dropout=dropout, ln=ln, ffn_mult=ffn_mult)

    def forward(self, X: Tensor, *, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B = int(X.size(0))
        S = self.S.expand(B, -1, -1)  # [B, k, D]
        return self.mab(S, X, key_padding_mask=key_padding_mask)


class SetTransformer(nn.Module):
    """
    Generic Set Transformer encoder + PMA pooling.

    Configure either SAB or ISAB layers.

    Input:
      X: [B, L, D] (padded)
      key_padding_mask: [B, L] bool (True for PAD)
    Output:
      out: [B, out_dim] (if num_seeds==1) or [B, num_seeds, out_dim] (if keep_seeds=True)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        block_type: Literal["sab", "isab"] = "sab",
        num_inducing_points: int = 16,
        num_seeds: int = 1,
        dropout: float = 0.0,
        ln: bool = True,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()

        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        blocks = []
        for _ in range(num_layers):
            if block_type == "sab":
                blocks.append(SAB(hidden_dim, num_heads=num_heads, dropout=dropout, ln=ln, ffn_mult=ffn_mult))
            elif block_type == "isab":
                blocks.append(ISAB(
                    hidden_dim,
                    num_heads=num_heads,
                    num_inducing_points=num_inducing_points,
                    dropout=dropout,
                    ln=ln,
                    ffn_mult=ffn_mult,
                ))
            else:
                raise ValueError(f"Unknown block_type='{block_type}'")
        self.blocks = nn.ModuleList(blocks)

        self.pma = PMA(hidden_dim, num_heads=num_heads, num_seeds=num_seeds, dropout=dropout, ln=ln, ffn_mult=ffn_mult)
        self.out_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        self.num_seeds = num_seeds

    def forward(
        self,
        X: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
        keep_seeds: bool = False,
    ) -> Tensor:
        H = self.in_proj(X)

        for blk in self.blocks:
            H = blk(H, key_padding_mask=key_padding_mask)

        P = self.pma(H, key_padding_mask=key_padding_mask)  # [B, k, Dh]
        P = self.out_proj(P)                                # [B, k, Dout]

        if keep_seeds or self.num_seeds != 1:
            return P
        return P[:, 0, :]  # [B, Dout]


    # ============================================================
    # Helpers: pack ragged sets (group_id) -> padded [G, Lmax, D]
    # ============================================================
    @staticmethod
    @dataclass(frozen=True)
    class PackedSets:
        """
        Padded sets.

        X: [G, Lmax, D]
        key_padding_mask: [G, Lmax] bool (True for PAD)
        counts: [G] long
        """
        X: Tensor
        key_padding_mask: Tensor
        counts: Tensor

    @staticmethod
    def pack_sets_by_group(
        feats: Tensor,          # [M, D]
        group_id: Tensor,       # [M] long in [0..G-1]
        num_groups: int,
    ) -> SetTransformer.PackedSets:
        """
        Pack ragged sets into a padded batch.

        This is fully vectorized (no Python loop over groups).
        """
        if feats.dim() != 2:
            raise ValueError("feats must be [M, D]")
        if group_id.dim() != 1 or group_id.dtype != torch.long:
            raise ValueError("group_id must be 1D torch.long")

        M, D = int(feats.size(0)), int(feats.size(1))
        G = int(num_groups)

        if M == 0:
            X = torch.zeros((G, 0, D), dtype=feats.dtype, device=feats.device)
            mask = torch.zeros((G, 0), dtype=torch.bool, device=feats.device)
            counts = torch.zeros((G,), dtype=torch.long, device=feats.device)
            return SetTransformer.PackedSets(X=X, key_padding_mask=mask, counts=counts)

        if int(group_id.min().item()) < 0 or int(group_id.max().item()) >= G:
            raise ValueError(f"group_id must be in [0, {G})")

        # Sort by group_id so each group's elements are contiguous
        order = torch.argsort(group_id)
        gid = group_id[order]  # [M]
        x = feats[order]       # [M, D]

        counts = torch.bincount(gid, minlength=G)  # [G]
        Lmax = int(counts.max().item()) if G > 0 else 0

        if Lmax == 0:
            X = torch.zeros((G, 0, D), dtype=feats.dtype, device=feats.device)
            mask = torch.zeros((G, 0), dtype=torch.bool, device=feats.device)
            return SetTransformer.PackedSets(X=X, key_padding_mask=mask, counts=counts)

        # Compute position within each group for each sorted element
        # group_starts[g] = starting index in the sorted array for group g
        group_starts = torch.cumsum(counts, dim=0) - counts                  # [G]
        starts_for_each_elem = group_starts[gid]                              # [M]
        pos_in_group = torch.arange(M, device=feats.device) - starts_for_each_elem  # [M], 0..count-1 per group

        # Scatter into padded tensor
        X = torch.zeros((G, Lmax, D), dtype=feats.dtype, device=feats.device)
        X[gid, pos_in_group] = x

        # Build padding mask: True indicates padding (to be ignored)
        # mask_valid[g, j] = j < counts[g]
        j = torch.arange(Lmax, device=feats.device).unsqueeze(0)  # [1, Lmax]
        valid = j < counts.unsqueeze(1)                           # [G, Lmax]
        key_padding_mask = ~valid                                 # True for PAD

        return SetTransformer.PackedSets(X=X, key_padding_mask=key_padding_mask, counts=counts)

