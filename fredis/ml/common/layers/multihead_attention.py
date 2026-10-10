import math
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """
    PyTorch-only Multi-head attention (fairseq-like).

    Shapes (batch_first=True):
      - query: [B, Tq, C]
      - key:   [B, Tk, C]
      - value: [B, Tk, C]

    Shapes (batch_first=False):
      - query: [Tq, B, C]
      - key:   [Tk, B, C]
      - value: [Tk, B, C]

    Masks:
      - attn_bias:
          [B, H, Tq, Tk] or [B*H, Tq, Tk]
      - attn_mask:
          bool [B, Tq, Tk]  (True=keep, False=mask out)
      - key_padding_mask:
          [B, Tk] (True indicates padding -> masked out)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    @staticmethod
    def _bool_to_additive_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
        # True = keep (0), False = mask out (-inf)
        out = torch.zeros_like(mask, dtype=dtype)
        out.masked_fill_(~mask, float("-inf"))
        return out

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attn_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # ---- normalize to [T,B,C] ----
        if self.batch_first:
            # [B,T,C] -> [T,B,C]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tq, bsz, c = query.shape
        tk, bsz2, c2 = key.shape

        if c != self.embed_dim or c2 != self.embed_dim or bsz2 != bsz:
            raise ValueError("query/key/value shape mismatch")

        # ---- projections ----
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        # ---- reshape to [B*H, T, Dh] ----
        q = q.view(tq, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(tk, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(tk, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # ---- attention logits [B*H, Tq, Tk] ----
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        # ---- attn_bias ----
        if attn_bias is not None:
            if attn_bias.dim() == 4:
                # [B,H,Tq,Tk] -> [B*H,Tq,Tk]
                attn_bias = attn_bias.reshape(bsz * self.num_heads, tq, tk)
            attn_weights = attn_weights + attn_bias

        # ---- attn_mask: bool [B,Tq,Tk] ----
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be bool [B,Tq,Tk]")
            additive = self._bool_to_additive_mask(attn_mask, attn_weights.dtype)
            additive = additive.unsqueeze(1).expand(bsz, self.num_heads, tq, tk)
            attn_weights = attn_weights.view(bsz, self.num_heads, tq, tk)
            attn_weights = attn_weights + additive
            attn_weights = attn_weights.view(bsz * self.num_heads, tq, tk)

        # ---- key_padding_mask: [B,Tk] ----
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tq, tk)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tq, tk)

        # ---- softmax ----
        # Fully masked query rows occur for padded tokens when pairwise masks are
        # used. softmax([-inf, ...]) produces NaN, so keep those rows at zero.
        valid_query = torch.isfinite(attn_weights).any(dim=-1, keepdim=True)
        safe_attn_weights = torch.where(
            valid_query,
            attn_weights,
            torch.zeros_like(attn_weights),
        )
        attn_probs = torch.softmax(safe_attn_weights, dim=-1)
        attn_probs = torch.where(valid_query, attn_probs, torch.zeros_like(attn_probs))
        attn_probs = self.dropout(attn_probs)
        attn = torch.bmm(attn_probs, v)

        # ---- back to [T,B,C] ----
        attn = attn.transpose(0, 1).contiguous().view(tq, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        # ---- output weights ----
        attn_weights_out = None
        if need_weights:
            w = attn_probs.view(bsz, self.num_heads, tq, tk)
            attn_weights_out = w if need_head_weights else w.mean(dim=1)

        # ---- restore batch_first ----
        if self.batch_first:
            attn = attn.transpose(0, 1)
            if attn_weights_out is not None and not need_head_weights:
                # [B,Tq,Tk] already correct
                pass

        return attn, attn_weights_out


class SelfAttention(nn.Module):
    """
    Self-attention wrapper over MultiheadAttention.

    Input/Output:
      - x: [T, B, C] -> out: [T, B, C]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.mha = MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)

    def forward(
        self,
        x: Tensor,                               # [T,B,C]
        *,
        attn_bias: Optional[Tensor] = None,       # [B*H,T,T] or None
        key_padding_mask: Optional[Tensor] = None,# [B,T] or None
        attn_mask: Optional[Tensor] = None,       # bool/additive [T,T] or broadcastable
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.mha(
            query=x,
            key=x,
            value=x,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            need_head_weights=need_head_weights,
        )

class CrossAttention(nn.Module):
    """
    Cross-attention wrapper over MultiheadAttention.

    Inputs:
      - q:  [Tq, B, C]
      - kv: [Tk, B, C]   (key=value=kv by default)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.mha = MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)

    def forward(
        self,
        q: Tensor,                               # [Tq,B,C]
        kv: Tensor,                              # [Tk,B,C]
        *,
        attn_bias: Optional[Tensor] = None,       # [B*H,Tq,Tk] or None
        key_padding_mask: Optional[Tensor] = None,# [B,Tk] or None
        attn_mask: Optional[Tensor] = None,       # bool/additive [Tq,Tk] or broadcastable
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.mha(
            query=q,
            key=kv,
            value=kv,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            need_head_weights=need_head_weights,
        )



# ------------------------------------------------------------
# __main__ quick test
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    T, B, C = 6, 1, 32
    H = 4

    x = torch.randn(T, B, C)

    # Example: attn_bias = [B*H, T, T]
    attn_bias = torch.randn(B * H, T, T) * 0.01

    # Example: additive mask (disallow last position attending to first two)
    attn_mask = torch.zeros(T, T)
    attn_mask[:, :2] = float("-inf")

    # Example: key_padding_mask (mask out last token)
    key_padding_mask = torch.zeros(B, T, dtype=torch.bool)
    key_padding_mask[:, -1] = True

    mha = MultiheadAttention(embed_dim=C, num_heads=H, dropout=0.1)

    y, w = mha(
        query=x,
        key=x,
        value=x,
        attn_bias=attn_bias,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        attn_mask=attn_mask,
        need_head_weights=False,
    )

    loss = y.mean()
    loss.backward()

    print("y:", tuple(y.shape))         # [T,B,C]
    print("w:", None if w is None else tuple(w.shape))  # [B,T,T]
    print("loss:", float(loss.item()))