"""Sparse graph attention and outgoing transition log probabilities.

Based on the edge-aware attention and residual/FFN structure in
https://github.com/MotohiroOGAWA/Arms/blob/main/arms/layers/gat.py.
All graph operations use edge lists; no dense adjacency or per-node loops are
needed. Message-passing attention and transition probabilities are separate.

For a physical graph A->B, A->C, use reversed message edges to let the
parent read its children, while retaining the physical edges for scoring::

    encoder = GAT(node_dim, reaction_dim, num_layers=3, directed=True)
    head = GATTransitionHead(node_dim, reaction_dim, condition_dim=condition_dim)
    h, _ = encoder(node_h, reaction_index.flip(0), reaction_h)
    events, log_p = head(
        h, reaction_index, reaction_h, condition_h=condition_h
    )

With node order A, B, C, event order is A->B, A->C, A->A, B->B, C->C.
The probabilities for the three A events sum to one after exp().
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .feed_forward import FeedForwardBlock


def segment_log_softmax(logits: Tensor, groups: Tensor, num_groups: int) -> Tensor:
    """LogSoftmax over ragged groups, preserving input order.

    Equivalent to applying nn.LogSoftmax(dim=0) independently to each group's
    finite logits, without padding, sorting, or copying group sizes to the CPU.
    Accumulation uses float32 for float16/bfloat16 inputs. Empty groups have no
    output entries. The returned dtype matches logits.
    """
    if logits.ndim != 1 or groups.shape != logits.shape:
        raise ValueError("logits and groups must be one-dimensional and equally sized")
    if groups.dtype != torch.long or groups.device != logits.device:
        raise ValueError("groups must be int64 on the logits device")
    if num_groups < 0:
        raise ValueError("num_groups must be nonnegative")
    values = (
        logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    )
    maximum = values.new_full((num_groups,), -torch.inf)
    maximum.scatter_reduce_(
        0, groups, values.detach(), reduce="amax", include_self=True
    )
    shifted = values - maximum[groups]
    total = values.new_zeros(num_groups).index_add_(0, groups, shifted.exp())
    return (shifted - total[groups].log()).to(logits.dtype)


def _check_graph(
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None,
    node_dim: int,
    edge_dim: int,
) -> Tensor:
    if x.ndim != 2 or x.shape[1] != node_dim:
        raise ValueError(f"x must have shape [N, {node_dim}]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long or edge_index.device != x.device:
        raise ValueError("edge_index must be int64 on the x device")
    expected = (edge_index.shape[1], edge_dim)
    if edge_attr is None:
        if edge_dim:
            raise ValueError("edge_attr is required when edge_dim > 0")
        edge_attr = x.new_empty(expected)
    if edge_attr.shape != expected or edge_attr.device != x.device:
        raise ValueError(f"edge_attr must have shape {expected} on the x device")
    return edge_attr


class GATLayer(nn.Module):
    """Edge-aware additive attention over incoming edges of each destination.

    x: [N, in_features], edge_index: [2, E] (sender, receiver),
    edge_attr: [E, edge_dim]. With directed=False, reverse message edges are
    appended using the same attributes. Supply directed=True and explicit reverse
    edges to encode different direction attributes. Parallel edges are retained.
    softmax_per='dst' normalizes incoming edges per destination; 'graph'
    normalizes all message edges per graph. Optional batch [N] uses contiguous
    graph IDs as in PyG Batch; without it all nodes form one graph. Graph-mode
    attention is not a parent-wise transition probability. T is temperature.

    Returns (node_features, attention). Attention is None unless requested;
    otherwise it is (message_edge_index, weights), including appended self loops.
    These self loops update representations; they are not spectrum Stay events.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dim: int,
        directed: bool = False,
        softmax_per: str = "dst",
        T: float = 1.0,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if min(in_features, out_features) <= 0 or edge_dim < 0:
            raise ValueError(
                "Node dimensions must be positive and edge_dim nonnegative"
            )
        if softmax_per not in {"dst", "graph"}:
            raise ValueError("softmax_per must be 'dst' or 'graph'")
        if not math.isfinite(T) or T <= 0:
            raise ValueError("T must be finite and positive")
        self.directed = directed
        self.softmax_per = softmax_per
        self.T = T
        self.in_features = in_features
        self.edge_dim = edge_dim
        self.add_self_loops = add_self_loops
        self.node_projection = nn.Linear(in_features, out_features, bias=False)
        self.source_attention = nn.Parameter(torch.empty(out_features))
        self.target_attention = nn.Parameter(torch.empty(out_features))
        self.edge_attention = nn.Parameter(torch.empty(edge_dim))
        self.edge_projection = (
            nn.Linear(edge_dim, out_features, bias=False) if edge_dim else None
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.xavier_uniform_(self.source_attention.unsqueeze(0))
        nn.init.xavier_uniform_(self.target_attention.unsqueeze(0))
        if self.edge_dim:
            nn.init.xavier_uniform_(self.edge_attention.unsqueeze(0))
            nn.init.xavier_uniform_(self.edge_projection.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        batch: Tensor | None = None,
        need_weights: bool = False,
    ):
        edge_attr = _check_graph(
            x, edge_index, edge_attr, self.in_features, self.edge_dim
        )
        if batch is not None and (
            batch.shape != (x.shape[0],)
            or batch.dtype != torch.long
            or batch.device != x.device
        ):
            raise ValueError("batch must be int64 [N] on the x device")
        if not self.directed:
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
            edge_attr = torch.cat((edge_attr, edge_attr))
        if self.add_self_loops:
            # Replace input self loops with one feature-neutral loop per node.
            keep = edge_index[0] != edge_index[1]
            loops = torch.arange(x.shape[0], device=x.device)
            edge_index = torch.cat((edge_index[:, keep], loops.expand(2, -1)), dim=1)
            edge_attr = torch.cat(
                (edge_attr[keep], x.new_zeros((x.shape[0], self.edge_dim)))
            )
        source, target = edge_index
        h = self.node_projection(x)
        scores = (
            (h * self.source_attention).sum(-1)[source]
            + (h * self.target_attention).sum(-1)[target]
            + (edge_attr * self.edge_attention).sum(-1)
        )
        groups = target
        if self.softmax_per == "graph":
            groups = torch.zeros_like(target) if batch is None else batch[target]
        # A PyG-style batch uses contiguous graph IDs in [0, N), so reserving N
        # groups avoids a GPU-to-CPU batch.max().item() synchronization.
        weights = segment_log_softmax(
            self.leaky_relu(scores) / self.T, groups, x.shape[0]
        ).exp()
        messages = h[source]
        if self.edge_projection is not None:
            # Preserve reaction information even for a single incoming edge.
            messages = messages + self.edge_projection(edge_attr)
        out = h.new_zeros(h.shape).index_add_(
            0, target, weights[:, None].to(messages.dtype) * messages
        )
        return out, (edge_index, weights) if need_weights else None


class GATBlock(nn.Module):
    """GAT, residual/LayerNorm, then the shared FFN and residual/LayerNorm.

    Inputs are never mutated. Repeated blocks propagate information farther
    through the supplied message edges. Edge attributes remain available at
    every layer; this block updates node features only.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
        softmax_per: str = "dst",
        T: float = 1.0,
        directed: bool = True,
    ) -> None:
        super().__init__()
        if type(ffn_hidden_mult) is not int or ffn_hidden_mult <= 0:
            raise ValueError("ffn_hidden_mult must be a positive integer")
        self.gat = GATLayer(
            hidden_dim,
            hidden_dim,
            edge_dim,
            directed=directed,
            softmax_per=softmax_per,
            T=T,
            add_self_loops=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForwardBlock(
            hidden_dim, ffn_hidden_mult * hidden_dim, dropout
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        batch: Tensor | None = None,
        need_weights: bool = False,
    ):
        update, weights = self.gat(
            x, edge_index, edge_attr, batch=batch, need_weights=need_weights
        )
        h = self.norm1(x + self.dropout(update))
        h = self.norm2(h + self.dropout(self.feed_forward(h)))
        return h, weights


class GAT(nn.Module):
    """Repeated independent GATBlocks, each with attention, FFN and two norms.

    num_layers counts message-passing blocks, not reaction depths. Returns node
    features and optionally one sparse attention tuple per block. No dense
    graph padding or graph/node iteration is used; only layers are repeated.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
        softmax_per: str = "dst",
        T: float = 1.0,
        directed: bool = True,
    ) -> None:
        super().__init__()
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        self.layers = nn.ModuleList(
            GATBlock(
                hidden_dim, edge_dim, ffn_hidden_mult, dropout, softmax_per, T, directed
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        batch: Tensor | None = None,
        need_weights: bool = False,
    ):
        weights = [] if need_weights else None
        for layer in self.layers:
            x, attention = layer(
                x, edge_index, edge_attr, batch=batch, need_weights=need_weights
            )
            if need_weights:
                weights.append(attention)
        return x, weights


class GATTransitionHead(nn.Module):
    """Score physical parent->child transitions and one Stay event per node.

    Supply GAT-updated x, physical directed reaction edges (not message reverse
    edges), and optional per-node conditions [N, condition_dim]. Conditions may
    concatenate adduct, collision energy and root features at the caller.

    Returns (event_index, log_probability): the original E edges first, then
    N Stay edges. Log probabilities normalize over each source's outgoing events.
    A node with no children has Stay probability one. No Sink event is added.
    Attention self loops have no role in constructing these output events.
    """

    def __init__(self, node_dim: int, edge_dim: int, *, condition_dim: int = 0) -> None:
        super().__init__()
        if node_dim <= 0 or min(edge_dim, condition_dim) < 0:
            raise ValueError("Invalid node, edge or condition dimensions")
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.condition_dim = condition_dim
        self.stay_edge = nn.Parameter(torch.zeros(edge_dim))
        # An explicit type flag distinguishes Stay even when edge_dim is zero.
        self.score = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + condition_dim + 1, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        condition_h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        edge_attr = _check_graph(x, edge_index, edge_attr, self.node_dim, self.edge_dim)
        if condition_h is None:
            if self.condition_dim:
                raise ValueError("condition_h is required when condition_dim > 0")
            condition_h = x.new_empty((x.shape[0], 0))
        if (
            condition_h.shape != (x.shape[0], self.condition_dim)
            or condition_h.device != x.device
        ):
            raise ValueError(
                "condition_h must have shape [N, condition_dim] on the x device"
            )
        nodes = torch.arange(x.shape[0], device=x.device)
        event_index = torch.cat((edge_index, nodes.expand(2, -1)), dim=1)
        source, target = event_index
        reaction = torch.cat((edge_attr, self.stay_edge.expand(x.shape[0], -1)))
        stay = torch.cat((x.new_zeros(edge_index.shape[1]), x.new_ones(x.shape[0])))
        inputs = torch.cat(
            (x[source], x[target], reaction, condition_h[source], stay[:, None]), dim=1
        )
        logits = self.score(inputs).squeeze(-1)
        return event_index, segment_log_softmax(logits, source, x.shape[0])
