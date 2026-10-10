from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj

from .multihead_attention import *

@dataclass
class PaddedGraphBatch:
    """A bucketed mini-batch with padded graph tensors."""
    cap: int
    batch: Batch  # original PyG Batch (graph list)

    x: torch.Tensor             # [B, cap, d]
    node_ids: torch.Tensor      # [B, cap] long, padded positions = -1
    node_mask: torch.Tensor     # [B, cap] bool
    num_nodes: torch.Tensor     # [B] long

    edge_index: torch.Tensor    # [B, 2, Emax]
    edge_attr: torch.Tensor     # [B, Emax, edge_dim] (edge_dim may be 0)
    edge_mask: torch.Tensor     # [B, Emax] bool
    num_edges: torch.Tensor     # [B] long

    def __post_init__(self):
        B, cap, d = self.x.size()
        assert cap == self.cap
        assert self.node_ids.size() == (B, cap)
        assert self.node_mask.size() == (B, cap)
        assert self.num_nodes.size() == (B,)

        assert self.edge_index.size(0) == B
        assert self.edge_index.size(1) == 2
        assert self.edge_mask.size(0) == B
        assert self.edge_index.size(2) == self.edge_mask.size(1)
        assert self.edge_attr.size(0) == B
        assert self.edge_attr.size(1) == self.edge_mask.size(1)
        assert self.num_edges.size() == (B,)

    @property
    def B(self) -> int:
        return int(self.x.size(0))


    def build_adj(
        self,
        *,
        undirected: bool = True,
        add_self_loops: bool = False,
    ) -> torch.Tensor:
        """
        Build adjacency matrix from this batch.

        Returns:
          adj: [B, cap, cap] bool
        """
        B = self.edge_index.size(0)
        device = self.edge_index.device

        adj = torch.zeros((B, self.cap, self.cap), device=device, dtype=torch.bool)

        # No edges -> only optional self-loops
        if self.edge_index.numel() == 0 or self.edge_mask.numel() == 0:
            if add_self_loops:
                eye = torch.eye(self.cap, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
                adj |= eye
            # Mask padded nodes
            pair = self.node_mask[:, :, None] & self.node_mask[:, None, :]
            return adj & pair

        src = self.edge_index[:, 0, :]  # [B, Emax]
        dst = self.edge_index[:, 1, :]  # [B, Emax]
        m = self.edge_mask              # [B, Emax]

        # Flatten valid edges into (b, u, v)
        b_ids = torch.arange(B, device=device).unsqueeze(1).expand_as(src)  # [B, Emax]
        bf = b_ids[m]   # [M]
        uf = src[m]     # [M]
        vf = dst[m]     # [M]

        # Safety: drop edges touching padded nodes
        ok = self.node_mask[bf, uf] & self.node_mask[bf, vf]
        bf, uf, vf = bf[ok], uf[ok], vf[ok]

        adj[bf, uf, vf] = True
        if undirected:
            adj[bf, vf, uf] = True

        if add_self_loops:
            eye = torch.eye(self.cap, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
            adj |= eye

        # Mask padded nodes (final)
        pair = self.node_mask[:, :, None] & self.node_mask[:, None, :]
        return adj & pair

    @staticmethod
    def from_graph_batch(batch: Batch, cap: int) -> "PaddedGraphBatch":
        """
        Convert PyG Batch -> PaddedGraphBatch (single bucket with fixed cap).
        - No Python loop over graphs for node packing.
        - No Python loop over graphs for edge packing (uses sort + scatter).
        Assumes all graphs satisfy num_nodes <= cap.
        """
        if batch.x is None or batch.edge_index is None:
            raise ValueError("batch must have x and edge_index")

        device = batch.x.device
        B = int(batch.num_graphs)
        N = int(batch.x.size(0))
        d = int(batch.x.size(1))

        # ---- nodes: num_nodes, local positions, pack x ----
        # ptr: [B+1], nodes of graph g are [ptr[g], ptr[g+1])
        ptr = batch.ptr
        num_nodes = (ptr[1:] - ptr[:-1]).to(torch.long)  # [B]
        if (num_nodes > cap).any():
            raise ValueError("Some graphs exceed cap")

        # graph id per node: [N]
        node_g = batch.batch  # [N]
        # local index per node within its graph: local = global - ptr[g]
        node_local = torch.arange(N, device=device, dtype=torch.long) - ptr[node_g]
        # pack
        x_pad = torch.zeros((B, cap, d), device=device, dtype=batch.x.dtype)
        node_mask = torch.zeros((B, cap), device=device, dtype=torch.bool)
        x_pad[node_g, node_local] = batch.x
        node_mask[node_g, node_local] = True

        # node_ids_pad[b, i] = global node index (row in batch.x) for the node placed at (b, i)
        # padded positions remain -1
        node_ids = torch.full((B, cap), -1, device=device, dtype=torch.long)
        node_ids[node_g, node_local] = torch.arange(N, device=device, dtype=torch.long)

        # ---- edges: make per-edge graph id, local src/dst, compute position in each graph ----
        src, dst = batch.edge_index  # [E]
        E = int(src.numel())
        if E == 0:
            # edge_dim handling
            edge_dim = 0
            if getattr(batch, "edge_attr", None) is not None:
                edge_dim = int(batch.edge_attr.size(-1))
            edge_index = torch.zeros((B, 2, 0), device=device, dtype=torch.long)
            edge_attr = torch.zeros((B, 0, edge_dim), device=device, dtype=batch.x.dtype)
            edge_mask = torch.zeros((B, 0), device=device, dtype=torch.bool)
            num_edges = torch.zeros((B,), device=device, dtype=torch.long)
            return PaddedGraphBatch(
                cap=cap,
                batch=batch,
                x=x_pad,
                node_ids=node_ids,
                node_mask=node_mask,
                num_nodes=num_nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_mask=edge_mask,
                num_edges=num_edges,
            )

        # graph id per edge (Batchでは cross-graph edge は基本ないが安全のため src側で決める)
        edge_g = node_g[src]  # [E]

        # local src/dst in each graph
        src_l = src - ptr[edge_g]
        dst_l = dst - ptr[edge_g]

        # optional edge_attr
        ea = getattr(batch, "edge_attr", None)
        if ea is None:
            edge_dim = 0
            ea = torch.empty((E, 0), device=device, dtype=batch.x.dtype)
        else:
            if ea.size(0) != E:
                raise ValueError("batch.edge_attr must have shape [E, edge_dim]")
            edge_dim = int(ea.size(1))

        # count edges per graph -> [B]
        num_edges = torch.bincount(edge_g, minlength=B).to(torch.long)
        Emax = int(num_edges.max().item())

        # position within each graph: sort edges by graph id, then take within-group index
        perm = torch.argsort(edge_g)  # [E]
        edge_g_s = edge_g[perm]
        src_l_s = src_l[perm]
        dst_l_s = dst_l[perm]
        ea_s = ea[perm]

        # group start index for each graph in the sorted edge list
        # start[g] = cumulative edges before g
        start = torch.cumsum(num_edges, dim=0) - num_edges  # [B]
        # for each edge in sorted list, subtract its group's start to get 0..E_g-1
        pos_in_g = torch.arange(E, device=device, dtype=torch.long) - start[edge_g_s]  # [E]

        # allocate padded edge tensors
        edge_index = torch.zeros((B, 2, Emax), device=device, dtype=torch.long)
        edge_attr = torch.zeros((B, Emax, edge_dim), device=device, dtype=ea.dtype)
        edge_mask = torch.zeros((B, Emax), device=device, dtype=torch.bool)

        # scatter into padded buffers
        edge_index[edge_g_s, 0, pos_in_g] = src_l_s
        edge_index[edge_g_s, 1, pos_in_g] = dst_l_s
        if edge_dim > 0:
            edge_attr[edge_g_s, pos_in_g] = ea_s
        edge_mask[edge_g_s, pos_in_g] = True

        return PaddedGraphBatch(
            cap=cap,
            batch=batch,
            x=x_pad,
            node_ids=node_ids,
            node_mask=node_mask,
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_mask=edge_mask,
            num_edges=num_edges,
        )
    
    @staticmethod
    def bucketize_to_padded_graph_batches(
        graphs: Union[Data, Batch, Sequence[Data]],
        *,
        start_cap: int = 64,
        growth: float = 2.0,
    ) -> Tuple[List["PaddedGraphBatch"], List[torch.Tensor]]:
        """
        Loop-free (no for / while in cap generation) bucketization.

        Accepts:
          - Data
          - Batch
          - Sequence[Data]
        """

        # ------------------------------------------------------------
        # 0) normalize input -> List[Data]
        # ------------------------------------------------------------
        if isinstance(graphs, Batch):
            graphs = graphs.to_data_list()
        elif isinstance(graphs, Data):
            graphs = [graphs]
        else:
            graphs = list(graphs)

        if len(graphs) == 0:
            return [], []

        if start_cap <= 0:
            raise ValueError("start_cap must be positive")
        if growth <= 1.0:
            raise ValueError("growth must be > 1.0")

        device = graphs[0].x.device

        # ------------------------------------------------------------
        # 1) num_nodes for all graphs
        # ------------------------------------------------------------
        num_nodes = torch.tensor(
            [int(getattr(g, "num_nodes", None) or g.x.size(0)) for g in graphs],
            device=device,
            dtype=torch.long,
        )

        # graphs: List[Data]
        node_offsets = torch.cat((torch.tensor([0], device=device), num_nodes[:-1])).cumsum(dim=0)  # [G], node_offsets[g] = global start node index of graph g

        max_nodes = int(num_nodes.max().item())

        # ------------------------------------------------------------
        # 2) generate caps (NO while / for)
        # ------------------------------------------------------------
        # K = ceil(log(max_nodes / start_cap) / log(growth))
        K = torch.ceil(
            torch.log(torch.tensor(max_nodes / start_cap, device=device, dtype=torch.float))
            / torch.log(torch.tensor(growth, device=device))
        ).clamp(min=0).to(torch.long)

        exponents = torch.arange(K + 1, device=device, dtype=torch.float)
        caps = torch.ceil(
            start_cap * torch.pow(torch.tensor(growth, device=device), exponents)
        ).to(torch.long)

        # ensure last cap >= max_nodes
        caps[-1] = torch.maximum(caps[-1], torch.tensor(max_nodes, device=device))

        # ------------------------------------------------------------
        # 3) assign each graph to smallest fitting cap (vectorized)
        # ------------------------------------------------------------
        # num_nodes: [G]
        # caps:      [C]
        fits = num_nodes[:, None] <= caps[None, :]      # [G, C]
        bucket_id = fits.float().argmax(dim=1)          # [G]

        # ------------------------------------------------------------
        # 4) build PaddedGraphBatch list
        # ------------------------------------------------------------
        out: List[PaddedGraphBatch] = []
        ids: List[torch.Tensor] = []

        for i, cap in enumerate(caps.tolist()):
            mask = bucket_id == i
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=False).view(-1)
            graphs_i = [graphs[j] for j in idx]

            pyg_batch = Batch.from_data_list(graphs_i)
            bucket = PaddedGraphBatch.from_graph_batch(pyg_batch, cap=int(cap))

            # idx: this bucket's original graph indices in the input `graphs` list (same order as graphs_i used to build pyg_batch)
            idx_t = idx.to(torch.long)  # [B] (B = num graphs in this bucket)

            # base offset per graph in this bucket: [B]
            base = node_offsets[idx_t]  # [B]

            # bucket offsets inside pyg_batch concatenation
            bucket_ptr = pyg_batch.ptr[:-1].to(device)  # [B]  (start index of each graph inside pyg_batch.x)

            node_ids = bucket.node_ids  # [B, cap] indices in pyg_batch.x (already prefix-summed), pad=-1
            valid = node_ids >= 0

            # Convert: (pyg_batch global) -> (original global)
            bucket.node_ids = torch.where(
                valid,
                node_ids - bucket_ptr[:, None] + base[:, None],
                node_ids,  # keep -1
            )

            out.append(bucket)
            ids.append(idx_t)
            
        return out, ids
    


class CentralityEncoding(nn.Module):
    """Centrality encoding for PaddedGraphBatch via learnable degree embeddings."""

    def __init__(
        self,
        dim: int,
        max_degree: int,
        *,
        undirected: bool = False,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.undirected = undirected

        self.in_deg_emb = nn.Embedding(max_degree + 1, dim)
        self.out_deg_emb = nn.Embedding(max_degree + 1, dim)

    def forward(
        self,
        x: torch.Tensor,               # [B, cap, D]
        edge_index: torch.Tensor,      # [B, 2, Emax]
        edge_mask: torch.Tensor,       # [B, Emax] bool
        node_mask: torch.Tensor,       # [B, cap] bool
    ) -> torch.Tensor:
        B, cap, D = x.shape
        device = x.device

        # ---- degrees: [B, cap] ----
        out_deg = torch.zeros((B, cap), device=device, dtype=torch.long)
        in_deg  = torch.zeros((B, cap), device=device, dtype=torch.long)

        if edge_index.numel() > 0 and edge_mask.numel() > 0:
            src = edge_index[:, 0, :]   # [B, Emax]
            dst = edge_index[:, 1, :]   # [B, Emax]
            m = edge_mask.to(torch.long)

            ones = torch.ones_like(src, dtype=torch.long)

            out_deg.scatter_add_(dim=1, index=src, src=ones * m)
            in_deg.scatter_add_(dim=1, index=dst, src=ones * m)

        # mask padded nodes
        node_mask_l = node_mask.to(torch.long)
        out_deg = out_deg * node_mask_l
        in_deg  = in_deg  * node_mask_l

        # ---- undirected handling ----
        if self.undirected:
            deg = (out_deg + in_deg).clamp(max=self.max_degree)
            return x + self.in_deg_emb(deg) + self.out_deg_emb(deg)

        # ---- directed case ----
        out_deg = out_deg.clamp(max=self.max_degree)
        in_deg  = in_deg.clamp(max=self.max_degree)

        return x + self.in_deg_emb(in_deg) + self.out_deg_emb(out_deg)
    
class SpatialEncoding(nn.Module):
    """Distance embedding producing per-head additive attention bias (supports [..., N, N])."""

    def __init__(self, num_heads: int, max_dist: int):
        super().__init__()
        self.max_dist = int(max_dist)
        self.dist_emb = nn.Embedding(self.max_dist + 1, num_heads)

    def forward(self, spatial_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
          spatial_pos: [..., N, N] (integer distances)

        Returns:
          dist_bias:   [..., H, N, N]
        """
        # clamp distances into embedding range
        dist = spatial_pos.clamp(max=self.max_dist).long()          # [..., N, N]

        # embedding -> [..., N, N, H]
        bias = self.dist_emb(dist)

        # move last dim (H) to just before (N,N): [..., H, N, N]
        # (works for any number of prefix dims)
        return bias.movedim(-1, -3)

    @staticmethod
    def compute_spatial_pos(
        adj: torch.Tensor,                 # [..., N, N] bool
        max_dist: int,
        *,
        node_mask: Optional[torch.Tensor] = None,  # [..., N] bool
        add_self_loops: bool = True,
    ) -> torch.Tensor:
        """
        Shortest path distances up to max_dist using iterative reachability expansion.

        Args:
        adj          : [..., N, N] bool adjacency (can be directed or undirected)
        max_dist     : maximum distance to compute (unreachable -> max_dist)
        node_mask    : optional [..., N] bool mask (False for padded nodes)
        add_self_loops: whether to set dist[i,i]=0 only for valid nodes

        Returns:
        dist: [..., N, N] long
                dist=0 on diagonal (valid nodes),
                dist=1 for edges,
                dist=d (2..max_dist) for shortest distance,
                unreachable -> max_dist.
        """
        if adj.dtype != torch.bool:
            raise ValueError("adj must be bool tensor")
        if adj.dim() < 2:
            raise ValueError("adj must have shape [..., N, N]")
        if adj.size(-1) != adj.size(-2):
            raise ValueError("adj must be square on the last two dims")
        if max_dist < 1:
            raise ValueError("max_dist must be >= 1")

        *prefix, N, _ = adj.shape
        device = adj.device

        # Flatten prefix dims -> [B, N, N]
        if len(prefix) == 0:
            B = 1
            adj_b = adj.unsqueeze(0)
            node_mask_b = node_mask.unsqueeze(0) if node_mask is not None else None
            out_shape_prefix = ()
        else:
            B = int(torch.tensor(prefix).prod().item()) if len(prefix) > 1 else int(prefix[0])
            adj_b = adj.reshape(B, N, N)
            node_mask_b = node_mask.reshape(B, N) if node_mask is not None else None
            out_shape_prefix = tuple(prefix)

        # Valid node pair mask
        if node_mask_b is None:
            pair = torch.ones((B, N, N), device=device, dtype=torch.bool)
        else:
            pair = node_mask_b[:, :, None] & node_mask_b[:, None, :]

        # Clamp adjacency to valid pairs only
        adj_b = adj_b & pair

        inf = max_dist + 1
        dist = torch.full((B, N, N), inf, device=device, dtype=torch.long)

        if add_self_loops:
            eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # [1,N,N]
            dist[eye & pair] = 0
        else:
            # even without self-loops, we still set diagonal to 0 for valid nodes
            eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
            dist[eye & pair] = 0

        # d=1
        dist[adj_b] = 1

        reached = adj_b.clone()  # reached within <= current d (excluding d=0)

        # For boolean reachability by multiplication
        adj_f = adj_b.to(torch.float32)

        # d=2..max_dist
        for d in range(2, max_dist + 1):
            cand = torch.bmm(reached.to(torch.float32), adj_f) > 0.0  # [B,N,N] bool
            cand = cand & pair
            new = cand & (~reached) & (dist == inf)
            dist[new] = d
            reached = reached | cand

        dist = dist.clamp(max=max_dist)

        # Unflatten back to [..., N, N]
        if len(prefix) == 0:
            return dist.squeeze(0)  # [N,N]
        return dist.reshape(*out_shape_prefix, N, N)

class EdgeEncoding(nn.Module):
    """
    k-hop edge bias (up to max_dist) computed from shortest-path distances.

    This implementation matches your requested forward() signature:

      edge_index: [2,E] or [B,2,E]
      edge_attr : [E,D] or [B,E,D]
      spatial_pos: optional [N,N] or [B,N,N] (shortest-path distances)
      edge_mask : optional [B,E] (True for real edges)

    What it computes
    ----------------
    For each hop distance d = 1..max_dist:
      - build a (possibly directed) adjacency adj
      - build per-step edge value matrix EV_d where
          EV_d[u,v,:] = edge_attr(u,v) @ w_pos[d-1]   (only for 1-hop edges)
      - use a DP-style reachability expansion to get "first time reached at distance d"
        (or use spatial_pos==d if spatial_pos is provided)
      - assign edge_bias for those pairs using an average-of-edge-values over length-d walks:
          bias_d(i,j) = (SumEdgeValsOverWalks(i,j) / NumWalks(i,j)) / d

    Notes
    -----
    - If spatial_pos is provided, we use it as the "exact distance" mask (recommended).
    - If spatial_pos is not provided, we infer distances by iterative expansion on adj
      and fill pairs when they are first discovered at hop d.
    - This is vectorized over batch B; it avoids per-graph Python loops.
    """

    def __init__(self, edge_dim: int, num_heads: int, max_dist: int, *, undirected: bool = True):
        super().__init__()
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.undirected = undirected

        # w_pos[step]: [edge_dim, num_heads]
        self.w_pos = nn.Parameter(torch.empty(max_dist, edge_dim, num_heads))
        nn.init.xavier_uniform_(self.w_pos)

    @staticmethod
    def _infer_N_and_prefix(edge_index: torch.Tensor, spatial_pos: Optional[torch.Tensor]) -> tuple[int, tuple]:
        if spatial_pos is None:
            if edge_index.dim() == 2:
                N = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
                prefix = ()
            else:
                N = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
                prefix = (edge_index.size(0),)
        else:
            N = int(spatial_pos.size(-1))
            prefix = tuple(spatial_pos.shape[:-2])
        return N, prefix

    def _build_adj_and_ev(
        self,
        edge_index: torch.Tensor,   # [B,2,E]
        edge_attr: torch.Tensor,    # [B,E,D]
        edge_mask: torch.Tensor,    # [B,E] bool
        N: int,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build:
          adj: [B,N,N] bool
          EV : [B,N,N,H] (only direct edges carry values; others 0)
        """
        device = edge_attr.device
        dtype = edge_attr.dtype
        B, _, E = edge_index.shape
        H = self.num_heads

        src = edge_index[:, 0, :].long()  # [B,E]
        dst = edge_index[:, 1, :].long()  # [B,E]
        m = edge_mask

        # adjacency
        adj = torch.zeros((B, N, N), device=device, dtype=torch.bool)
        if E > 0 and m.any():
            b_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, E)  # [B,E]
            bf = b_ids[m]   # [M]
            sf = src[m]     # [M]
            df = dst[m]     # [M]
            adj[bf, sf, df] = True
            if self.undirected:
                adj[bf, df, sf] = True

        # edge values for this step: val = edge_attr @ w_pos[step] -> [B,E,H]
        W = self.w_pos[step].to(device=device, dtype=dtype)  # [D,H]
        val = torch.matmul(edge_attr, W)  # [B,E,H]

        EV = torch.zeros((B, N, N, H), device=device, dtype=dtype)
        if E > 0 and m.any():
            b_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, E)  # [B,E]
            bf = b_ids[m]       # [M]
            sf = src[m]         # [M]
            df = dst[m]         # [M]
            vf = val[m]         # [M,H]
            EV[bf, sf, df, :] += vf
            if self.undirected:
                EV[bf, df, sf, :] += vf

        return adj, EV

    def forward(
        self,
        edge_index: torch.Tensor,          # [2,E] or [B,2,E]
        edge_attr: torch.Tensor,           # [E,D] or [B,E,D]
        spatial_pos: Optional[torch.Tensor] = None,  # [N,N] or [B,N,N] (recommended)
        edge_mask: Optional[torch.Tensor] = None,    # [B,E]
    ) -> torch.Tensor:
        """
        Returns:
          single graph : [H, N, N]
          batched      : [B, H, N, N]
        """
        device = edge_attr.device
        dtype = edge_attr.dtype
        H = self.num_heads

        N, prefix = self._infer_N_and_prefix(edge_index, spatial_pos)
        if N == 0:
            # empty graph(s)
            if edge_index.dim() == 2:
                return torch.zeros((H, 0, 0), device=device, dtype=dtype)
            else:
                B = edge_index.size(0)
                return torch.zeros((B, H, 0, 0), device=device, dtype=dtype)

        # -------------------------
        # Normalize to batched form
        # -------------------------
        single = (edge_index.dim() == 2)
        if single:
            if edge_attr.dim() != 2 or edge_attr.size(1) != self.edge_dim:
                raise ValueError(f"edge_attr must be [E,{self.edge_dim}] for edge_index [2,E]")
            edge_index_b = edge_index.unsqueeze(0)         # [1,2,E]
            edge_attr_b  = edge_attr.unsqueeze(0)          # [1,E,D]
            B = 1
            if spatial_pos is not None:
                if spatial_pos.dim() != 2:
                    raise ValueError("single-graph spatial_pos must be [N,N]")
                spatial_b = spatial_pos.unsqueeze(0)       # [1,N,N]
            else:
                spatial_b = None
        else:
            if edge_attr.dim() != 3 or edge_attr.size(-1) != self.edge_dim:
                raise ValueError(f"edge_attr must be [B,E,{self.edge_dim}] for edge_index [B,2,E]")
            B = edge_index.size(0)
            edge_index_b = edge_index                      # [B,2,E]
            edge_attr_b  = edge_attr                       # [B,E,D]
            if spatial_pos is not None:
                if spatial_pos.dim() != 3:
                    raise ValueError("batched spatial_pos must be [B,N,N]")
                spatial_b = spatial_pos
            else:
                spatial_b = None

        E = edge_index_b.size(2)
        if edge_mask is None:
            edge_mask_b = torch.ones((B, E), device=device, dtype=torch.bool)
        else:
            edge_mask_b = edge_mask.to(torch.bool)
            if edge_mask_b.shape != (B, E):
                raise ValueError("edge_mask must be [B,E]")

        # output
        edge_bias = torch.zeros((B, H, N, N), device=device, dtype=dtype)

        if E == 0 or not edge_mask_b.any():
            return edge_bias.squeeze(0) if single else edge_bias

        # -------------------------------------------------------
        # Build base adjacency once (we will reuse it for reach DP)
        # We can get it from step=0 construction (same adj for all steps)
        # -------------------------------------------------------
        adj, EV1 = self._build_adj_and_ev(
            edge_index=edge_index_b,
            edge_attr=edge_attr_b,
            edge_mask=edge_mask_b,
            N=N,
            step=0,
        )
        adj_f = adj.to(torch.float32)  # [B,N,N]

        # Distance discovery (if spatial_pos is not provided)
        if spatial_b is None:
            reached = torch.zeros((B, N, N), device=device, dtype=torch.bool)
        else:
            reached = None  # not needed

        # d = 1 init
        counts = adj_f.clone()                   # [B,N,N] float
        sumEV  = EV1.clone()                     # [B,N,N,H]

        if spatial_b is None:
            exact = (counts > 0.0) & (~reached)  # first time discovered at d=1
            reached = reached | (counts > 0.0)
        else:
            exact = (spatial_b == 1)

        # bias for d=1: (sumEV/counts)/1
        if exact.any():
            denom = counts.clamp_min(1e-12).unsqueeze(-1)  # [B,N,N,1]
            avg = (sumEV / denom) / 1.0                    # [B,N,N,H]
            # assign
            edge_bias.permute(0, 2, 3, 1)[exact] = avg[exact]

        # d = 2..max_dist
        for d in range(2, self.max_dist + 1):
            # step-specific direct-edge values (last-edge contribution uses w_pos[d-1])
            _, EVd = self._build_adj_and_ev(
                edge_index=edge_index_b,
                edge_attr=edge_attr_b,
                edge_mask=edge_mask_b,
                N=N,
                step=d - 1,
            )
            EVd_h = EVd  # [B,N,N,H]
            EVd_h_f = EVd_h.to(torch.float32)

            # new_counts = counts @ adj
            new_counts = torch.bmm(counts, adj_f)  # [B,N,N]

            if new_counts.max().item() == 0.0:
                break

            # Update sumEV for each head:
            # new_sum = (sumEV @ adj) + (counts @ (EVd * adj))
            # where EVd already exists only on edges; (EVd * adj) is safe.
            sumEV_new = torch.empty_like(sumEV)
            EVd_adj = EVd_h_f * adj_f.unsqueeze(-1)  # [B,N,N,H]

            # head-wise bmm without Python loops:
            # reshape to [B*H, N, N] and bmm once
            sumEV_bh = sumEV.to(torch.float32).permute(0, 3, 1, 2).reshape(B * H, N, N)
            counts_bh = counts.unsqueeze(1).expand(B, H, N, N).reshape(B * H, N, N)
            adj_bh = adj_f.unsqueeze(1).expand(B, H, N, N).reshape(B * H, N, N)
            EVd_adj_bh = EVd_adj.permute(0, 3, 1, 2).reshape(B * H, N, N)

            term1 = torch.bmm(sumEV_bh, adj_bh)        # [B*H,N,N]
            term2 = torch.bmm(counts_bh, EVd_adj_bh)   # [B*H,N,N]
            sumEV_new_f = term1 + term2                # [B*H,N,N]
            sumEV_new = sumEV_new_f.reshape(B, H, N, N).permute(0, 2, 3, 1).to(dtype)

            # exact mask
            if spatial_b is None:
                exact = (new_counts > 0.0) & (~reached)
                reached = reached | (new_counts > 0.0)
            else:
                exact = (spatial_b == d)

            if exact.any():
                denom = new_counts.clamp_min(1e-12).unsqueeze(-1)  # [B,N,N,1]
                avg = (sumEV_new / denom) / float(d)               # [B,N,N,H]
                edge_bias.permute(0, 2, 3, 1)[exact] = avg[exact]

            counts = new_counts
            sumEV = sumEV_new

        return edge_bias.squeeze(0) if single else edge_bias

class GraphormerMHA(nn.Module):
    """
    Graphormer-like MHA wrapper using fairseq-like SelfAttention/MultiheadAttention.

    Supports:
      x        : [..., N, D]
      dist_bias: [..., H, N, N] or None
      edge_bias: [..., H, N, N] or None
      attn_mask: [..., N, N] bool (True = allowed) or None

    Returns:
      out: [..., N, D]
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.self_attn = SelfAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    @staticmethod
    def _prod(shape: List[int]) -> int:
        # small helper (no need for torch) – prefix is typically tiny
        p = 1
        for s in shape:
            p *= int(s)
        return p

    def forward(
        self,
        x: torch.Tensor,                           # [..., N, D]
        *,
        node_mask: torch.Tensor,                   # [B, N] bool (True = valid node, False = padding)
        dist_bias: Optional[torch.Tensor] = None,  # [..., H, N, N]
        edge_bias: Optional[torch.Tensor] = None,  # [..., H, N, N]
        attention_mask: Optional[torch.Tensor] = None,  # [B, N, N] bool (True = allowed)
        build_pairwise_mask: bool = False,
    ) -> torch.Tensor:
        # -------------------------
        # Input validation
        # -------------------------
        if x.dim() < 3:
            raise ValueError("x must be [..., N, D]")

        *prefix, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"x last dim {D} != model dim {self.dim}")

        # Flatten prefix dimensions into batch dimension B
        B = int(torch.tensor(prefix).prod().item()) if len(prefix) > 1 else int(prefix[0])
        x_b = x.reshape(B, N, D)

        if node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be boolean")
        if node_mask.shape != (B, N):
            raise ValueError(f"node_mask must have shape [B, N] = [{B}, {N}]")

        device = x.device
        dtype = x.dtype

        # -------------------------
        # Query tensor
        # -------------------------
        # Keep q as [B, N, D] as requested
        q = x_b

        # -------------------------
        # Build attention bias [B*H, N, N]
        # -------------------------
        attn_bias = None
        if dist_bias is not None or edge_bias is not None:
            if dist_bias is None:
                bias_sum = edge_bias
            elif edge_bias is None:
                bias_sum = dist_bias
            else:
                bias_sum = dist_bias + edge_bias

            if bias_sum.dim() < 4:
                raise ValueError("bias must be [..., H, N, N]")

            *bprefix, H, N1, N2 = bias_sum.shape
            if N1 != N or N2 != N:
                raise ValueError("bias last two dims must be [N, N]")
            if H != self.num_heads:
                raise ValueError("head dimension mismatch")

            # Flatten bias prefix to match B
            B2 = int(torch.tensor(bprefix).prod().item()) if len(bprefix) > 1 else int(bprefix[0])
            if B2 != B:
                raise ValueError("bias batch size must match x batch size")

            bias_b = bias_sum.reshape(B, H, N, N)
            attn_bias = bias_b.contiguous()

        # -------------------------
        # Key padding mask
        # -------------------------
        # True indicates padding positions
        key_padding_mask = (~node_mask).contiguous()  # [B, N]

        # -------------------------
        # Optional pairwise additive mask
        # -------------------------
        additive_mask = None
        if attention_mask is not None or build_pairwise_mask:
            additive_mask = node_mask[:, :, None] & node_mask[:, None, :]  # [B, N, N]
            if attention_mask is not None:
                if attention_mask.shape != (B, N, N):
                    raise ValueError(
                        f"attention_mask must have shape {(B, N, N)}, "
                        f"got {tuple(attention_mask.shape)}."
                    )
                additive_mask = additive_mask & attention_mask.to(torch.bool)

        # -------------------------
        # Self-attention
        # -------------------------
        out, _ = self.self_attn(
            x=q,                        # [B, N, D]
            attn_bias=attn_bias,        # [B*H, N, N] or None
            key_padding_mask=key_padding_mask,  # [B, N]
            need_weights=False,
            attn_mask=additive_mask,    # None or [B*H, N, N]
            need_head_weights=False,
        )  # [B, N, D]

        # -------------------------
        # Restore original prefix shape
        # -------------------------
        out_full = out.reshape(*prefix, N, D)
        return out_full


class GraphormerBlock(nn.Module):
    """
    Pre-LN Graphormer block (MHA + FFN) with selectable execution mode.

    mode="loop":
      - Caller provides per-graph node_slices and per-graph bias lists.
      - Block runs attention per graph internally (no cross-graph attention possible).

    mode="mask":
      - Caller provides full-batch biases and an attention mask (block-diagonal).
      - Block runs one big attention; cross-graph positions are masked out.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, ffn_mult: int = 4, freeze_token_count: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mha = GraphormerMHA(dim, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.drop2 = nn.Dropout(dropout)

        self.freeze_token_count = freeze_token_count

    def forward(
        self,
        x: torch.Tensor,                            # [B, num_tokens, D]
        dist_bias: Optional[torch.Tensor] = None,   # [B, H, num_tokens, num_tokens]
        edge_bias: Optional[torch.Tensor] = None,   # [B, H, num_tokens, num_tokens]
        node_mask: Optional[torch.Tensor] = None,   # [B, num_tokens] bool
        attention_mask: Optional[torch.Tensor] = None,  # [B, num_tokens, num_tokens] bool
    ) -> torch.Tensor:

        if self.freeze_token_count < 0:
            raise ValueError("freeze_token_count must be non-negative.")

        if self.freeze_token_count > x.size(1):
            raise ValueError(
                "freeze_token_count must be <= number of tokens. "
                f"Got freeze_token_count={self.freeze_token_count}, "
                f"num_tokens={x.size(1)}."
            )

        if self.freeze_token_count > 0:
            frozen_tokens = x[:, :self.freeze_token_count, :].clone()
            # [B, freeze_token_count, D]
        else:
            frozen_tokens = None

        h = self.norm1(x)
        h = self.mha(
            h,
            node_mask=node_mask,
            dist_bias=dist_bias,
            edge_bias=edge_bias,
            attention_mask=attention_mask,
        )
        x = x + self.drop1(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)

        if frozen_tokens is not None:
            x[:, :self.freeze_token_count, :] = frozen_tokens

        return x


class GraphormerEncoder(nn.Module):
    """Generic Graphormer encoder for graph-structured node tokens.

    Token layout inside attention is:
        [condition tokens, graph tokens, node tokens]

    ``node_dim`` is the input node feature width. ``hidden_dim`` is the shared
    token width used by node, graph, and condition tokens. A graph
    representation is always returned: graph tokens are used when
    ``graph_token_count > 0``; otherwise masked mean pooling over node tokens is
    used. Condition tokens are fixed context tokens for sample-level inputs such
    as collision energy or adduct type.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        edge_dim: int = 0,
        num_heads: int = 8,
        num_layers: int = 2,
        max_degree: int = 8,
        max_spatial_dist: int = 4,
        max_edge_dist: int = 4,
        *,
        graph_token_count: int = 0,
        condition_dim: Optional[int] = None,
        condition_token_count: int = 0,
        dropout: float = 0.0,
        graph_to_node: bool = True,
        condition_to_node: bool = True,
        condition_to_graph: bool = True,
        undirected_for_spd: bool = True,
        undirected_for_path: bool = True,
        start_cap: int = 64,
        cap_growth: float = 2.0,
    ):
        super().__init__()

        node_dim = int(node_dim)
        hidden_dim = int(hidden_dim)
        graph_token_count = int(graph_token_count)

        if node_dim <= 0:
            raise ValueError("node_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if graph_token_count < 0:
            raise ValueError("graph_token_count must be non-negative.")

        if condition_dim is not None and int(condition_dim) > 0 and condition_token_count <= 0:
            condition_token_count = 1
        condition_token_count = int(condition_token_count)
        if condition_token_count < 0:
            raise ValueError("condition_token_count must be non-negative.")

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_spatial_dist = int(max_spatial_dist)
        self.max_edge_dist = int(max_edge_dist)
        self.max_degree = int(max_degree)
        self.edge_dim = int(edge_dim)
        self.graph_token_count = graph_token_count
        self.condition_token_count = condition_token_count
        self.graph_repr_dim = self.hidden_dim * max(1, self.graph_token_count)
        self.condition_repr_dim = self.hidden_dim * self.condition_token_count
        self.graph_to_node = bool(graph_to_node)
        self.condition_to_node = bool(condition_to_node)
        self.condition_to_graph = bool(condition_to_graph)
        self.undirected_for_spd = undirected_for_spd
        self.undirected_for_path = undirected_for_path
        self.start_cap = int(start_cap)
        self.cap_growth = float(cap_growth)

        self.node_proj = (
            nn.Identity()
            if self.node_dim == self.hidden_dim
            else nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        )
        self.centrality = CentralityEncoding(dim=self.hidden_dim, max_degree=self.max_degree)
        self.spatial = SpatialEncoding(num_heads=self.num_heads, max_dist=self.max_spatial_dist)

        self.edge_enc = (
            EdgeEncoding(
                edge_dim=self.edge_dim,
                num_heads=self.num_heads,
                max_dist=self.max_edge_dist,
                undirected=undirected_for_path,
            )
            if self.edge_dim > 0
            else None
        )

        self.blocks = nn.ModuleList([
            GraphormerBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                freeze_token_count=self.condition_token_count,
            )
            for _ in range(self.num_layers)
        ])

        if self.graph_token_count > 0:
            self.graph_tokens = nn.Parameter(torch.empty(self.graph_token_count, self.hidden_dim))
            nn.init.normal_(self.graph_tokens, mean=0.0, std=0.02)
        else:
            self.graph_tokens = None

        if self.condition_token_count > 0:
            condition_dim = int(condition_dim or self.condition_repr_dim)
            self.condition_dim = condition_dim
            self.condition_proj = (
                nn.Identity()
                if condition_dim == self.condition_repr_dim
                else nn.Linear(condition_dim, self.condition_repr_dim, bias=False)
            )
        else:
            self.condition_dim = 0
            self.condition_proj = None

    def forward(
        self,
        data: Data | Batch,
        condition_repr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(data, (Data, Batch)), "data must be a Data or Batch instance"
        assert data.x is not None, "data.x (node features) must be provided"

        num_graphs = self._num_graphs(data)
        if condition_repr is not None:
            if self.condition_token_count <= 0:
                raise ValueError("condition_repr was provided but condition_token_count is 0.")
            if condition_repr.size(0) != num_graphs:
                raise ValueError("condition_repr batch size mismatch.")
            if condition_repr.size(1) != self.condition_dim:
                raise ValueError(
                    f"condition_repr feature size mismatch: expected {self.condition_dim}, "
                    f"got {condition_repr.size(1)}."
                )

        bucket_list, ids_list = PaddedGraphBatch.bucketize_to_padded_graph_batches(
            data,
            start_cap=self.start_cap,
            growth=self.cap_growth,
        )

        device = data.x.device
        out_h = data.x.new_zeros((data.num_nodes, self.hidden_dim))
        out_graph_repr = data.x.new_zeros((num_graphs, self.graph_repr_dim))

        for bucket, ids in zip(bucket_list, ids_list):
            ids = ids.to(device)
            h = self._project_and_centrality_bucket(bucket)
            dist_bias, edge_bias = self._compute_dist_and_edge_biases(bucket)
            node_mask = bucket.node_mask

            local_condition = None
            if condition_repr is not None:
                local_condition = self._condition_repr_to_tokens(condition_repr[ids])

            h, node_mask, dist_bias, edge_bias, attention_mask = self._add_context_tokens(
                h=h,
                node_mask=node_mask,
                dist_bias=dist_bias,
                edge_bias=edge_bias,
                condition_tokens=local_condition,
            )

            for blk in self.blocks:
                h = blk(
                    h,
                    node_mask=node_mask,
                    dist_bias=dist_bias,
                    edge_bias=edge_bias,
                    attention_mask=attention_mask,
                )

            node_h, node_mask_wo_context, local_graph_repr = self._remove_context_tokens_and_readout(
                h=h,
                node_mask=node_mask,
            )
            out_graph_repr[ids] = local_graph_repr

            node_ids = bucket.node_ids.to(device)
            valid = node_mask_wo_context.to(torch.bool)
            out_h[node_ids[valid]] = node_h[valid]

        return out_h, out_graph_repr

    @staticmethod
    def _num_graphs(data: Data | Batch) -> int:
        if hasattr(data, "num_graphs") and data.num_graphs is not None:
            return int(data.num_graphs)
        if hasattr(data, "batch") and data.batch is not None and data.batch.numel() > 0:
            return int(data.batch.max().item() + 1)
        return 1

    def _project_and_centrality_bucket(self, bb: PaddedGraphBatch) -> torch.Tensor:
        x = bb.x
        if x is None:
            raise ValueError("bb.x is required")
        h = self.node_proj(x)
        return self.centrality(
            h,
            edge_index=bb.edge_index,
            edge_mask=bb.edge_mask,
            node_mask=bb.node_mask,
        )

    def _compute_dist_and_edge_biases(self, bb: PaddedGraphBatch) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        adj = bb.build_adj(undirected=self.undirected_for_spd, add_self_loops=False)
        spatial_pos = SpatialEncoding.compute_spatial_pos(
            adj=adj,
            node_mask=bb.node_mask,
            max_dist=self.max_spatial_dist,
            add_self_loops=False,
        )
        dist_bias = self.spatial(spatial_pos)
        edge_bias = torch.zeros_like(dist_bias) if self.edge_enc is not None else None
        if self.edge_enc is not None and bb.edge_attr is not None and bb.edge_attr.numel() > 0:
            edge_bias = self.edge_enc(
                edge_index=bb.edge_index,
                edge_attr=bb.edge_attr,
                spatial_pos=spatial_pos,
                edge_mask=bb.edge_mask,
            )
        return dist_bias, edge_bias

    def _condition_repr_to_tokens(self, condition_repr: torch.Tensor) -> torch.Tensor:
        if self.condition_proj is None:
            raise ValueError("condition tokens are disabled.")
        condition_tokens = self.condition_proj(condition_repr)
        return condition_tokens.view(condition_repr.size(0), self.condition_token_count, self.hidden_dim)

    def _add_context_tokens(
        self,
        *,
        h: torch.Tensor,
        node_mask: torch.Tensor,
        dist_bias: Optional[torch.Tensor],
        edge_bias: Optional[torch.Tensor],
        condition_tokens: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        device = h.device
        B, N, hidden_dim = h.shape
        C = self.condition_token_count
        G = self.graph_token_count
        prefix_parts = []

        if C > 0:
            if condition_tokens is None:
                raise ValueError("condition_repr is required when condition_token_count > 0.")
            if condition_tokens.shape != (B, C, hidden_dim):
                raise ValueError(
                    f"condition_tokens must have shape {(B, C, hidden_dim)}, "
                    f"got {tuple(condition_tokens.shape)}."
                )
            prefix_parts.append(condition_tokens)

        if G > 0:
            if self.graph_tokens is None:
                raise ValueError("graph tokens are not initialized.")
            prefix_parts.append(self.graph_tokens.view(1, G, hidden_dim).expand(B, G, hidden_dim))

        if prefix_parts:
            prefix = torch.cat(prefix_parts, dim=1)
            h = torch.cat([prefix, h], dim=1)
            prefix_mask = torch.ones((B, C + G), dtype=torch.bool, device=device)
            node_mask = torch.cat([prefix_mask, node_mask.to(torch.bool)], dim=1)
            dist_bias = self._prepend_zero_bias(dist_bias, C + G)
            edge_bias = self._prepend_zero_bias(edge_bias, C + G)

        attention_mask = self._build_attention_mask(
            batch_size=B,
            num_nodes=N,
            device=device,
        )
        return h, node_mask, dist_bias, edge_bias, attention_mask

    def _prepend_zero_bias(
        self,
        bias: Optional[torch.Tensor],
        prefix_count: int,
    ) -> Optional[torch.Tensor]:
        if bias is None or prefix_count <= 0:
            return bias
        B, H, N, _ = bias.shape
        out = bias.new_zeros((B, H, prefix_count + N, prefix_count + N))
        out[:, :, prefix_count:, prefix_count:] = bias
        return out

    def _build_attention_mask(
        self,
        *,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        C = self.condition_token_count
        G = self.graph_token_count
        total = C + G + num_nodes
        mask = torch.ones((batch_size, total, total), dtype=torch.bool, device=device)

        node_start = C + G
        graph_start = C
        graph_end = C + G

        if C > 0 and not self.condition_to_node:
            mask[:, node_start:, :C] = False
        if C > 0 and G > 0 and not self.condition_to_graph:
            mask[:, graph_start:graph_end, :C] = False
        if G > 0 and not self.graph_to_node:
            mask[:, node_start:, graph_start:graph_end] = False

        return mask

    def _remove_context_tokens_and_readout(
        self,
        *,
        h: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C = self.condition_token_count
        G = self.graph_token_count
        node_start = C + G
        node_h = h[:, node_start:, :]
        node_mask = node_mask[:, node_start:]

        if G > 0:
            graph_token_h = h[:, C: C + G, :]
            graph_repr = graph_token_h.reshape(h.size(0), G * self.hidden_dim)
            return node_h, node_mask, graph_repr

        mask = node_mask.to(h.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        graph_repr = (node_h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return node_h, node_mask, graph_repr

if __name__ == "__main__":
    torch.manual_seed(0)

    # Build two small graphs and batch them
    Fin = 12
    dim = 32
    num_heads = 2
    num_layers = 2
    edge_dim = 8


    graph_list = [
        # 1) Path graph (chain) N=6
        Data(
            x=torch.randn(6, Fin),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(5, edge_dim),
        ),

        # 2) Cycle graph (ring) N=6
        Data(
            x=torch.randn(6, Fin),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 0]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(6, edge_dim),
        ),

        # 3) Star graph N=7 (center=0)
        Data(
            x=torch.randn(7, Fin),
            edge_index=torch.tensor(
                [[0, 0, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(6, edge_dim),
        ),

        # 4) Binary tree (directed parent->child) N=7
        Data(
            x=torch.randn(7, Fin),
            edge_index=torch.tensor(
                [[0, 0, 1, 1, 2, 2],
                [1, 2, 3, 4, 5, 6]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(6, edge_dim),
        ),

        # 5) Diamond / two-paths merge N=5
        # 0 -> {1,2} -> 3 -> 4  and 1->2 cross
        Data(
            x=torch.randn(5, Fin),
            edge_index=torch.tensor(
                [[0, 0, 1, 2, 1, 3],
                [1, 2, 3, 3, 2, 4]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(6, edge_dim),
        ),

        # 6) Bipartite-like directed K(3,3) partial N=6 (L={0,1,2}, R={3,4,5})
        Data(
            x=torch.randn(6, Fin),
            edge_index=torch.tensor(
                [[0, 0, 0, 1, 1, 2, 2],
                [3, 4, 5, 3, 5, 4, 5]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(7, edge_dim),
        ),

        # 7) Grid 3x3 (directed right + down) N=9 nodes indexed row-major
        # edges: (r,c)->(r,c+1) and (r,c)->(r+1,c)
        Data(
            x=torch.randn(9, Fin),
            edge_index=torch.tensor(
                [[0, 1, 3, 4, 6, 7, 0, 1, 2, 3, 4, 5],
                [1, 2, 4, 5, 7, 8, 3, 4, 5, 6, 7, 8]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(12, edge_dim),
        ),

        # 8) Two triangles connected by a bridge N=6
        # triangle A: 0->1->2->0, triangle B: 3->4->5->3, bridge: 2->3
        Data(
            x=torch.randn(6, Fin),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5, 2],
                [1, 2, 0, 4, 5, 3, 3]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(7, edge_dim),
        ),

        # 9) DAG with skip connections N=8
        # 0->1->2->3->4->5->6->7 plus skips 0->2,1->3,2->5,4->7
        Data(
            x=torch.randn(8, Fin),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 4],
                [1, 2, 3, 4, 5, 6, 7, 2, 3, 5, 7]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(11, edge_dim),
        ),

        # 10) "Molecule-ish" graph: benzene ring + tail N=8
        # ring: 0-1-2-3-4-5-0 (directed), tail: 2->6->7, and 6->3 (extra chord)
        Data(
            x=torch.randn(8, Fin),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5, 2, 6, 6],
                [1, 2, 3, 4, 5, 0, 6, 7, 3]],
                dtype=torch.long
            ),
            edge_attr=torch.randn(9, edge_dim),
        )
    ]

    batch = Batch.from_data_list(graph_list)

    # Test both execution modes
    enc = GraphormerEncoder(
        node_dim=Fin,
        hidden_dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_spatial_dist=6,
        max_edge_dist=6,
        edge_dim=edge_dim,
        max_degree=3,
        graph_token_count=1,
        dropout=0.0,
        undirected_for_spd=True,
        undirected_for_path=True,
        
        start_cap=8,
        cap_growth=2.0,
    )
    enc2 = GraphormerEncoder(
        node_dim=dim,
        hidden_dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_spatial_dist=6,
        max_edge_dist=6,
        edge_dim=edge_dim,
        max_degree=3,
        graph_token_count=1,
        dropout=0.0,
        undirected_for_spd=True,
        undirected_for_path=True,
        
        start_cap=8,
        cap_growth=2.0,
    )

    out, graph_repr = enc(batch)
    batch.x = out  # feed first encoder output as input to second encoder
    out, graph_repr = enc2(batch)
    loss = out.mean() + (graph_repr.mean() if graph_repr is not None else 0.0)
    loss.backward()

    print("out_h shape:", tuple(out.shape))
    if graph_repr is not None:
        print("graph_repr shape:", tuple(graph_repr.shape))
    print("loss:", float(loss.item()))