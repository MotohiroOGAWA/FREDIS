from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import Tensor,nn
from torch_geometric.data import Data

from .edge_gcn import EdgeAwareGCN


class PatternReactantEncoder(nn.Module):
    def __init__(self,atom_dim:int,bond_dim:int,context_dim:int,position_count:int,output_dim:int,num_layers:int=2)->None:
        super().__init__()
        if position_count<1:raise ValueError("position_count must be positive")
        self.position_count=position_count
        self.gcn=EdgeAwareGCN(atom_dim+context_dim+position_count,bond_dim,output_dim,num_layers)
        self.output_dim=output_dim

    def forward(self,graph:Data,context_by_atom:Tensor,position:Tensor|None=None)->Tensor:
        count=graph.x.shape[0]
        if context_by_atom.shape[0]!=count:raise ValueError("one context embedding is required per reactant atom")
        if position is None:position=torch.arange(count,device=graph.x.device)
        if torch.any((position<0)|(position>=self.position_count)):raise ValueError("SMARTS position is out of range")
        one_hot=torch.nn.functional.one_hot(position,num_classes=self.position_count).to(graph.x.dtype)
        return self.gcn(torch.cat((graph.x,context_by_atom,one_hot),dim=-1),graph.edge_index,graph.edge_attr)


class ReactantEncoderRegistry(nn.Module):
    """Own one parameter set per ReactionPattern while sharing output size."""
    def __init__(self,pattern_atom_counts:Mapping[int,int],atom_dim:int,bond_dim:int,context_dim:int,output_dim:int,num_layers:int=2)->None:
        super().__init__();self.output_dim=output_dim
        self.encoders=nn.ModuleDict({str(int(pid)):PatternReactantEncoder(atom_dim,bond_dim,context_dim,int(count),output_dim,num_layers) for pid,count in pattern_atom_counts.items()})

    def forward(self,pattern_id:int,graph:Data,context_by_atom:Tensor,position:Tensor|None=None)->Tensor:
        try:encoder=self.encoders[str(int(pattern_id))]
        except KeyError as exc:raise KeyError(f"Unknown reaction pattern ID: {pattern_id}") from exc
        return encoder(graph,context_by_atom,position)
