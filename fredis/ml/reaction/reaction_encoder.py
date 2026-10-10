from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor,nn


@dataclass(frozen=True)
class ReactionTransformation:
    bond_atom_index:Tensor
    before_bond_features:Tensor
    after_bond_features:Tensor
    bond_kind:Tensor
    retained_mask:Tensor
    removed_mask:Tensor
    product_role:Tensor


class ReactionEncoder(nn.Module):
    """Encode one concrete ReactionInstance from atoms and transformations."""
    def __init__(self,atom_dim:int,bond_dim:int,output_dim:int,product_role_count:int=32)->None:
        super().__init__();self.output_dim=output_dim
        self.bond_encoder=nn.Sequential(nn.Linear(atom_dim*2+bond_dim*2+3,output_dim),nn.GELU(),nn.Linear(output_dim,output_dim))
        self.atom_status=nn.Linear(atom_dim+2,output_dim)
        self.product_role=nn.Embedding(product_role_count,output_dim)
        self.output=nn.Sequential(nn.Linear(output_dim*3,output_dim),nn.GELU(),nn.LayerNorm(output_dim))

    def forward(self,atom_h:Tensor,transformation:ReactionTransformation)->Tensor:
        device=atom_h.device
        if transformation.bond_atom_index.numel():
            u,v=transformation.bond_atom_index
            kind=torch.nn.functional.one_hot(transformation.bond_kind.long(),3).to(atom_h.dtype)
            bonds=self.bond_encoder(torch.cat((atom_h[u],atom_h[v],transformation.before_bond_features,transformation.after_bond_features,kind),dim=-1)).mean(0)
        else:bonds=atom_h.new_zeros(self.output_dim)
        status=torch.stack((transformation.retained_mask,transformation.removed_mask),dim=-1).to(atom_h.dtype)
        atoms=self.atom_status(torch.cat((atom_h,status),dim=-1)).mean(0) if atom_h.shape[0] else atom_h.new_zeros(self.output_dim)
        role=self.product_role(transformation.product_role.to(device).long()).reshape(-1,self.output_dim).mean(0)
        return self.output(torch.cat((bonds,atoms,role),dim=-1))
