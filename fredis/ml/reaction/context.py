from __future__ import annotations

from collections import deque
import torch
from torch import nn
from torch_geometric.data import Batch,Data

from ...libs.mmkit.mmkit import Compound
from ..mol.graph_builder import MolGraphBuilder
from .edge_gcn import EdgeAwareGCN,graph_mean_pool


class LocalContextGraphBuilder:
    """Extract the k-hop graph outside a concrete SMARTS match."""

    def __init__(self,graph_builder:MolGraphBuilder,radius:int=2)->None:
        if radius<1:raise ValueError("radius must be positive")
        self.graph_builder=graph_builder;self.radius=int(radius)

    def build(self,compound:Compound,matched_atom_indices:list[int]|tuple[int,...]|set[int])->Data:
        base=self.graph_builder.build(compound);matched=set(map(int,matched_atom_indices))
        atoms=self.graph_builder.graph_atoms(compound);rd_to_graph={atom.GetIdx():i for i,atom in enumerate(atoms)}
        matched_graph={rd_to_graph[i] for i in matched if i in rd_to_graph}
        adjacency=[set() for _ in atoms]
        for u,v in base.edge_index.t().tolist():adjacency[u].add(v)
        distance={index:0 for index in matched_graph};queue=deque(matched_graph)
        external=set()
        while queue:
            current=queue.popleft();depth=distance[current]
            if depth>=self.radius:continue
            for neighbor in adjacency[current]:
                if neighbor in matched_graph:continue
                external.add(neighbor)
                if neighbor not in distance:
                    distance[neighbor]=depth+1;queue.append(neighbor)
        anchors={m for m in matched_graph if adjacency[m]&external}
        if not external:
            return Data(x=base.x.new_empty((0,base.x.shape[1])),edge_index=torch.empty((2,0),dtype=torch.long,device=base.x.device),
                        edge_attr=base.edge_attr.new_empty((0,base.edge_attr.shape[1])),boundary_anchor_mask=torch.empty(0,dtype=torch.bool,device=base.x.device),
                        source_atom_index=torch.empty(0,dtype=torch.long,device=base.x.device))
        selected=sorted(external|anchors);remap={old:new for new,old in enumerate(selected)}
        keep=[]
        for i,(u,v) in enumerate(base.edge_index.t().tolist()):
            if u not in remap or v not in remap:continue
            if u in anchors and v in anchors:continue
            keep.append(i)
        edge_index=torch.tensor([[remap[int(base.edge_index[0,i])],remap[int(base.edge_index[1,i])]] for i in keep],dtype=torch.long,device=base.x.device).t()
        if not keep:edge_index=torch.empty((2,0),dtype=torch.long,device=base.x.device)
        x=base.x[selected].clone();mask=torch.tensor([value in anchors for value in selected],dtype=torch.bool,device=x.device);x[mask]=0
        return Data(x=x,edge_index=edge_index,edge_attr=base.edge_attr[keep],boundary_anchor_mask=mask,
                    source_atom_index=torch.tensor(selected,dtype=torch.long,device=x.device))

    def build_from_source_atom_maps(self,compound:Compound,source_atom_maps:list[int]|tuple[int,...]|set[int])->Data:
        """Build context from the stable mapped-Source identifiers on a ReactionInstance."""
        by_map={atom.GetAtomMapNum():atom.GetIdx() for atom in compound.mapped_mol.GetAtoms()}
        missing=set(map(int,source_atom_maps))-by_map.keys()
        if missing:raise ValueError(f"Source atom maps are absent from the molecule: {sorted(missing)}")
        return self.build(compound,[by_map[int(value)] for value in source_atom_maps])


class LocalContextEncoder(nn.Module):
    def __init__(self,atom_dim:int,bond_dim:int,hidden_dim:int,num_layers:int=2,dropout:float=0.0)->None:
        super().__init__();self.gcn=EdgeAwareGCN(atom_dim,bond_dim,hidden_dim,num_layers,dropout);self.output_dim=hidden_dim

    def forward(self,data:Data|Batch):
        if data.x.shape[0]==0:
            count=int(getattr(data,"num_graphs",1));return data.x.new_zeros((count,self.output_dim))
        h=self.gcn(data.x,data.edge_index,data.edge_attr)
        return graph_mean_pool(h,getattr(data,"batch",None),getattr(data,"num_graphs",None))
