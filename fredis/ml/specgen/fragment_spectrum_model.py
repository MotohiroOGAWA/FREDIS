"""Prepared physical-ion candidates and CE-conditioned intensity model."""
from __future__ import annotations
import math
from dataclasses import dataclass,replace
from collections.abc import Sequence
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from torch_geometric.data import Batch,Data
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.domain.fragment.ion_tree import FragmentIonTree
from fredis.libs.mmkit.mmkit import Adduct
from ..common.layers.graphormer import GraphormerEncoder
from ..mol.formula_encoder import FormulaTensorizer
from .materialization import DecodedFragmentTreeBatch
from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES


@dataclass(frozen=True)
class SpectrumModelBatch:
    decoded: DecodedFragmentTreeBatch
    tree_graph: Batch
    physical_candidate_node_index: Tensor
    physical_candidate_sample_index: Tensor
    physical_candidate_formula_tensor: Tensor
    physical_candidate_mz: Tensor
    physical_candidate_charge: Tensor
    physical_candidate_explanation_ptr: Tensor
    explanation_adduct_index: Tensor
    explanation_hydrogen_shift: Tensor
    target_intensity: Tensor|None=None
    fragment_supervision_node_index: Tensor|None=None
    fragment_supervision_sample_index: Tensor|None=None
    fragment_is_positive: Tensor|None=None
    fragment_positive_weight: Tensor|None=None
    explanation_is_positive: Tensor|None=None
    explanation_positive_weight: Tensor|None=None
    physical_candidate_adduct: tuple[str,...]=()
    ion_type_strs: tuple[str,...]=()
    unique_node_graph: Batch|None=None
    node_graph_inverse: Tensor|None=None
    unique_source_index: Tensor|None=None
    unique_smiles: tuple[str,...]=()
    node_smiles: tuple[str,...]=()

    def to(self,device:str|torch.device):
        names=('physical_candidate_node_index','physical_candidate_sample_index','physical_candidate_formula_tensor',
               'physical_candidate_mz','physical_candidate_charge','physical_candidate_explanation_ptr',
               'explanation_adduct_index','explanation_hydrogen_shift','target_intensity',
               'fragment_supervision_node_index','fragment_supervision_sample_index','fragment_is_positive',
               'fragment_positive_weight','explanation_is_positive','explanation_positive_weight',
               'node_graph_inverse','unique_source_index')
        return replace(self,decoded=self.decoded.to(device),tree_graph=self.tree_graph.to(device),
            unique_node_graph=self.unique_node_graph.to(device) if self.unique_node_graph is not None else None,
            **{name:getattr(self,name).to(device) for name in names if getattr(self,name) is not None})


@dataclass(frozen=True)
class SpectrumModelOutput:
    intensity:Tensor
    sample_index:Tensor
    formula_tensor:Tensor
    mz:Tensor
    loss:Tensor
    confidence:Tensor
    ion_probability:Tensor
    metrics:dict[str,Tensor]
    fragment_probability:Tensor|None=None
    state_probability:Tensor|None=None
    log_physical_probability:Tensor|None=None
    state_logit:Tensor|None=None
    state_log_physical_probability:Tensor|None=None
    adduct_logit:Tensor|None=None
    hydrogen_shift_logit:Tensor|None=None
    explanation_log_probability:Tensor|None=None


def normalized_attention_weights(values:Tensor,group:Tensor,count:int,score_layer:nn.Module)->Tensor:
    """Return attention weights that sum to one inside every candidate."""
    if count==0:return values.new_empty(0)
    score=score_layer(values).squeeze(-1)
    maximum=score.new_full((count,),-torch.inf).scatter_reduce_(0,group,score,reduce='amax',include_self=True)
    weight=(score-maximum[group]).exp()
    denominator=weight.new_zeros(count).scatter_add_(0,group,weight).clamp_min(1e-12)
    return weight/denominator[group]


def normalized_attention_pool(values:Tensor,group:Tensor,count:int,score_layer:nn.Module)->Tensor:
    """Attention pooling whose weights sum to one inside every candidate."""
    if count==0:return values.new_zeros((0,values.shape[-1]))
    weight=normalized_attention_weights(values,group,count,score_layer)
    return values.new_zeros((count,values.shape[-1])).index_add_(0,group,values*weight[:,None])


def sum_explanation_log_probabilities(
    explanation_log_probability: Tensor,
    explanation_ptr: Tensor,
) -> Tensor:
    """Sum all explanation log probabilities for each physical ion.

    This is intentionally neither log-sum-exp nor attention pooling.  It is
    the scoring rule specified by FREDIS for equivalent explanations.
    """
    counts=explanation_ptr[1:]-explanation_ptr[:-1]
    owner=torch.repeat_interleave(torch.arange(counts.numel(),device=counts.device),counts)
    result=explanation_log_probability.new_zeros(counts.numel())
    if explanation_log_probability.numel():result.index_add_(0,owner,explanation_log_probability)
    return result


def balanced_binary_loss(logits:Tensor,positive:Tensor,positive_weight:Tensor|None=None)->tuple[Tensor,Tensor,Tensor]:
    """Class-balanced BCE; positive examples may additionally carry peak weights."""
    positive=positive.bool();negative=~positive;zero=logits.sum()*0
    if torch.any(positive):
        values=F.softplus(-logits[positive])
        if positive_weight is None:positive_loss=values.mean()
        else:
            weights=positive_weight[positive].to(values.dtype).clamp_min(0)
            positive_loss=(values*weights).sum()/weights.sum().clamp_min(1e-12)
    else:positive_loss=zero
    negative_loss=F.softplus(logits[negative]).mean() if torch.any(negative) else zero
    if torch.any(positive) and torch.any(negative):loss=.5*(positive_loss+negative_loss)
    elif torch.any(positive):loss=positive_loss
    elif torch.any(negative):loss=negative_loss
    else:loss=zero
    return loss,positive_loss,negative_loss


def binary_metrics(logits:Tensor,positive:Tensor,prefix:str)->dict[str,Tensor]:
    positive=positive.bool();negative=~positive;predicted=logits.sigmoid()>=.5;zero=logits.sum()*0
    tp=(predicted&positive).sum().to(logits.dtype);fp=(predicted&negative).sum().to(logits.dtype)
    fn=((~predicted)&positive).sum().to(logits.dtype)
    precision=tp/(tp+fp).clamp_min(1);recall=tp/(tp+fn).clamp_min(1)
    return {f'{prefix}_positive_count':positive.sum().to(logits.dtype),
            f'{prefix}_negative_count':negative.sum().to(logits.dtype),
            f'{prefix}_positive_probability_mean':logits[positive].sigmoid().mean() if torch.any(positive) else zero,
            f'{prefix}_negative_probability_mean':logits[negative].sigmoid().mean() if torch.any(negative) else zero,
            f'{prefix}_precision':precision,f'{prefix}_recall':recall,
            f'{prefix}_f1':2*precision*recall/(precision+recall).clamp_min(1e-12)}


@dataclass(frozen=True)
class SpectrumLossSettings:
    """Training objective of :class:`FragmentSpectrumModel`; not part of the architecture."""
    fragment_loss_weight: float = .5
    state_loss_weight: float = .5
    intensity_loss_weight: float = 1.
    cosine_loss_weight: float = .5
    precursor_free_loss_weight: float = .5
    intensity_power: float = .5

    def __post_init__(self) -> None:
        for name in ('fragment_loss_weight','state_loss_weight','intensity_loss_weight','cosine_loss_weight','precursor_free_loss_weight'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0:raise ValueError(f'{name} must be finite and non-negative')
        if not math.isfinite(self.intensity_power) or self.intensity_power<=0:raise ValueError('intensity_power must be positive')


class FragmentSpectrumModel(nn.Module):
    requires_reaction_features=True
    def __init__(self,mol_encoder:nn.Module,reaction_dim:int,
                 hidden_dim:int=128,num_heads:int=4,num_layers:int=2,max_reaction_count:int=3,dropout:float=0.,
                 ion_type_count:int=1,hydrogen_shift_min:int=0,hydrogen_shift_max:int=0,
                 ion_embedding_dim:int=32,hydrogen_shift_embedding_dim:int=16,state_hidden_dim:int|None=None,
                 main_adduct_dim:int|None=None,collision_energy_dim:int=16,descriptors:Sequence[str]=())->None:
        super().__init__()
        self.mol_encoder=mol_encoder
        self.descriptors = tuple(descriptors)
        if len(set(self.descriptors)) != len(self.descriptors) or set(self.descriptors) - set(SUPPORTED_DESCRIPTOR_NAMES):
            raise ValueError('Invalid intensity descriptor selection')
        self.descriptor_indices = tuple(SUPPORTED_DESCRIPTOR_NAMES.index(name) for name in self.descriptors)
        state_hidden_dim=state_hidden_dim or hidden_dim;main_adduct_dim=main_adduct_dim or 128
        self.edge_encoder=nn.Sequential(nn.Linear(reaction_dim+mol_encoder.graph_dim*2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
        self.tree_encoder=GraphormerEncoder(node_dim=mol_encoder.graph_dim,hidden_dim=hidden_dim,edge_dim=hidden_dim,
            num_heads=num_heads,num_layers=num_layers,max_spatial_dist=max_reaction_count+1,max_edge_dist=max_reaction_count+1,
            undirected_for_spd=False,undirected_for_path=False,dropout=dropout)
        if hydrogen_shift_min>hydrogen_shift_max:raise ValueError('hydrogen_shift_min must not exceed hydrogen_shift_max')
        self.hydrogen_shift_min=int(hydrogen_shift_min);self.hydrogen_shift_max=int(hydrogen_shift_max)
        self.ion_embedding=nn.Embedding(max(1,ion_type_count),ion_embedding_dim)
        shift_count=self.hydrogen_shift_max-self.hydrogen_shift_min+1
        self.hydrogen_shift_embedding=nn.Embedding(shift_count,hydrogen_shift_embedding_dim)
        self.hydrogen_shift_encoder=nn.Sequential(nn.Linear(ion_embedding_dim+hydrogen_shift_embedding_dim,state_hidden_dim),nn.GELU(),nn.Linear(state_hidden_dim,hidden_dim))
        self.explanation_attention=nn.Linear(hidden_dim,1)
        self.main_adduct_projection=nn.Linear(main_adduct_dim,hidden_dim)
        self.collision_energy_projection=nn.Linear(collision_energy_dim,hidden_dim)
        self.intensity_head=nn.Sequential(nn.Linear(hidden_dim*4+len(self.descriptors),hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
        self.fragment_score=nn.Linear(hidden_dim,1)
        self.adduct_score=nn.Linear(hidden_dim,max(1,ion_type_count))
        self.hydrogen_shift_score=nn.Linear(hidden_dim*2,1)

    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.mol_encoder.parameters()):self.mol_encoder.eval()
        return self

    def forward(self,data:SpectrumModelBatch,*,reaction_h:Tensor,main_adduct_h:Tensor,
                collision_energy_h:Tensor,source_embeddings:Tensor|None=None,molecular_embeddings:Tensor|None=None,
                loss_settings:SpectrumLossSettings|None=None):
        loss_settings=loss_settings or SpectrumLossSettings()
        if molecular_embeddings is None:
            if data.node_graph_inverse is not None and data.unique_source_index is not None:
                if source_embeddings is None or data.unique_node_graph is None:
                    raise ValueError('Deduplicated prepared graphs require source embeddings and unique graph tensors')
                unique=source_embeddings.new_zeros((data.unique_source_index.numel(),source_embeddings.shape[1]))
                source_mask=data.unique_source_index>=0
                unique[source_mask]=source_embeddings[data.unique_source_index[source_mask]]
                graph_mask=~source_mask
                if torch.any(graph_mask):
                    graph_h=self.mol_encoder(data.unique_node_graph)
                    unique[graph_mask]=graph_h[:int(graph_mask.sum())]
                molecular=unique[data.node_graph_inverse]
            else:
                molecular=self.mol_encoder(data.decoded.node_graph)
        else:molecular=molecular_embeddings
        edges=data.decoded.edge_index;added=data.decoded.edge_added_reaction_index
        reaction=reaction_h.new_zeros((added.numel(),reaction_h.shape[1]));normal=added>=0
        reaction[normal]=reaction_h[added[normal]]
        edge_h=self.edge_encoder(torch.cat((reaction,molecular[edges[0]],molecular[edges[1]]),1)) if edges.numel() else molecular.new_zeros((0,self.tree_encoder.hidden_dim))
        tree=data.tree_graph.clone();tree.x=molecular;tree.edge_attr=edge_h
        node_h,_=self.tree_encoder(tree)
        shift_index=data.explanation_hydrogen_shift-self.hydrogen_shift_min
        if torch.any((shift_index<0)|(shift_index>=self.hydrogen_shift_embedding.num_embeddings)):
            raise ValueError('Prepared hydrogen shift is outside the configured shared range')
        explanation=torch.cat((self.ion_embedding(data.explanation_adduct_index),
            self.hydrogen_shift_embedding(shift_index)),1)
        explanation=self.hydrogen_shift_encoder(explanation)
        counts=data.physical_candidate_explanation_ptr[1:]-data.physical_candidate_explanation_ptr[:-1]
        owner=torch.repeat_interleave(torch.arange(counts.numel(),device=node_h.device),counts)
        attention_weight=normalized_attention_weights(explanation,owner,counts.numel(),self.explanation_attention)
        physical=explanation.new_zeros((counts.numel(),explanation.shape[-1])).index_add_(0,owner,explanation*attention_weight[:,None])
        sample=data.physical_candidate_sample_index;fragment=node_h[data.physical_candidate_node_index]
        adduct=self.main_adduct_projection(main_adduct_h[sample]);energy=self.collision_energy_projection(collision_energy_h[sample])
        intensity_inputs=[fragment,physical,adduct,energy]
        if self.descriptor_indices:
            values=data.decoded.node_graph.descriptors[data.physical_candidate_node_index][:,self.descriptor_indices].to(fragment.dtype)
            intensity_inputs.append(torch.sign(values)*torch.log1p(values.abs()))
        intensity=F.softplus(self.intensity_head(torch.cat(intensity_inputs,1)).squeeze(-1))
        fragment_logit_by_node=self.fragment_score(node_h).squeeze(-1)
        all_adduct_logits=self.adduct_score(fragment)
        adduct_logit=all_adduct_logits[owner,data.explanation_adduct_index]
        hydrogen_shift_logit=self.hydrogen_shift_score(torch.cat((fragment[owner],explanation),1)).squeeze(-1)
        explanation_log_probability=F.logsigmoid(adduct_logit)+F.logsigmoid(hydrogen_shift_logit)
        state_log_probability=sum_explanation_log_probabilities(
            explanation_log_probability,data.physical_candidate_explanation_ptr)
        # A compatibility logit whose sigmoid is exactly P(A|F)P(ΔH|F,A).
        # Probability composition itself remains in log space above.
        state_logit=explanation_log_probability-torch.log(-torch.expm1(explanation_log_probability))
        physical_log_probability=F.logsigmoid(fragment_logit_by_node[data.physical_candidate_node_index])+state_log_probability
        sample_count=main_adduct_h.shape[0]
        maximum=intensity.new_zeros(sample_count).scatter_reduce_(0,sample,intensity,reduce='amax',include_self=True)
        normalized=intensity/maximum[sample].clamp_min(1e-8)
        node_group=data.decoded.node_sample_index
        roots=node_group.new_full((data.tree_graph.num_graphs,),node_group.numel())
        roots.scatter_reduce_(0,node_group,torch.arange(node_group.numel(),device=node_group.device),reduce='amin',include_self=True)
        candidate_group=node_group[data.physical_candidate_node_index]
        precursor=data.physical_candidate_node_index==roots[candidate_group]

        def spectrum_loss(keep):
            total=intensity.sum()*0
            if data.target_intensity is not None and torch.any(keep):
                masked=intensity*keep;peak_max=masked.new_zeros(sample_count).scatter_reduce_(0,sample,masked,reduce='amax',include_self=True)
                pred=(masked/peak_max[sample].clamp_min(1e-8))[keep].pow(loss_settings.intensity_power)
                target_raw=data.target_intensity.clamp_min(0)*keep
                target_max=target_raw.new_zeros(sample_count).scatter_reduce_(0,sample,target_raw,reduce='amax',include_self=True)
                target=(target_raw/target_max[sample].clamp_min(1e-8))[keep].pow(loss_settings.intensity_power)
                total=F.mse_loss(pred,target);group=sample[keep]
                dot=pred.new_zeros(sample_count).scatter_add_(0,group,pred*target)
                pn=pred.new_zeros(sample_count).scatter_add_(0,group,pred.square())
                tn=pred.new_zeros(sample_count).scatter_add_(0,group,target.square());observed=tn>0
                if torch.any(observed):total=total+loss_settings.cosine_loss_weight*(1-dot[observed]/(pn[observed]*tn[observed]).sqrt().clamp_min(1e-8)).mean()
            return total
        all_candidates=torch.ones_like(precursor);intensity_loss=spectrum_loss(all_candidates)
        if loss_settings.precursor_free_loss_weight>0:intensity_loss=intensity_loss+loss_settings.precursor_free_loss_weight*spectrum_loss(~precursor)
        if data.target_intensity is not None and (data.fragment_is_positive is None or data.explanation_is_positive is None):
            raise ValueError('Prepared data lacks independent fragment/state labels; regenerate the fragment-tree dataset')
        fragment_loss=fragment_logit_by_node.sum()*0;fragment_positive_loss=fragment_loss;fragment_negative_loss=fragment_loss
        fragment_metrics={}
        if data.fragment_is_positive is not None and data.fragment_supervision_node_index is not None:
            supervised_fragment_logit=fragment_logit_by_node[data.fragment_supervision_node_index]
            fragment_weight=data.fragment_positive_weight
            if fragment_weight is not None:fragment_weight=fragment_weight.pow(loss_settings.intensity_power)
            fragment_loss,fragment_positive_loss,fragment_negative_loss=balanced_binary_loss(
                supervised_fragment_logit,data.fragment_is_positive,fragment_weight)
            fragment_metrics=binary_metrics(supervised_fragment_logit,data.fragment_is_positive,'fragment')
        state_loss=state_logit.sum()*0;state_positive_loss=state_loss;state_negative_loss=state_loss;state_metrics={}
        state_labels=data.explanation_is_positive
        state_weights=data.explanation_positive_weight
        if state_labels is not None:
            powered=state_weights.pow(loss_settings.intensity_power) if state_weights is not None else None
            state_loss,state_positive_loss,state_negative_loss=balanced_binary_loss(state_logit,state_labels,powered)
            state_metrics=binary_metrics(state_logit,state_labels,'state')
        loss=loss_settings.fragment_loss_weight*fragment_loss+loss_settings.state_loss_weight*state_loss+loss_settings.intensity_loss_weight*intensity_loss
        def cosine_metric(keep):
            if data.target_intensity is None or not torch.any(keep):return loss*0
            pred=normalized[keep];target_raw=data.target_intensity.clamp_min(0)*keep
            target_max=target_raw.new_zeros(sample_count).scatter_reduce_(0,sample,target_raw,reduce='amax',include_self=True)
            target=(target_raw/target_max[sample].clamp_min(1e-8))[keep];group=sample[keep]
            dot=pred.new_zeros(sample_count).scatter_add_(0,group,pred*target)
            pn=pred.new_zeros(sample_count).scatter_add_(0,group,pred.square());tn=pred.new_zeros(sample_count).scatter_add_(0,group,target.square())
            observed=tn>0;value=dot/(pn*tn).sqrt().clamp_min(1e-8)
            return value[observed].mean() if torch.any(observed) else loss*0
        target_max=data.target_intensity.new_zeros(sample_count).scatter_reduce_(0,sample,data.target_intensity.clamp_min(0),reduce='amax',include_self=True) if data.target_intensity is not None else None
        normalized_target=data.target_intensity.clamp_min(0)/target_max[sample].clamp_min(1e-8) if data.target_intensity is not None else None
        metrics={'fragment_classification_loss':fragment_loss,'fragment_positive_loss':fragment_positive_loss,
            'fragment_negative_loss':fragment_negative_loss,'state_classification_loss':state_loss,
            'state_positive_loss':state_positive_loss,'state_negative_loss':state_negative_loss,
            'intensity_loss':intensity_loss,'intensity_mae':(normalized-normalized_target).abs().mean() if normalized_target is not None and normalized.numel() else loss*0,
            'full_spectrum_cosine':cosine_metric(all_candidates),'precursor_free_cosine':cosine_metric(~precursor),
            'mean_physical_probability':physical_log_probability.exp().mean() if physical_log_probability.numel() else loss*0,
            'num_physical_ion_candidates':intensity.new_tensor(float(intensity.numel())),
            'num_ion_explanations':intensity.new_tensor(float(explanation.shape[0])),
            'mean_explanations_per_physical_candidate':counts.float().mean() if counts.numel() else loss*0,
            'max_explanations_per_physical_candidate':counts.max().to(intensity.dtype) if counts.numel() else loss*0,
            **fragment_metrics,**state_metrics}
        for prefix in ('fragment','state'):
            for suffix in ('positive_count','negative_count','positive_probability_mean','negative_probability_mean','precision','recall','f1'):
                metrics.setdefault(f'{prefix}_{suffix}',loss*0)
        per_state_log_physical=F.logsigmoid(fragment_logit_by_node[data.physical_candidate_node_index[owner]])+explanation_log_probability
        return SpectrumModelOutput(normalized,sample,data.physical_candidate_formula_tensor,data.physical_candidate_mz,
            loss,physical_log_probability.exp(),physical_log_probability.exp(),metrics,
            F.logsigmoid(fragment_logit_by_node[data.physical_candidate_node_index]).exp(),
            state_log_probability.exp(),physical_log_probability,state_logit,per_state_log_physical,
            adduct_logit,hydrogen_shift_logit,explanation_log_probability)


def prepare_spectrum_model_batch(decoded:DecodedFragmentTreeBatch,fragmenter:Fragmenter,
                                 precursor_types:Sequence[Adduct],tensorizer:FormulaTensorizer,
                                 state_positive_weights:Sequence[dict[tuple[int,str],float]]|None=None,
                                 sample_branch_group_index:Tensor|None=None,
                                 ion_type_strs:Sequence[str]|None=None)->SpectrumModelBatch:
    """Run all formula/ion chemistry and physical-candidate grouping."""
    sample_count=len(precursor_types)
    sample_branch_group_index=torch.as_tensor(sample_branch_group_index if sample_branch_group_index is not None else torch.arange(sample_count),dtype=torch.long)
    ion_vocab=list(dict.fromkeys(ion_type_strs or [str(value) for rule in fragmenter.adduct_rule_set.adduct_rules for value in rule.fragment_ion_adducts]))
    ion_to_index={value:i for i,value in enumerate(ion_vocab)}
    graphs=[];nodes=[];samples=[];formulas=[];mzs=[];charges=[];explanation_rows=[];adducts=[]
    explanation_labels=[];explanation_weights=[];fragment_nodes=[];fragment_samples=[];fragment_labels=[];fragment_weights=[]
    node_offset=0
    for branch_group,tree in enumerate(decoded.trees):
        ion_tree=FragmentIonTree.from_fragment_tree(tree,fragment_ion_adduct_rule_set=fragmenter.adduct_rule_set,
            fragment_compounds=decoded.compounds[node_offset:node_offset+tree.num_nodes])
        group_samples=(sample_branch_group_index==branch_group).nonzero().flatten().tolist()
        if not group_samples:node_offset+=tree.num_nodes;continue
        main=fragmenter._resolve_main_adduct_type(precursor_types[group_samples[0]])
        formula_group=ion_tree.get_formula_candidate_group(main)
        local={}
        for formula_index,formula in enumerate(formula_group.formulas):
            for entry in range(formula_group.indptr[formula_index],formula_group.indptr[formula_index+1]):
                node=int(formula_group.node_indices[entry]);net=formula_group.candidate_adducts[formula_group.candidate_adduct_indices[entry]]
                ion=str(formula_group.fragment_ion_adducts[formula_group.fragment_ion_adduct_positions[entry]]);hydrogen_shift=int(formula_group.hydrogen_shifts[entry])
                key=(node,str(formula.normalized),str(net),int(net.charge))
                local.setdefault(key,dict(node=node,formula=formula,net=net,explanations=[]))['explanations'].append((ion_to_index[ion],hydrogen_shift))
        edge_store=tree.edge_store
        graphs.append(Data(x=torch.zeros((tree.num_nodes,1)),edge_index=torch.stack((torch.as_tensor(edge_store.source_indices),torch.as_tensor(edge_store.target_indices)))))
        for sample in group_samples:
            supervision=state_positive_weights[sample] if state_positive_weights is not None else None
            def peak_weight(key):
                if supervision is None:return 0.
                return float(supervision.get(key,0.)) if hasattr(supervision,'get') else float(key in supervision)
            for local_node in range(tree.num_nodes):
                global_node=node_offset+local_node
                matching=[key for key in supervision if key[0]==global_node] if supervision is not None else []
                node_weight=max((peak_weight(key) for key in matching),default=0.)
                fragment_nodes.append(global_node);fragment_samples.append(sample);fragment_labels.append(bool(matching));fragment_weights.append(node_weight)
            for item in local.values():
                nodes.append(node_offset+item['node']);samples.append(sample);formulas.append(item['formula']);mzs.append(item['formula'].exact_mass)
                charges.append(item['net'].charge);adducts.append(str(item['net']));row=tuple(dict.fromkeys(item['explanations']));explanation_rows.append(row)
                if supervision is not None:
                    key=(node_offset+item['node'],str(item['net']));weight=peak_weight(key);positive=key in supervision
                    explanation_labels.extend([positive]*len(row));explanation_weights.extend([weight]*len(row))
        node_offset+=tree.num_nodes
    explanation_ptr,flat=__import__('fredis.ml.input.source_reaction_structure',fromlist=['csr']).csr(explanation_rows)
    pairs=flat.reshape(-1,2) if flat.numel() else torch.empty((0,2),dtype=torch.long)
    return SpectrumModelBatch(decoded=decoded,tree_graph=Batch.from_data_list(graphs),physical_candidate_node_index=torch.tensor(nodes,dtype=torch.long),physical_candidate_sample_index=torch.tensor(samples,dtype=torch.long),
        physical_candidate_formula_tensor=tensorizer.formulas_to_tensor(formulas),physical_candidate_mz=torch.tensor(mzs,dtype=torch.float64),physical_candidate_charge=torch.tensor(charges,dtype=torch.long),physical_candidate_explanation_ptr=explanation_ptr,
        explanation_adduct_index=pairs[:,0],explanation_hydrogen_shift=pairs[:,1],
        fragment_supervision_node_index=torch.tensor(fragment_nodes,dtype=torch.long) if state_positive_weights is not None else None,
        fragment_supervision_sample_index=torch.tensor(fragment_samples,dtype=torch.long) if state_positive_weights is not None else None,
        fragment_is_positive=torch.tensor(fragment_labels,dtype=torch.bool) if state_positive_weights is not None else None,
        fragment_positive_weight=torch.tensor(fragment_weights,dtype=torch.float32) if state_positive_weights is not None else None,
        explanation_is_positive=torch.tensor(explanation_labels,dtype=torch.bool) if state_positive_weights is not None else None,
        explanation_positive_weight=torch.tensor(explanation_weights,dtype=torch.float32) if state_positive_weights is not None else None,
        physical_candidate_adduct=tuple(adducts),ion_type_strs=tuple(ion_vocab),node_smiles=tuple(c.smiles for c in decoded.compounds))


def collate_spectrum_model_batches(items:Sequence[SpectrumModelBatch],reaction_offsets:Sequence[int])->SpectrumModelBatch:
    molecules=[];trees=[];compounds=[];metadata=[];edges=[];node_groups=[];edge_reactions=[];terminals=[];scores=[]
    nodes=[];samples=[];formulas=[];mzs=[];charges=[];explanations=[];targets=[];adducts=[];node_smiles=[]
    fragment_nodes=[];fragment_samples=[];fragment_labels=[];fragment_weights=[];explanation_labels=[];explanation_weights=[]
    node_offset=sample_offset=0
    ion_vocab=list(dict.fromkeys(value for item in items for value in item.ion_type_strs));ion_index={v:i for i,v in enumerate(ion_vocab)}
    for item,reaction_offset in zip(items,reaction_offsets):
        d=item.decoded;molecules.extend(d.node_graph.to_data_list());trees.extend(item.tree_graph.to_data_list());compounds.extend(d.compounds);metadata.extend(d.trees)
        edges.append(d.edge_index+node_offset);node_groups.append(d.node_sample_index+len(metadata)-len(d.trees));edge_reactions.append(torch.where(d.edge_added_reaction_index>=0,d.edge_added_reaction_index+reaction_offset,d.edge_added_reaction_index))
        terminals.append(d.terminal_node_index+node_offset);scores.append(d.node_log_score)
        nodes.append(item.physical_candidate_node_index+node_offset);samples.append(item.physical_candidate_sample_index+sample_offset);formulas.append(item.physical_candidate_formula_tensor);mzs.append(item.physical_candidate_mz);charges.append(item.physical_candidate_charge)
        for candidate in range(item.physical_candidate_node_index.numel()):
            start,stop=item.physical_candidate_explanation_ptr[candidate:candidate+2]
            explanations.append([(ion_index[item.ion_type_strs[int(i)]],int(hydrogen_shift)) for i,hydrogen_shift in zip(item.explanation_adduct_index[start:stop],item.explanation_hydrogen_shift[start:stop])])
        if item.target_intensity is not None:targets.append(item.target_intensity)
        if item.fragment_supervision_node_index is not None:
            fragment_nodes.append(item.fragment_supervision_node_index+node_offset)
            fragment_samples.append(item.fragment_supervision_sample_index+sample_offset)
            fragment_labels.append(item.fragment_is_positive);fragment_weights.append(item.fragment_positive_weight)
        if item.explanation_is_positive is not None:
            explanation_labels.append(item.explanation_is_positive);explanation_weights.append(item.explanation_positive_weight)
        adducts.extend(item.physical_candidate_adduct);node_smiles.extend(item.node_smiles);node_offset+=len(d.compounds);sample_offset+=int(item.physical_candidate_sample_index.max())+1 if item.physical_candidate_sample_index.numel() else 0
    from ..input.source_reaction_structure import csr
    ptr,flat=csr(explanations);pairs=flat.reshape(-1,2) if flat.numel() else torch.empty((0,2),dtype=torch.long)
    decoded=DecodedFragmentTreeBatch(Batch.from_data_list(molecules),torch.cat(edges,1),torch.cat(node_groups),torch.cat(edge_reactions),torch.cat(terminals),torch.cat(scores),tuple(metadata),tuple(compounds),sum(i.decoded.rdkit_run_count for i in items),sum(i.decoded.failed_effect_count for i in items))
    return SpectrumModelBatch(decoded=decoded,tree_graph=Batch.from_data_list(trees),physical_candidate_node_index=torch.cat(nodes),physical_candidate_sample_index=torch.cat(samples),physical_candidate_formula_tensor=torch.cat(formulas),physical_candidate_mz=torch.cat(mzs),physical_candidate_charge=torch.cat(charges),physical_candidate_explanation_ptr=ptr,
        explanation_adduct_index=pairs[:,0],explanation_hydrogen_shift=pairs[:,1],
        target_intensity=torch.cat(targets) if targets else None,
        fragment_supervision_node_index=torch.cat(fragment_nodes) if fragment_nodes else None,
        fragment_supervision_sample_index=torch.cat(fragment_samples) if fragment_samples else None,
        fragment_is_positive=torch.cat(fragment_labels) if fragment_labels else None,
        fragment_positive_weight=torch.cat(fragment_weights) if fragment_weights else None,
        explanation_is_positive=torch.cat(explanation_labels) if explanation_labels else None,
        explanation_positive_weight=torch.cat(explanation_weights) if explanation_weights else None,
        physical_candidate_adduct=tuple(adducts),ion_type_strs=tuple(ion_vocab),node_smiles=tuple(node_smiles))


def deduplicate_molecular_graphs(data:SpectrumModelBatch,source_smiles=()):
    graphs=data.decoded.node_graph.to_data_list();source_index={s:i for i,s in enumerate(source_smiles)};keys={};unique=[];inverse=[];shared=[]
    if len(data.node_smiles)!=len(graphs):raise ValueError('Molecular identities must be prepared before training')
    for graph,key in zip(graphs,data.node_smiles):
        if key not in keys:
            keys[key]=len(shared);shared.append(source_index.get(key,-1))
            if key not in source_index:unique.append(graph)
        inverse.append(keys[key])
    unique_batch=Batch.from_data_list(unique if unique else [graphs[0]])
    return replace(data,unique_node_graph=unique_batch,node_graph_inverse=torch.tensor(inverse),unique_source_index=torch.tensor(shared),unique_smiles=tuple(keys))
