"""Slice prepared spectra while retaining each shared branch group once."""
from dataclasses import replace
import torch
from torch_geometric.data import Batch
from .source_reaction_structure import csr


def select_samples(data,indices):
    indices=torch.as_tensor(indices,dtype=torch.long,device=data.device)
    if data.device.type!='cpu':raise ValueError('Slice structures before transfer to CUDA')
    sample_map=torch.full((data.num_samples,),-1,dtype=torch.long);sample_map[indices]=torch.arange(indices.numel())
    old_groups=torch.unique(data.sample_branch_group_index[indices],sorted=True)
    group_map=torch.full((data.num_branch_groups,),-1,dtype=torch.long);group_map[old_groups]=torch.arange(old_groups.numel())
    state_ids=(group_map[data.teacher_node_branch_group_index]>=0).nonzero().flatten()
    state_map=torch.full((data.teacher_node_branch_group_index.numel(),),-1,dtype=torch.long);state_map[state_ids]=torch.arange(state_ids.numel())
    transition_state_ids=(group_map[data.transition_state_branch_group_index]>=0).nonzero().flatten()
    transition_state_map=torch.full((data.transition_state_branch_group_index.numel(),),-1,dtype=torch.long)
    transition_state_map[transition_state_ids]=torch.arange(transition_state_ids.numel())
    def rows(ptr,values,ids,offset=0):return csr([(values[ptr[i]:ptr[i+1]]+offset).tolist() for i in ids.tolist()])
    state_ptr,state_index=rows(data.teacher_node_reaction_ptr,data.teacher_node_reaction_index,state_ids)
    positive_ptr,positive_index=rows(data.teacher_positive_reaction_ptr,data.teacher_positive_reaction_index,state_ids)
    valid_ptr,valid_reactions=rows(data.state_transition_ptr,data.state_transition_reaction_index,state_ids)
    valid_next=[]
    for state in state_ids.tolist():
        a,b=data.state_transition_ptr[state:state+2];values=data.state_transition_next_state_index[a:b]
        valid_next.extend(transition_state_map[values].tolist())
    transition_state_ptr,transition_state_reactions=rows(data.transition_state_reaction_ptr,data.transition_state_reaction_index,transition_state_ids)
    negative_ptr,negative_reactions=rows(data.weak_negative_reaction_ptr,data.weak_negative_reaction_index,state_ids)
    transitions=(state_map[data.transition_parent_state_index]>=0)&(state_map[data.transition_child_state_index]>=0)
    peak_ids=(group_map[data.teacher_peak_branch_group_index]>=0).nonzero().flatten();paths=[];steps=[];peak_groups=[]
    for peak in peak_ids.tolist():
        peak_groups.append(int(group_map[data.teacher_peak_branch_group_index[peak]]));paths.append([])
        for path in range(int(data.teacher_peak_path_ptr[peak]),int(data.teacher_peak_path_ptr[peak+1])):
            paths[-1].append(len(steps));a,b=data.teacher_path_step_ptr[path:path+2]
            steps.append([(int(state_map[s]),int(reaction)) for s,reaction in zip(data.teacher_path_step_state_index[a:b],data.teacher_path_step_reaction_index[a:b])])
    peak_ptr,_=csr(paths);step_ptr,step_states=csr([[s for s,_ in row] for row in steps]);step_reactions=torch.tensor([a for row in steps for _,a in row],dtype=torch.long)
    annotations=tuple(data.sample_annotations[i] for i in indices.tolist()) if data.sample_annotations else ()
    spectrum_batch=data.spectrum_batch;state_nodes=data.state_fragment_node_index[state_ids]
    if spectrum_batch is not None:
        decoded=spectrum_batch.decoded;node_ids=(group_map[decoded.node_sample_index]>=0).nonzero().flatten()
        node_map=torch.full((len(decoded.compounds),),-1,dtype=torch.long);node_map[node_ids]=torch.arange(node_ids.numel())
        edge_keep=(node_map[decoded.edge_index[0]]>=0)&(node_map[decoded.edge_index[1]]>=0)
        edge_map=torch.full((edge_keep.numel(),),-1,dtype=torch.long);edge_map[edge_keep]=torch.arange(int(edge_keep.sum()))
        terminal=decoded.terminal_node_index[node_map[decoded.terminal_node_index]>=0]
        subset=replace(decoded,node_graph=Batch.from_data_list([decoded.node_graph.to_data_list()[i] for i in node_ids.tolist()]),
            edge_index=node_map[decoded.edge_index[:,edge_keep]],node_sample_index=group_map[decoded.node_sample_index[node_ids]],
            edge_added_reaction_index=decoded.edge_added_reaction_index[edge_keep],terminal_node_index=node_map[terminal],
            node_log_score=decoded.node_log_score[node_ids],trees=tuple(decoded.trees[i] for i in old_groups.tolist()),
            compounds=tuple(decoded.compounds[i] for i in node_ids.tolist()),
            materialized_state_index=torch.empty(0,dtype=torch.long),materialized_node_index=torch.empty(0,dtype=torch.long))
        candidates=(sample_map[spectrum_batch.physical_candidate_sample_index]>=0).nonzero().flatten();explanations=[];explanation_ids=[]
        for candidate in candidates.tolist():
            a,b=spectrum_batch.physical_candidate_explanation_ptr[candidate:candidate+2]
            explanation_ids.extend(range(int(a),int(b)))
            explanations.append(list(zip(spectrum_batch.explanation_adduct_index[a:b].tolist(),spectrum_batch.explanation_hydrogen_shift[a:b].tolist())))
        explanation_ptr,flat=csr(explanations);pairs=flat.reshape(-1,2) if flat.numel() else torch.empty((0,2),dtype=torch.long)
        explanation_ids=torch.tensor(explanation_ids,dtype=torch.long)
        fragment_ids=(sample_map[spectrum_batch.fragment_supervision_sample_index]>=0).nonzero().flatten() if spectrum_batch.fragment_supervision_sample_index is not None else None
        spectrum_batch=replace(spectrum_batch,decoded=subset,tree_graph=Batch.from_data_list([spectrum_batch.tree_graph.to_data_list()[i] for i in old_groups.tolist()]),
            physical_candidate_node_index=node_map[spectrum_batch.physical_candidate_node_index[candidates]],
            physical_candidate_sample_index=sample_map[spectrum_batch.physical_candidate_sample_index[candidates]],
            physical_candidate_formula_tensor=spectrum_batch.physical_candidate_formula_tensor[candidates],physical_candidate_mz=spectrum_batch.physical_candidate_mz[candidates],
            physical_candidate_charge=spectrum_batch.physical_candidate_charge[candidates],physical_candidate_explanation_ptr=explanation_ptr,
            explanation_adduct_index=pairs[:,0],explanation_hydrogen_shift=pairs[:,1],
            target_intensity=spectrum_batch.target_intensity[candidates] if spectrum_batch.target_intensity is not None else None,
            fragment_supervision_node_index=node_map[spectrum_batch.fragment_supervision_node_index[fragment_ids]] if fragment_ids is not None else None,
            fragment_supervision_sample_index=sample_map[spectrum_batch.fragment_supervision_sample_index[fragment_ids]] if fragment_ids is not None else None,
            fragment_is_positive=spectrum_batch.fragment_is_positive[fragment_ids] if fragment_ids is not None else None,
            fragment_positive_weight=spectrum_batch.fragment_positive_weight[fragment_ids] if fragment_ids is not None else None,
            explanation_is_positive=spectrum_batch.explanation_is_positive[explanation_ids] if spectrum_batch.explanation_is_positive is not None else None,
            explanation_positive_weight=spectrum_batch.explanation_positive_weight[explanation_ids] if spectrum_batch.explanation_positive_weight is not None else None,
            physical_candidate_adduct=tuple(spectrum_batch.physical_candidate_adduct[i] for i in candidates.tolist()),unique_node_graph=None,node_graph_inverse=None,unique_source_index=None,
            node_smiles=tuple(spectrum_batch.node_smiles[i] for i in node_ids.tolist()))
        state_nodes=torch.where(state_nodes>=0,node_map[state_nodes.clamp_min(0)],state_nodes)
    return replace(data,sample_tree_index=data.sample_tree_index[indices],condition_features=data.condition_features[indices],
        sample_branch_group_index=group_map[data.sample_branch_group_index[indices]],branch_group_tree_index=data.branch_group_tree_index[old_groups],branch_group_adduct_index=data.branch_group_adduct_index[old_groups],
        teacher_node_branch_group_index=group_map[data.teacher_node_branch_group_index[state_ids]],teacher_node_reaction_ptr=state_ptr,teacher_node_reaction_index=state_index,
        teacher_positive_reaction_ptr=positive_ptr,teacher_positive_reaction_index=positive_index,teacher_node_observed=data.teacher_node_observed[state_ids],
        teacher_node_parent_index=torch.where(data.teacher_node_parent_index[state_ids]>=0,state_map[data.teacher_node_parent_index[state_ids].clamp_min(0)],data.teacher_node_parent_index[state_ids]),
        teacher_node_added_reaction_index=data.teacher_node_added_reaction_index[state_ids],teacher_node_reaction_count=data.teacher_node_reaction_count[state_ids],
        transition_parent_state_index=state_map[data.transition_parent_state_index[transitions]],transition_child_state_index=state_map[data.transition_child_state_index[transitions]],transition_added_reaction_index=data.transition_added_reaction_index[transitions],
        state_transition_ptr=valid_ptr,state_transition_reaction_index=valid_reactions,state_transition_next_state_index=torch.tensor(valid_next,dtype=torch.long),weak_negative_reaction_ptr=negative_ptr,weak_negative_reaction_index=negative_reactions,
        transition_state_branch_group_index=group_map[data.transition_state_branch_group_index[transition_state_ids]],
        transition_state_reaction_ptr=transition_state_ptr,transition_state_reaction_index=transition_state_reactions,
        teacher_node_transition_state_index=transition_state_map[data.teacher_node_transition_state_index[state_ids]],
        teacher_peak_path_ptr=peak_ptr,teacher_peak_branch_group_index=torch.tensor(peak_groups,dtype=torch.long),teacher_path_step_ptr=step_ptr,teacher_path_step_state_index=step_states,teacher_path_step_reaction_index=step_reactions,
        state_fragment_node_index=state_nodes,spectrum_batch=spectrum_batch,sample_annotations=annotations)
