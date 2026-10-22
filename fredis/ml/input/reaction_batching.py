"""Select spectra while retaining shared reaction and fragment graphs once.

Samples are selected in the requested order. Shared branch groups, states and
nodes retain their original order, with references remapped to compact indices.
Reaction IDs still refer to the original source reaction pool.
"""

from dataclasses import replace

import torch
from torch import Tensor
from torch_geometric.data import Batch

from .source_reaction_structure import SourceReactionStructure, csr


def _index_map(original_size: int, selected_indices: Tensor) -> Tensor:
    """Map original indices to subset indices; excluded entries remain -1."""
    mapping = torch.full((original_size,), -1, dtype=torch.long)
    mapping[selected_indices] = torch.arange(selected_indices.numel())
    return mapping


def _included_indices(owners: Tensor, owner_map: Tensor) -> Tensor:
    return (owner_map[owners] >= 0).nonzero().flatten()


def _select_csr_rows(ptr: Tensor, values: Tensor, row_indices: Tensor) -> tuple[Tensor, Tensor]:
    rows = [values[ptr[row] : ptr[row + 1]].tolist() for row in row_indices.tolist()]
    return csr(rows)


def _select_teacher_paths(
    data: SourceReactionStructure,
    peak_indices: Tensor,
    group_map: Tensor,
    state_map: Tensor,
) -> dict[str, Tensor]:
    """Rebuild peak -> path -> step CSR tables for retained teacher peaks."""
    paths_by_peak = []
    steps_by_path = []
    peak_groups = []

    for peak in peak_indices.tolist():
        peak_groups.append(int(group_map[data.teacher.peak_branch_group_index[peak]]))
        selected_paths = []
        first_path, last_path = data.teacher.peak_path_ptr[peak : peak + 2]
        for path in range(int(first_path), int(last_path)):
            selected_paths.append(len(steps_by_path))
            start, stop = data.teacher.path_step_ptr[path : path + 2]
            states = data.teacher.path_step_state_index[start:stop]
            reactions = data.teacher.path_step_reaction_index[start:stop]
            steps_by_path.append(
                [
                    (int(state_map[state]), int(reaction))
                    for state, reaction in zip(states, reactions)
                ]
            )
        paths_by_peak.append(selected_paths)

    peak_ptr, _ = csr(paths_by_peak)
    step_ptr, step_states = csr([[state for state, _ in steps] for steps in steps_by_path])
    step_reactions = torch.tensor(
        [reaction for steps in steps_by_path for _, reaction in steps],
        dtype=torch.long,
    )
    return {
        "peak_path_ptr": peak_ptr,
        "peak_branch_group_index": torch.tensor(peak_groups, dtype=torch.long),
        "path_step_ptr": step_ptr,
        "path_step_state_index": step_states,
        "path_step_reaction_index": step_reactions,
    }


def _select_spectrum_batch(spectrum_batch, sample_indices, sample_map, group_indices, group_map):
    """Select fragment graphs, ion candidates and their supervision together."""
    decoded = spectrum_batch.decoded
    node_indices = _included_indices(decoded.node_sample_index, group_map)
    node_map = _index_map(len(decoded.compounds), node_indices)
    edge_keep = (node_map[decoded.edge_index[0]] >= 0) & (node_map[decoded.edge_index[1]] >= 0)
    terminal_indices = decoded.terminal_node_index[node_map[decoded.terminal_node_index] >= 0]
    node_graphs = decoded.node_graph.to_data_list()
    selected_decoded = replace(
        decoded,
        node_graph=Batch.from_data_list([node_graphs[i] for i in node_indices.tolist()]),
        edge_index=node_map[decoded.edge_index[:, edge_keep]],
        node_sample_index=group_map[decoded.node_sample_index[node_indices]],
        edge_added_reaction_index=decoded.edge_added_reaction_index[edge_keep],
        terminal_node_index=node_map[terminal_indices],
        node_log_score=decoded.node_log_score[node_indices],
        trees=tuple(decoded.trees[i] for i in group_indices.tolist()),
        compounds=tuple(decoded.compounds[i] for i in node_indices.tolist()),
        # Decoder-state bookkeeping is not retained across sample slicing.
        materialized_state_index=torch.empty(0, dtype=torch.long),
        materialized_node_index=torch.empty(0, dtype=torch.long),
    )

    candidate_indices = _included_indices(spectrum_batch.state_sample_index, sample_map)

    def select_optional(values, indices):
        return values[indices] if values is not None else None

    selected_batch = replace(
        spectrum_batch,
        decoded=selected_decoded,
        state_node_index=node_map[spectrum_batch.state_node_index[candidate_indices]],
        state_sample_index=sample_map[spectrum_batch.state_sample_index[candidate_indices]],
        state_formula_tensor=spectrum_batch.state_formula_tensor[candidate_indices],
        state_mz=spectrum_batch.state_mz[candidate_indices],
        state_adduct_index=spectrum_batch.state_adduct_index[candidate_indices],
        state_hydrogen_shift=spectrum_batch.state_hydrogen_shift[candidate_indices],
        target_intensity=select_optional(spectrum_batch.target_intensity, candidate_indices),
        precursor_adducts=(
            tuple(spectrum_batch.precursor_adducts[i] for i in sample_indices.tolist())
            if getattr(spectrum_batch, "precursor_adducts", ())
            else ()
        ),
        node_smiles=tuple(spectrum_batch.node_smiles[i] for i in node_indices.tolist()),
        # Molecular deduplication must be rebuilt for the selected graphs.
        unique_node_graph=None,
        node_graph_inverse=None,
        unique_source_index=None,
    )
    return selected_batch, node_map


def select_samples(data: SourceReactionStructure, indices) -> SourceReactionStructure:
    """Select CPU samples and remap references into their shared graph subsets."""
    indices = torch.as_tensor(indices, dtype=torch.long, device=data.device)
    if data.device.type != "cpu":
        raise ValueError("Slice structures before transfer to CUDA")

    # Each index space needs its own original -> subset mapping.
    sample_map = _index_map(data.num_samples, indices)
    group_indices = torch.unique(data.sample_branch_group_index[indices], sorted=True)
    group_map = _index_map(data.num_branch_groups, group_indices)
    state_indices = _included_indices(data.states.branch_group_index, group_map)
    state_map = _index_map(data.states.branch_group_index.numel(), state_indices)
    transition_state_indices = _included_indices(
        data.states.candidate_branch_group_index, group_map
    )
    transition_state_map = _index_map(
        data.states.candidate_branch_group_index.numel(), transition_state_indices
    )

    positive_ptr, positive_reactions = _select_csr_rows(
        data.teacher.positive_reaction_ptr, data.teacher.positive_reaction_index, state_indices
    )
    valid_ptr, valid_reactions = _select_csr_rows(
        data.states.next_ptr, data.states.next_reaction_index, state_indices
    )
    valid_next_states = []
    for state in state_indices.tolist():
        start, stop = data.states.next_ptr[state : state + 2]
        next_states = data.states.next_state_index[start:stop]
        valid_next_states.extend(transition_state_map[next_states].tolist())
    negative_ptr, negative_reactions = _select_csr_rows(
        data.teacher.negative_reaction_ptr, data.teacher.negative_reaction_index, state_indices
    )
    transition_keep = (state_map[data.states.transition_parent_index] >= 0) & (
        state_map[data.states.transition_child_index] >= 0
    )
    peak_indices = _included_indices(data.teacher.peak_branch_group_index, group_map)
    observed_peak_map = None
    if data.annotations is not None:
        ptr = data.annotations.spectra.peak_ptr
        selected = torch.cat([torch.arange(ptr[i], ptr[i + 1]) for i in indices.tolist()])
        observed_peak_map = _index_map(data.annotations.spectra.mz.numel(), selected)
        peak_indices = peak_indices[observed_peak_map[data.teacher.peak_index[peak_indices]] >= 0]
    teacher_paths = _select_teacher_paths(data, peak_indices, group_map, state_map)
    teacher_paths["peak_index"] = (
        observed_peak_map[data.teacher.peak_index[peak_indices]]
        if observed_peak_map is not None
        else data.teacher.peak_index[peak_indices]
    )

    spectrum_batch = data.spectrum_batch
    state_nodes = data.states.fragment_node_index[state_indices]
    if spectrum_batch is not None:
        spectrum_batch, node_map = _select_spectrum_batch(
            spectrum_batch, indices, sample_map, group_indices, group_map
        )
        state_nodes = torch.where(state_nodes >= 0, node_map[state_nodes.clamp_min(0)], state_nodes)

    parent_states = data.states.parent_index[state_indices]
    annotations = (
        tuple(data.sample_annotations[i] for i in indices.tolist())
        if data.sample_annotations
        else ()
    )
    if spectrum_batch is not None and annotations:
        annotations = tuple(
            {
                **sample,
                "peaks": [
                    {
                        **peak,
                        "matches": [
                            {
                                **match,
                                "nodeIndices": [
                                    int(node_map[node])
                                    for node in match["nodeIndices"]
                                    if int(node_map[node]) >= 0
                                ],
                            }
                            for match in peak["matches"]
                        ],
                    }
                    for peak in sample["peaks"]
                ],
            }
            for sample in annotations
        )
    return replace(
        data,
        sample_tree_index=data.sample_tree_index[indices],
        collision_energy_eV=data.collision_energy_eV[indices],
        sample_branch_group_index=group_map[data.sample_branch_group_index[indices]],
        branch_group_tree_index=data.branch_group_tree_index[group_indices],
        branch_group_adduct_index=data.branch_group_adduct_index[group_indices],
        spectrum_batch=spectrum_batch,
        observations=annotations,
        states=replace(
            data.states,
            candidate_fragment_index=(
                torch.where(
                    data.states.candidate_fragment_index[transition_state_indices] >= 0,
                    node_map[
                        data.states.candidate_fragment_index[transition_state_indices].clamp_min(0)
                    ],
                    data.states.candidate_fragment_index[transition_state_indices],
                )
                if spectrum_batch is not None
                else data.states.candidate_fragment_index[transition_state_indices]
            ),
            branch_group_index=group_map[data.states.branch_group_index[state_indices]],
            parent_index=torch.where(
                parent_states >= 0, state_map[parent_states.clamp_min(0)], parent_states
            ),
            added_reaction_index=data.states.added_reaction_index[state_indices],
            transition_parent_index=state_map[data.states.transition_parent_index[transition_keep]],
            transition_child_index=state_map[data.states.transition_child_index[transition_keep]],
            transition_reaction_index=data.states.transition_reaction_index[transition_keep],
            next_ptr=valid_ptr,
            next_reaction_index=valid_reactions,
            next_state_index=torch.tensor(valid_next_states, dtype=torch.long),
            candidate_branch_group_index=group_map[
                data.states.candidate_branch_group_index[transition_state_indices]
            ],
            candidate_reaction_set_index=data.states.candidate_reaction_set_index[transition_state_indices],
            candidate_index=transition_state_map[data.states.candidate_index[state_indices]],
            fragment_node_index=state_nodes,
        ),
        teacher=replace(
            data.teacher,
            positive_reaction_ptr=positive_ptr,
            positive_reaction_index=positive_reactions,
            observed=data.teacher.observed[state_indices],
            negative_reaction_ptr=negative_ptr,
            negative_reaction_index=negative_reactions,
            **teacher_paths,
        ),
    )
