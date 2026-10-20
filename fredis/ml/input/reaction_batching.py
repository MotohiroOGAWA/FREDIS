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


def _select_csr_rows(
    ptr: Tensor, values: Tensor, row_indices: Tensor
) -> tuple[Tensor, Tensor]:
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
        peak_groups.append(int(group_map[data.teacher_peak_branch_group_index[peak]]))
        selected_paths = []
        first_path, last_path = data.teacher_peak_path_ptr[peak : peak + 2]
        for path in range(int(first_path), int(last_path)):
            selected_paths.append(len(steps_by_path))
            start, stop = data.teacher_path_step_ptr[path : path + 2]
            states = data.teacher_path_step_state_index[start:stop]
            reactions = data.teacher_path_step_reaction_index[start:stop]
            steps_by_path.append(
                [
                    (int(state_map[state]), int(reaction))
                    for state, reaction in zip(states, reactions)
                ]
            )
        paths_by_peak.append(selected_paths)

    peak_ptr, _ = csr(paths_by_peak)
    step_ptr, step_states = csr(
        [[state for state, _ in steps] for steps in steps_by_path]
    )
    step_reactions = torch.tensor(
        [reaction for steps in steps_by_path for _, reaction in steps],
        dtype=torch.long,
    )
    return {
        "teacher_peak_path_ptr": peak_ptr,
        "teacher_peak_branch_group_index": torch.tensor(peak_groups, dtype=torch.long),
        "teacher_path_step_ptr": step_ptr,
        "teacher_path_step_state_index": step_states,
        "teacher_path_step_reaction_index": step_reactions,
    }


def _select_spectrum_batch(
    spectrum_batch, sample_indices, sample_map, group_indices, group_map
):
    """Select fragment graphs, ion candidates and their supervision together."""
    decoded = spectrum_batch.decoded
    node_indices = _included_indices(decoded.node_sample_index, group_map)
    node_map = _index_map(len(decoded.compounds), node_indices)
    edge_keep = (node_map[decoded.edge_index[0]] >= 0) & (
        node_map[decoded.edge_index[1]] >= 0
    )
    terminal_indices = decoded.terminal_node_index[
        node_map[decoded.terminal_node_index] >= 0
    ]
    node_graphs = decoded.node_graph.to_data_list()
    selected_decoded = replace(
        decoded,
        node_graph=Batch.from_data_list(
            [node_graphs[i] for i in node_indices.tolist()]
        ),
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

    candidate_indices = _included_indices(
        spectrum_batch.physical_candidate_sample_index, sample_map
    )
    explanations = []
    explanation_indices = []
    for candidate in candidate_indices.tolist():
        start, stop = spectrum_batch.physical_candidate_explanation_ptr[
            candidate : candidate + 2
        ]
        explanation_indices.extend(range(int(start), int(stop)))
        explanations.append(
            list(
                zip(
                    spectrum_batch.explanation_adduct_index[start:stop].tolist(),
                    spectrum_batch.explanation_hydrogen_shift[start:stop].tolist(),
                )
            )
        )
    explanation_ptr, flat_explanations = csr(explanations)
    explanation_pairs = (
        flat_explanations.reshape(-1, 2)
        if flat_explanations.numel()
        else torch.empty((0, 2), dtype=torch.long)
    )
    explanation_indices = torch.tensor(explanation_indices, dtype=torch.long)

    def select_optional(values, indices):
        return values[indices] if values is not None else None

    selected_batch = replace(
        spectrum_batch,
        decoded=selected_decoded,
        physical_candidate_node_index=node_map[
            spectrum_batch.physical_candidate_node_index[candidate_indices]
        ],
        physical_candidate_sample_index=sample_map[
            spectrum_batch.physical_candidate_sample_index[candidate_indices]
        ],
        physical_candidate_formula_tensor=spectrum_batch.physical_candidate_formula_tensor[
            candidate_indices
        ],
        physical_candidate_mz=spectrum_batch.physical_candidate_mz[candidate_indices],
        physical_candidate_charge=spectrum_batch.physical_candidate_charge[
            candidate_indices
        ],
        physical_candidate_explanation_ptr=explanation_ptr,
        explanation_adduct_index=explanation_pairs[:, 0],
        explanation_hydrogen_shift=explanation_pairs[:, 1],
        target_intensity=select_optional(
            spectrum_batch.target_intensity, candidate_indices
        ),
        precursor_adducts=(
            tuple(spectrum_batch.precursor_adducts[i] for i in sample_indices.tolist())
            if getattr(spectrum_batch, "precursor_adducts", ())
            else ()
        ),
        physical_candidate_adduct=tuple(
            spectrum_batch.physical_candidate_adduct[i]
            for i in candidate_indices.tolist()
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
    state_indices = _included_indices(data.teacher_node_branch_group_index, group_map)
    state_map = _index_map(data.teacher_node_branch_group_index.numel(), state_indices)
    transition_state_indices = _included_indices(
        data.transition_state_branch_group_index, group_map
    )
    transition_state_map = _index_map(
        data.transition_state_branch_group_index.numel(), transition_state_indices
    )

    state_ptr, state_reactions = _select_csr_rows(
        data.teacher_node_reaction_ptr, data.teacher_node_reaction_index, state_indices
    )
    positive_ptr, positive_reactions = _select_csr_rows(
        data.teacher_positive_reaction_ptr,
        data.teacher_positive_reaction_index,
        state_indices,
    )
    valid_ptr, valid_reactions = _select_csr_rows(
        data.state_transition_ptr, data.state_transition_reaction_index, state_indices
    )
    valid_next_states = []
    for state in state_indices.tolist():
        start, stop = data.state_transition_ptr[state : state + 2]
        next_states = data.state_transition_next_state_index[start:stop]
        valid_next_states.extend(transition_state_map[next_states].tolist())
    transition_state_ptr, transition_state_reactions = _select_csr_rows(
        data.transition_state_reaction_ptr,
        data.transition_state_reaction_index,
        transition_state_indices,
    )
    negative_ptr, negative_reactions = _select_csr_rows(
        data.weak_negative_reaction_ptr,
        data.weak_negative_reaction_index,
        state_indices,
    )
    transition_keep = (state_map[data.transition_parent_state_index] >= 0) & (
        state_map[data.transition_child_state_index] >= 0
    )
    peak_indices = _included_indices(data.teacher_peak_branch_group_index, group_map)
    teacher_paths = _select_teacher_paths(data, peak_indices, group_map, state_map)

    spectrum_batch = data.spectrum_batch
    state_nodes = data.state_fragment_node_index[state_indices]
    if spectrum_batch is not None:
        spectrum_batch, node_map = _select_spectrum_batch(
            spectrum_batch, indices, sample_map, group_indices, group_map
        )
        state_nodes = torch.where(
            state_nodes >= 0, node_map[state_nodes.clamp_min(0)], state_nodes
        )

    parent_states = data.teacher_node_parent_index[state_indices]
    annotations = (
        tuple(data.sample_annotations[i] for i in indices.tolist())
        if data.sample_annotations
        else ()
    )
    return replace(
        data,
        sample_tree_index=data.sample_tree_index[indices],
        condition_features=data.condition_features[indices],
        sample_branch_group_index=group_map[data.sample_branch_group_index[indices]],
        branch_group_tree_index=data.branch_group_tree_index[group_indices],
        branch_group_adduct_index=data.branch_group_adduct_index[group_indices],
        teacher_node_branch_group_index=group_map[
            data.teacher_node_branch_group_index[state_indices]
        ],
        teacher_node_reaction_ptr=state_ptr,
        teacher_node_reaction_index=state_reactions,
        teacher_positive_reaction_ptr=positive_ptr,
        teacher_positive_reaction_index=positive_reactions,
        teacher_node_observed=data.teacher_node_observed[state_indices],
        teacher_node_parent_index=torch.where(
            parent_states >= 0, state_map[parent_states.clamp_min(0)], parent_states
        ),
        teacher_node_added_reaction_index=data.teacher_node_added_reaction_index[
            state_indices
        ],
        teacher_node_reaction_count=data.teacher_node_reaction_count[state_indices],
        transition_parent_state_index=state_map[
            data.transition_parent_state_index[transition_keep]
        ],
        transition_child_state_index=state_map[
            data.transition_child_state_index[transition_keep]
        ],
        transition_added_reaction_index=data.transition_added_reaction_index[
            transition_keep
        ],
        state_transition_ptr=valid_ptr,
        state_transition_reaction_index=valid_reactions,
        state_transition_next_state_index=torch.tensor(
            valid_next_states, dtype=torch.long
        ),
        weak_negative_reaction_ptr=negative_ptr,
        weak_negative_reaction_index=negative_reactions,
        transition_state_branch_group_index=group_map[
            data.transition_state_branch_group_index[transition_state_indices]
        ],
        transition_state_reaction_ptr=transition_state_ptr,
        transition_state_reaction_index=transition_state_reactions,
        teacher_node_transition_state_index=transition_state_map[
            data.teacher_node_transition_state_index[state_indices]
        ],
        state_fragment_node_index=state_nodes,
        spectrum_batch=spectrum_batch,
        sample_annotations=annotations,
        **teacher_paths,
    )
