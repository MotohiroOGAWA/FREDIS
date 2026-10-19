"""Structure-level counts and mean per-sample assignment coverage for manifests."""


def structure_manifest_fields(structure):
    spectrum_batch = getattr(structure, "spectrum_batch", None)
    decoded = getattr(spectrum_batch, "decoded", None)
    annotations = getattr(structure, "sample_annotations", ())

    def mean(key):
        values = [sample[key] for sample in annotations if sample.get(key) is not None]
        return sum(values) / len(values) if values else None

    return dict(
        num_teacher_nodes=structure.teacher_node_branch_group_index.numel(),
        num_positive_transitions=structure.transition_added_reaction_index.numel(),
        num_branch_groups=structure.num_branch_groups,
        num_transition_states=structure.transition_state_branch_group_index.numel(),
        num_physical_ion_candidates=(
            spectrum_batch.physical_candidate_node_index.numel()
            if spectrum_batch
            else 0
        ),
        num_ion_explanations=(
            spectrum_batch.explanation_adduct_index.numel() if spectrum_batch else 0
        ),
        max_teacher_reaction_count=(
            int(structure.teacher_node_reaction_count.max())
            if structure.teacher_node_reaction_count.numel()
            else 0
        ),
        num_nodes=sum(tree.num_nodes for tree in decoded.trees) if decoded else None,
        num_edges=sum(tree.num_edges for tree in decoded.trees) if decoded else None,
        assignment_score=mean("assignmentScore"),
        assignment_score_without_precursor=mean("assignmentScoreWithoutPrecursor"),
    )
