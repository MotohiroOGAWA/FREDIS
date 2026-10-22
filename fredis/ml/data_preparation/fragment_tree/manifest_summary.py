"""Counts and assignment coverage from canonical prepared components."""


def structure_manifest_fields(structure, annotations=None):
    if annotations is None:
        annotations = structure.sample_annotations

    def mean(key):
        values = [sample[key] for sample in annotations if sample.get(key) is not None]
        return sum(values) / len(values) if values else None

    labels = structure.reaction_contexts.edge_label
    execution = structure.execution_graph
    ptr = structure.reaction_sets.reaction_ptr
    depth = ptr[execution.reaction_set_index + 1] - ptr[execution.reaction_set_index]
    return dict(
        num_teacher_nodes=int((labels >= 0).sum()) if labels is not None else 0,
        num_positive_transitions=int((labels == 1).sum()) if labels is not None else 0,
        num_branch_groups=structure.num_branch_groups,
        num_transition_states=execution.compound_index.numel(),
        num_ion_states=structure.spectrum_explanations.formula_explanation_index.numel(),
        max_teacher_reaction_count=int(depth.max()) if depth.numel() else 0,
        num_nodes=len(structure.compounds.compounds),
        num_edges=structure.compound_graph.reaction_index.numel(),
        assignment_score=mean("assignmentScore"),
        assignment_score_without_precursor=mean("assignmentScoreWithoutPrecursor"),
    )
