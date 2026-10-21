"""Prepared tensors for CE-independent fragment-tree branch training.

Chemistry belongs in preparation. Objects loaded by the training loop contain
only graphs, indices and numeric targets; prepared datasets are regenerated
when this contract changes.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from collections.abc import Sequence
import torch
from torch import Tensor
from torch_geometric.data import Batch
from rdkit import Chem
from fredis.domain.fragment.reaction import ReactionInstance, ReactionSequence
from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations
from fredis.libs.mmkit.mmkit import Compound
from ..mol.graph_builder import MolGraphBuilder

SCHEMA = "fredis.fragment-tree-training"


class UnresolvedPrecursorError(ValueError):
    """A sample's precursor ion has no valid reaction sequence from Source.

    Not every observed precursor is reachable under a given reaction pattern
    set (e.g. an fragment-ion adduct the configured patterns cannot produce); such a
    sample cannot be supervised and must be dropped, not treated as a fatal
    error for the rest of its group.
    """


def coo(pairs: Sequence[tuple[int, int]]) -> Tensor:
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()


def csr(rows: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
    counts = torch.tensor([len(row) for row in rows], dtype=torch.long)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))), torch.tensor(
        [v for row in rows for v in row], dtype=torch.long
    )


@dataclass(frozen=True)
class SourceReactionStructure:
    source_graph: Batch
    reaction_tree_index: Tensor
    reaction_type: Tensor
    reaction_source_atom_ptr: Tensor
    reaction_source_atom_index: Tensor
    reaction_conflict_index: Tensor
    reaction_invalidation_index: Tensor
    reaction_dominance_index: Tensor
    reaction_retained_index: Tensor
    sample_tree_index: Tensor
    condition_features: Tensor
    teacher_node_branch_group_index: Tensor
    teacher_node_reaction_ptr: Tensor
    teacher_node_reaction_index: Tensor
    teacher_positive_reaction_ptr: Tensor
    teacher_positive_reaction_index: Tensor
    teacher_node_observed: Tensor
    # Downstream precomputed graphs are optional for scorer-only training.
    source_smiles: tuple[str, ...] = field(default=(), kw_only=True)
    max_reaction_role_count: int = field(default=0, kw_only=True)
    source_atom_capacity: int = field(default=0, kw_only=True)
    sample_annotations: tuple[dict, ...] = field(default=(), kw_only=True)
    spectrum_batch: object | None = None
    transition_parent_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    transition_child_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    transition_added_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    state_fragment_node_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )

    teacher_node_parent_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    teacher_node_added_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    teacher_node_reaction_count: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )

    # Spectrum -> (compound, main adduct) branch group. Collision energy is
    # intentionally absent from the group representation.
    sample_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    branch_group_tree_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    branch_group_adduct_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )

    # Preparation-computed valid transitions and weak negatives.
    state_transition_ptr: Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long), kw_only=True
    )
    state_transition_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    state_transition_next_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    transition_state_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    transition_state_reaction_ptr: Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long), kw_only=True
    )
    transition_state_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    teacher_node_transition_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    weak_negative_reaction_ptr: Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long), kw_only=True
    )
    weak_negative_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )

    # Positive MIL: peak -> alternative path -> prepared (state, reaction) steps.
    teacher_peak_path_ptr: Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long), kw_only=True
    )
    teacher_peak_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    teacher_path_step_ptr: Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long), kw_only=True
    )
    teacher_path_step_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    teacher_path_step_reaction_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )

    @property
    def num_samples(self) -> int:
        return self.sample_tree_index.numel()

    @property
    def num_branch_groups(self) -> int:
        return (
            self.branch_group_tree_index.numel()
            if self.branch_group_tree_index.numel()
            else self.num_samples
        )

    @property
    def device(self) -> torch.device:
        return self.reaction_type.device

    def to(self, device: str | torch.device) -> SourceReactionStructure:
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if hasattr(getattr(self, f.name), "to")
            },
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            dict(
                schema=SCHEMA, structure=self.to("cpu")
            ),
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        *,
        max_reaction_count: int = 3,
    ) -> SourceReactionStructure:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != SCHEMA
        ):
            raise ValueError(
                "Unsupported prepared data; regenerate from the original MSDataset"
            )
        result = payload["structure"]
        if not isinstance(result, cls):
            raise TypeError(
                "Prepared structure must be SourceReactionStructure; regenerate the dataset"
            )
        return result.to(device)

    @classmethod
    def from_structures(
        cls, structures: Sequence[SourceReactionStructure]
    ) -> SourceReactionStructure:
        if not structures:
            raise ValueError("Cannot collate an empty batch")
        graphs, reaction_types, atom_rows, retained, sample_trees, conditions = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        relations = {
            name: []
            for name in (
                "reaction_conflict_index",
                "reaction_invalidation_index",
                "reaction_dominance_index",
            )
        }
        state_samples, state_rows, next_rows, eos, tree_indices = (
            [],
            [],
            [],
            [],
            [],
        )
        atom_offset = reaction_offset = tree_offset = sample_offset = group_offset = (
            state_offset
        ) = node_offset = 0
        (
            spectrum_batches,
            reaction_offsets,
            parents,
            children,
            added_reactions,
            state_nodes,
        ) = ([], [], [], [], [], [])
        annotations = []
        node_parents = []
        node_added = []
        sample_groups = []
        group_trees = []
        group_adducts = []
        valid_rows = []
        valid_next = []
        negative_rows = []
        transition_state_groups = []
        transition_state_rows = []
        teacher_transition_states = []
        transition_state_offset = 0
        peak_paths = []
        peak_groups = []
        path_steps = []
        for item in structures:
            item_state_offset = state_offset
            reaction_offsets.append(reaction_offset)
            node_parents.append(
                torch.where(
                    item.teacher_node_parent_index >= 0,
                    item.teacher_node_parent_index + state_offset,
                    item.teacher_node_parent_index,
                )
            )
            node_added.append(
                torch.where(
                    item.teacher_node_added_reaction_index >= 0,
                    item.teacher_node_added_reaction_index + reaction_offset,
                    item.teacher_node_added_reaction_index,
                )
            )
            parents.append(item.transition_parent_state_index + state_offset)
            children.append(item.transition_child_state_index + state_offset)
            added_reactions.append(
                item.transition_added_reaction_index + reaction_offset
            )
            state_nodes.append(
                torch.where(
                    item.state_fragment_node_index >= 0,
                    item.state_fragment_node_index + node_offset,
                    item.state_fragment_node_index,
                )
            )
            for sample in getattr(item, "sample_annotations", ()):
                annotations.append(
                    {
                        **sample,
                        "peaks": [
                            {
                                **peak,
                                "matches": [
                                    {
                                        **match,
                                        "nodeIndices": [
                                            index + node_offset
                                            for index in match["nodeIndices"]
                                        ],
                                    }
                                    for match in peak["matches"]
                                ],
                            }
                            for peak in sample["peaks"]
                        ],
                    }
                )
            state_offset += item.teacher_node_branch_group_index.numel()
            if item.spectrum_batch is not None:
                spectrum_batches.append(item.spectrum_batch)
                node_offset += len(item.spectrum_batch.decoded.compounds)
            graphs.extend(item.source_graph.to_data_list())
            reaction_types.append(item.reaction_type)
            tree_indices.append(item.reaction_tree_index + tree_offset)
            for start, stop in zip(
                item.reaction_source_atom_ptr[:-1], item.reaction_source_atom_ptr[1:]
            ):
                atom_rows.append(
                    (item.reaction_source_atom_index[start:stop] + atom_offset).tolist()
                )
            retained.append(
                item.reaction_retained_index
                + torch.tensor(
                    [[reaction_offset], [atom_offset]],
                    device=item.reaction_retained_index.device,
                )
            )
            for name in relations:
                relations[name].append(getattr(item, name) + reaction_offset)
            sample_trees.append(item.sample_tree_index + tree_offset)
            conditions.append(item.condition_features)
            state_samples.append(item.teacher_node_branch_group_index + group_offset)
            for ptr, data, dest in (
                (
                    item.teacher_node_reaction_ptr,
                    item.teacher_node_reaction_index,
                    state_rows,
                ),
                (
                    item.teacher_positive_reaction_ptr,
                    item.teacher_positive_reaction_index,
                    next_rows,
                ),
            ):
                for start, stop in zip(ptr[:-1], ptr[1:]):
                    dest.append((data[start:stop] + reaction_offset).tolist())
            sample_groups.append(item.sample_branch_group_index + group_offset)
            group_trees.append(item.branch_group_tree_index + tree_offset)
            group_adducts.append(item.branch_group_adduct_index)
            for row in range(item.teacher_node_branch_group_index.numel()):
                a, b = item.state_transition_ptr[row : row + 2]
                valid_rows.append(
                    (
                        item.state_transition_reaction_index[a:b] + reaction_offset
                    ).tolist()
                )
                valid_next.extend(
                    (
                        item.state_transition_next_state_index[a:b]
                        + transition_state_offset
                    ).tolist()
                )
                a, b = item.weak_negative_reaction_ptr[row : row + 2]
                negative_rows.append(
                    (item.weak_negative_reaction_index[a:b] + reaction_offset).tolist()
                )
            transition_state_groups.append(
                item.transition_state_branch_group_index + group_offset
            )
            for start, stop in zip(
                item.transition_state_reaction_ptr[:-1],
                item.transition_state_reaction_ptr[1:],
            ):
                transition_state_rows.append(
                    (
                        item.transition_state_reaction_index[start:stop]
                        + reaction_offset
                    ).tolist()
                )
            teacher_transition_states.append(
                item.teacher_node_transition_state_index + transition_state_offset
            )
            transition_state_offset += item.transition_state_branch_group_index.numel()
            for peak in range(item.teacher_peak_path_ptr.numel() - 1):
                peak_paths.append([])
                peak_groups.append(
                    int(item.teacher_peak_branch_group_index[peak]) + group_offset
                )
                for path in range(
                    int(item.teacher_peak_path_ptr[peak]),
                    int(item.teacher_peak_path_ptr[peak + 1]),
                ):
                    peak_paths[-1].append(len(path_steps))
                    a, b = item.teacher_path_step_ptr[path : path + 2]
                    path_steps.append(
                        [
                            (
                                int(state) + item_state_offset,
                                int(reaction) + reaction_offset,
                            )
                            for state, reaction in zip(
                                item.teacher_path_step_state_index[a:b],
                                item.teacher_path_step_reaction_index[a:b],
                            )
                        ]
                    )
            eos.append(item.teacher_node_observed)
            atom_offset += item.source_graph.num_nodes
            reaction_offset += item.reaction_type.shape[0]
            tree_offset += item.source_graph.num_graphs
            sample_offset += item.sample_tree_index.numel()
            group_offset += item.num_branch_groups
        graph = Batch.from_data_list(graphs)
        atom_ptr, atom_index = csr(atom_rows)
        state_ptr, state_index = csr(state_rows)
        next_ptr, next_index = csr(next_rows)
        valid_ptr, valid_reactions = csr(valid_rows)
        negative_ptr, negative_reactions = csr(negative_rows)
        transition_state_ptr, transition_state_reactions = csr(transition_state_rows)
        peak_ptr, unused = csr(peak_paths)
        path_ptr, path_states = csr(
            [[state for state, _ in path] for path in path_steps]
        )
        path_reactions = torch.tensor(
            [reaction for path in path_steps for _, reaction in path], dtype=torch.long
        )
        spectrum_batch = None
        if spectrum_batches:
            if len(spectrum_batches) != len(structures):
                raise ValueError(
                    "Cannot mix reaction-only and spectrum_batch structures"
                )
            from ..specgen.fragment_spectrum_model import collate_spectrum_model_batches

            spectrum_batch = collate_spectrum_model_batches(
                spectrum_batches, reaction_offsets
            )
        return cls(
            graph,
            torch.cat(tree_indices),
            torch.cat(reaction_types),
            atom_ptr,
            atom_index,
            *(torch.cat(relations[name], dim=1) for name in relations),
            torch.cat(retained, dim=1),
            torch.cat(sample_trees),
            torch.cat(conditions),
            torch.cat(state_samples),
            state_ptr,
            state_index,
            next_ptr,
            next_index,
            torch.cat(eos),
            spectrum_batch,
            source_smiles=tuple(
                key for item in structures for key in item.source_smiles
            ),
            transition_parent_state_index=torch.cat(parents),
            transition_child_state_index=torch.cat(children),
            transition_added_reaction_index=torch.cat(added_reactions),
            state_fragment_node_index=torch.cat(state_nodes),
            teacher_node_parent_index=torch.cat(node_parents),
            teacher_node_added_reaction_index=torch.cat(node_added),
            teacher_node_reaction_count=torch.cat(
                [i.teacher_node_reaction_count for i in structures]
            ),
            sample_branch_group_index=torch.cat(sample_groups),
            branch_group_tree_index=torch.cat(group_trees),
            branch_group_adduct_index=torch.cat(group_adducts),
            state_transition_ptr=valid_ptr,
            state_transition_reaction_index=valid_reactions,
            state_transition_next_state_index=torch.tensor(
                valid_next, dtype=torch.long
            ),
            transition_state_branch_group_index=torch.cat(transition_state_groups),
            transition_state_reaction_ptr=transition_state_ptr,
            transition_state_reaction_index=transition_state_reactions,
            teacher_node_transition_state_index=torch.cat(teacher_transition_states),
            weak_negative_reaction_ptr=negative_ptr,
            weak_negative_reaction_index=negative_reactions,
            teacher_peak_path_ptr=peak_ptr,
            teacher_peak_branch_group_index=torch.tensor(peak_groups, dtype=torch.long),
            teacher_path_step_ptr=path_ptr,
            teacher_path_step_state_index=path_states,
            teacher_path_step_reaction_index=path_reactions,
            sample_annotations=(
                tuple(annotations)
                if len(annotations) == sum(item.num_samples for item in structures)
                else ()
            ),
            max_reaction_role_count=max(
                item.max_reaction_role_count for item in structures
            ),
            source_atom_capacity=max(item.source_atom_capacity for item in structures),
        )


def prepare_source_reactions(
    *,
    source: Compound,
    reactions: tuple[ReactionInstance, ...],
    graph_builder: MolGraphBuilder,
    condition_features: Tensor,
    max_reaction_count: int,
    teacher_paths=None,
    sample_branch_group_index: Tensor | None = None,
    branch_group_adduct_index: Tensor | None = None,
    valid_reaction_sequences: frozenset[ReactionSequence] | None = None,
    relations: ReactionRelations | None = None,
) -> SourceReactionStructure:
    """Pack source reactions, CE-aggregated teachers and all branch supervision.

    ``teacher_paths[spectrum][peak][alternative]`` contains complete paths
    rooted at Source.  Spectra mapped to the same branch group contribute to
    one union teacher DAG, so a reaction observed at any CE can never become a
    weak negative at another CE.
    """
    if relations is None:
        relations = ReactionRelations.from_reactions(reactions)
    reaction_index = {reaction: i for i, reaction in enumerate(reactions)}
    maps = {
        atom.GetAtomMapNum(): i
        for i, atom in enumerate(graph_builder.graph_atoms(source))
    }
    raw_graph = graph_builder.build(source)
    atom_ptr, atom_index = csr(
        [
            tuple(maps[v] for v in reaction.source_atom_maps if v in maps)
            for reaction in reactions
        ]
    )
    retained = coo(
        [
            (i, maps[v])
            for i, reaction in enumerate(reactions)
            for v in sorted(reaction.retained_atom_maps)
            if v in maps
        ]
    )

    sample_count = condition_features.shape[0]
    if sample_branch_group_index is None:
        sample_branch_group_index = torch.arange(sample_count, dtype=torch.long)
    sample_branch_group_index = torch.as_tensor(
        sample_branch_group_index, dtype=torch.long
    )
    if sample_branch_group_index.shape != (sample_count,):
        raise ValueError("sample_branch_group_index must have one entry per spectrum")
    group_count = int(sample_branch_group_index.max()) + 1 if sample_count else 0
    if branch_group_adduct_index is None:
        branch_group_adduct_index = torch.tensor(
            [
                int(
                    condition_features[(sample_branch_group_index == g).nonzero()[0], 0]
                )
                for g in range(group_count)
            ]
        )
    branch_group_adduct_index = torch.as_tensor(
        branch_group_adduct_index, dtype=torch.long
    )
    if branch_group_adduct_index.shape != (group_count,):
        raise ValueError(
            "branch_group_adduct_index must have one entry per branch group"
        )

    owners = []
    states = []
    positives = []
    observed = []
    parents = []
    added = []
    depths = []
    positive_edges = []
    peak_paths = []
    peak_groups = []
    if teacher_paths is not None and len(teacher_paths) != sample_count:
        raise ValueError("teacher_paths must have one row per spectrum")
    for group in range(group_count):
        known = {}
        outgoing = {}
        edge_set = set()

        def add_state(state, parent=-1, reaction=-1):
            key = (
                tuple(sorted(reaction_index[a] for a in state.reactions))
                if state
                else ()
            )
            if key in known:
                return known[key]
            row = len(states)
            known[key] = row
            owners.append(group)
            states.append(key)
            positives.append([])
            observed.append(False)
            parents.append(parent)
            added.append(reaction)
            depths.append(len(key))
            outgoing[row] = {}
            return row

        add_state(None)
        if teacher_paths is not None:
            for sample in (
                (sample_branch_group_index == group).nonzero().flatten().tolist()
            ):
                spectra = teacher_paths[sample]
                if (
                    spectra
                    and isinstance(spectra[0], tuple)
                    and len(spectra[0]) == 2
                    and isinstance(spectra[0][1], (float, int))
                ):
                    spectra = [[value] for value in spectra]
                for alternatives in spectra:
                    paths_for_peak = []
                    for value in alternatives:
                        chain = (
                            value[0]
                            if isinstance(value, tuple)
                            and len(value) == 2
                            and isinstance(value[1], (float, int))
                            else value
                        )
                        parent = add_state(None)
                        steps = []
                        for child, reaction in chain[1:]:
                            if reaction is None:
                                continue
                            reaction_id = reaction_index[reaction]
                            child_row = add_state(child, parent, reaction_id)
                            if child_row == parent:
                                continue
                            outgoing[parent].setdefault(reaction_id, set()).add(
                                child_row
                            )
                            edge_set.add((parent, child_row, reaction_id))
                            steps.append((parent, reaction_id))
                            parent = child_row
                        if steps:
                            paths_for_peak.append(steps)
                            observed[parent] = True
                    if paths_for_peak:
                        peak_paths.append(paths_for_peak)
                        peak_groups.append(group)
        for row in known.values():
            positives[row] = sorted(outgoing[row])
        positive_edges.extend(sorted(edge_set))

    # Chemically compatible continuations and weak negatives are computed once
    # here.  No reaction applicability or object chemistry remains in training.
    valid_rows = []
    valid_next = []
    negative_rows = []
    transition_states = list(states)
    transition_state_owners = list(owners)
    transition_state_lookup = {
        (owner, tuple(state)): row
        for row, (owner, state) in enumerate(
            zip(transition_state_owners, transition_states)
        )
    }
    teacher_transition_states = [
        transition_state_lookup[(owner, tuple(state))]
        for owner, state in zip(owners, states)
    ]
    valid_effects = {}
    for row, (owner, state) in enumerate(zip(owners, states)):
        row_reactions = []
        row_next = []
        if len(state) < max_reaction_count:
            for candidate in range(len(reactions)):
                if candidate in state:
                    continue
                raw = tuple(dict.fromkeys((*state, candidate)))
                if relations.rejection(raw):
                    continue
                sequence = ReactionSequence(reactions[i] for i in raw)
                if (
                    not sequence.retained_atom_maps
                    or len(sequence.reactions) > max_reaction_count
                ):
                    continue
                if sequence.effect_key not in valid_effects:
                    if valid_reaction_sequences is not None:
                        valid_effects[sequence.effect_key] = (
                            sequence in valid_reaction_sequences
                        )
                    else:
                        try:
                            valid_effects[sequence.effect_key] = (
                                len(sequence.compile(source).run(source)) == 1
                            )
                        except (
                            Chem.rdchem.MolSanitizeException,
                            ValueError,
                            RuntimeError,
                        ):
                            valid_effects[sequence.effect_key] = False
                if not valid_effects[sequence.effect_key]:
                    continue
                normalized = tuple(
                    sorted(reaction_index[a] for a in sequence.reactions)
                )
                key = (owner, normalized)
                if key not in transition_state_lookup:
                    transition_state_lookup[key] = len(transition_states)
                    transition_state_owners.append(owner)
                    transition_states.append(normalized)
                row_reactions.append(candidate)
                row_next.append(transition_state_lookup[key])
        valid_rows.append(row_reactions)
        valid_next.extend(row_next)
        missing = set(positives[row]) - set(row_reactions)
        if missing:
            raise ValueError(
                f"Teacher state {row} contains chemically invalid positive reactions: {sorted(missing)}"
            )
        negative_rows.append(sorted(set(row_reactions) - set(positives[row])))

    state_ptr, state_index = csr(states)
    positive_ptr, positive_index = csr(positives)
    valid_ptr, valid_reaction_index = csr(valid_rows)
    negative_ptr, negative_reaction_index = csr(negative_rows)
    transition_state_ptr, transition_state_reaction_index = csr(transition_states)
    flattened_paths = [path for peak in peak_paths for path in peak]
    peak_path_ptr = torch.tensor(
        [0, *torch.tensor([len(peak) for peak in peak_paths]).cumsum(0).tolist()],
        dtype=torch.long,
    )
    path_step_ptr, path_step_state_index = csr(
        [[state for state, _ in path] for path in flattened_paths]
    )
    path_step_reaction_index = torch.tensor(
        [reaction for path in flattened_paths for _, reaction in path], dtype=torch.long
    )
    transition_parent = torch.tensor(
        [a for a, _, _ in positive_edges], dtype=torch.long
    )
    transition_child = torch.tensor([b for _, b, _ in positive_edges], dtype=torch.long)
    transition_reaction = torch.tensor(
        [c for _, _, c in positive_edges], dtype=torch.long
    )

    graph = Batch.from_data_list([raw_graph])
    return SourceReactionStructure(
        graph,
        torch.zeros(len(reactions), dtype=torch.long),
        torch.tensor(
            [
                (a.reaction_pattern_id, a.reaction_rule_id, a.product_molecule_id)
                for a in reactions
            ],
            dtype=torch.long,
        ).reshape(-1, 3),
        atom_ptr,
        atom_index,
        coo(relations.conflict_pairs),
        coo(relations.invalidation_pairs),
        coo(relations.dominance_pairs),
        retained,
        torch.zeros(sample_count, dtype=torch.long),
        condition_features,
        torch.tensor(owners),
        state_ptr,
        state_index,
        positive_ptr,
        positive_index,
        torch.tensor(observed, dtype=torch.bool),
        source_smiles=(source.smiles,),
        transition_parent_state_index=transition_parent,
        transition_child_state_index=transition_child,
        transition_added_reaction_index=transition_reaction,
        state_fragment_node_index=torch.full((len(states),), -1, dtype=torch.long),
        teacher_node_parent_index=torch.tensor(parents),
        teacher_node_added_reaction_index=torch.tensor(added),
        teacher_node_reaction_count=torch.tensor(depths),
        sample_branch_group_index=sample_branch_group_index,
        branch_group_tree_index=torch.zeros(group_count, dtype=torch.long),
        branch_group_adduct_index=branch_group_adduct_index,
        state_transition_ptr=valid_ptr,
        state_transition_reaction_index=valid_reaction_index,
        state_transition_next_state_index=torch.tensor(valid_next, dtype=torch.long),
        transition_state_branch_group_index=torch.tensor(
            transition_state_owners, dtype=torch.long
        ),
        transition_state_reaction_ptr=transition_state_ptr,
        transition_state_reaction_index=transition_state_reaction_index,
        teacher_node_transition_state_index=torch.tensor(
            teacher_transition_states, dtype=torch.long
        ),
        weak_negative_reaction_ptr=negative_ptr,
        weak_negative_reaction_index=negative_reaction_index,
        teacher_peak_path_ptr=peak_path_ptr,
        teacher_peak_branch_group_index=torch.tensor(peak_groups, dtype=torch.long),
        teacher_path_step_ptr=path_step_ptr,
        teacher_path_step_state_index=path_step_state_index,
        teacher_path_step_reaction_index=path_step_reaction_index,
        source_atom_capacity=len(maps),
        max_reaction_role_count=max(
            (len(a.source_atom_maps) for a in reactions), default=0
        ),
    )


def make_structure_file_stem(smiles: str, *, index: int) -> str:
    """Make a stable readable file stem for one SMILES group."""
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    return f"smiles_{index:06d}_{digest}"


def save_fragment_tree_structure(
    *,
    structure: SourceReactionStructure,
    output_file: str | Path,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save one structure and a small metadata sidecar into a torch file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            schema=SCHEMA,
            structure=structure.to("cpu"),
            metadata=dict(metadata or {}),
        ),
        output_path,
    )
