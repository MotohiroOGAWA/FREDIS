"""Transient chemistry histories used to build the public canonical input tables."""

from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from collections.abc import Sequence
import torch
from torch import Tensor
from torch_geometric.data import Batch
from rdkit import Chem
from fredis.domain.fragment.reaction import ReactionInstance, ReactionSequence
from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations
from fredis.libs.mmkit.mmkit import Compound
from ..mol.graph_builder import MolGraphBuilder
from .reaction_tensor_data import ReactionTensorData
from .reaction_set import ReactionSet
from .source_compound import SourceCompound
from .adduct_data import AdductData
from .branch_group import BranchGroup
from ._preparation_histories import ReactionContextData, TeacherData
from ._annotation_preparation import AnnotatedSpectrumData


class UnresolvedPrecursorError(ValueError):
    """A sample's precursor ion has no valid reaction sequence from Source.

    Not every observed precursor is reachable under a given reaction pattern
    set (e.g. an fragment-ion adduct the configured patterns cannot produce); such a
    sample cannot be supervised and must be dropped, not treated as a fatal
    error for the rest of its group.
    """


def csr(rows: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
    counts = torch.tensor([len(row) for row in rows], dtype=torch.long)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))), torch.tensor(
        [v for row in rows for v in row], dtype=torch.long
    )


@dataclass(frozen=True)
class ReactionPreparation:
    # Independent SourceCompound table: graph Batch and structural identity.
    source_compounds: SourceCompound
    # Concrete source-specific chemistry; all reaction IDs refer to this registry.
    reactions: ReactionTensorData
    # Group-local contexts and transitions referencing independent ReactionSet IDs.
    states: ReactionContextData = field(
        default_factory=ReactionContextData, kw_only=True
    )
    # Optional-in-content training labels referencing states and annotations.
    teacher: TeacherData = field(default_factory=TeacherData, kw_only=True)
    adducts: AdductData = field(kw_only=True)
    branch_groups: BranchGroup = field(kw_only=True)
    # [S], float32: collision energy in eV.
    collision_energy_eV: Tensor
    # Unique sets of batch-global reaction IDs; no group/Compound/teacher ownership.
    reaction_sets: ReactionSet = field(kw_only=True)
    max_reaction_role_count: int = field(default=0, kw_only=True)
    source_atom_capacity: int = field(default=0, kw_only=True)
    # Normalized chemical/observational data; independent of branch supervision.
    annotations: AnnotatedSpectrumData | None = field(default=None, kw_only=True)
    # Preparation-only input. Dictionaries are normalized, never stored in the dataclass.
    observations: InitVar[tuple[dict, ...] | None] = field(default=None, kw_only=True)
    formula_elements: tuple[str, ...] = field(default=(), kw_only=True)
    # Runtime view: expanded group-local graph topology and sample/ion rows.
    # Redundant graphs/formulas/targets are omitted from serialized data.
    spectrum_batch: object | None = None

    # Spectrum -> (compound, main adduct) branch group. Collision energy is
    # intentionally absent from the group representation.
    sample_branch_group_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )

    def __post_init__(self, observations):
        if observations and self.spectrum_batch is not None:
            annotation = AnnotatedSpectrumData.from_runtime(
                self.spectrum_batch,
                observations,
                self.sample_branch_group_index,
                self.formula_elements,
                self.states.candidate_fragment_index,
                context_reaction_set_index=self.states.candidate_reaction_set_index,
                adducts=self.adducts,
            )
            annotation.validate(self.adducts)
            object.__setattr__(self, "annotations", annotation)

    @property
    def source_graph(self):
        """Graph view of the independent SourceCompound table."""
        return self.source_compounds.source_graph

    @property
    def source_smiles(self):
        return self.source_compounds.source_smiles

    def candidate_reactions(self):
        """Runtime CSR view of context -> shared ReactionSet -> global reactions."""
        return self.reaction_sets.gather(self.states.candidate_reaction_set_index)

    def active_reactions(self):
        ids = self.states.candidate_reaction_set_index[self.states.candidate_index]
        return self.reaction_sets.gather(ids)

    @property
    def state_reaction_count(self):
        ids = self.states.candidate_reaction_set_index[self.states.candidate_index]
        ptr = self.reaction_sets.reaction_ptr
        return ptr[ids + 1] - ptr[ids]

    @property
    def sample_tree_index(self):
        return self.branch_groups.source_compound_index[self.sample_branch_group_index]

    @property
    def num_samples(self) -> int:
        return self.sample_tree_index.numel()

    @property
    def num_branch_groups(self) -> int:
        return len(self.branch_groups)

    @property
    def device(self) -> torch.device:
        return self.reactions.reaction_type.device


def prepare_source_reactions(
    *,
    source: Compound,
    reactions: tuple[ReactionInstance, ...],
    graph_builder: MolGraphBuilder,
    # [S], float32: collision energy in eV.
    collision_energy_eV: Tensor,
    main_adduct_index: Tensor,
    adducts: AdductData,
    max_reaction_count: int,
    teacher_paths=None,
    sample_branch_group_index: Tensor | None = None,
    valid_reaction_sequences: frozenset[ReactionSequence] | None = None,
    relations: ReactionRelations | None = None,
) -> ReactionPreparation:
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

    collision_energy_eV = torch.as_tensor(collision_energy_eV, dtype=torch.float32)
    if collision_energy_eV.ndim != 1:
        raise ValueError("collision_energy_eV must have shape [S]")
    sample_count = collision_energy_eV.shape[0]
    if sample_branch_group_index is None:
        sample_branch_group_index = torch.arange(sample_count, dtype=torch.long)
    sample_branch_group_index = torch.as_tensor(
        sample_branch_group_index, dtype=torch.long
    )
    if sample_branch_group_index.shape != (sample_count,):
        raise ValueError("sample_branch_group_index must have one entry per spectrum")
    group_count = int(sample_branch_group_index.max()) + 1 if sample_count else 0
    main_adduct_index = torch.as_tensor(main_adduct_index, dtype=torch.long)
    if main_adduct_index.shape != (group_count,):
        raise ValueError("main_adduct_index must have one entry per branch group")

    owners = []
    states = []
    positives = []
    observed = []
    parents = []
    added = []
    positive_edges = []
    peak_paths = []
    peak_groups = []
    peak_observations = []
    peak_offsets = [0]
    if teacher_paths is not None:
        for peaks in teacher_paths:
            peak_offsets.append(peak_offsets[-1] + len(peaks))
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
                for peak_number, alternatives in enumerate(spectra):
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
                        peak_observations.append(peak_offsets[sample] + peak_number)
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

    positive_ptr, positive_index = csr(positives)
    valid_ptr, valid_reaction_index = csr(valid_rows)
    negative_ptr, negative_reaction_index = csr(negative_rows)
    reaction_sets, context_set_ids = ReactionSet.from_rows(transition_states)
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
    return ReactionPreparation(
        source_compounds=SourceCompound(graph, (source.smiles,), graph_builder.symbols),
        reactions=ReactionTensorData.from_reactions(reactions, maps, relations),
        collision_energy_eV=collision_energy_eV,
        reaction_sets=reaction_sets,
        sample_branch_group_index=sample_branch_group_index,
        adducts=adducts,
        branch_groups=BranchGroup(
            torch.zeros(group_count, dtype=torch.long), main_adduct_index
        ),
        source_atom_capacity=len(maps),
        max_reaction_role_count=max(
            (len(a.source_atom_maps) for a in reactions), default=0
        ),
        states=ReactionContextData(
            branch_group_index=torch.tensor(owners),
            transition_parent_index=transition_parent,
            transition_child_index=transition_child,
            transition_reaction_index=transition_reaction,
            fragment_node_index=torch.full((len(states),), -1, dtype=torch.long),
            parent_index=torch.tensor(parents),
            added_reaction_index=torch.tensor(added),
            next_ptr=valid_ptr,
            next_reaction_index=valid_reaction_index,
            next_state_index=torch.tensor(valid_next, dtype=torch.long),
            candidate_branch_group_index=torch.tensor(
                transition_state_owners, dtype=torch.long
            ),
            candidate_reaction_set_index=context_set_ids,
            candidate_fragment_index=torch.full(
                (len(transition_states),), -1, dtype=torch.long
            ),
            candidate_index=torch.tensor(teacher_transition_states, dtype=torch.long),
        ),
        teacher=TeacherData(
            peak_index=torch.tensor(peak_observations, dtype=torch.long),
            positive_reaction_ptr=positive_ptr,
            positive_reaction_index=positive_index,
            observed=torch.tensor(observed, dtype=torch.bool),
            negative_reaction_ptr=negative_ptr,
            negative_reaction_index=negative_reaction_index,
            peak_path_ptr=peak_path_ptr,
            peak_branch_group_index=torch.tensor(peak_groups, dtype=torch.long),
            path_step_ptr=path_step_ptr,
            path_step_state_index=path_step_state_index,
            path_step_reaction_index=path_step_reaction_index,
        ),
    )
