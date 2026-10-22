"""Preparation-only reaction teachers from original spectra and Source chemistry."""

from __future__ import annotations
from dataclasses import replace
from collections.abc import Sequence
from types import SimpleNamespace
from fredis.ml.input.adduct_data import AdductData
import torch
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.domain.fragment.reaction import FragmentTreeLimitExceeded
from .reaction_set import ReactionSet
from ._reaction_preparation import (
    ReactionPreparation as SourceReactionStructure,
    prepare_source_reactions,
    UnresolvedPrecursorError,
    csr,
)
from ..specgen.source_anchored_spectrum_predictor import (
    SourceAnchoredFragmentSpectrumGenerator,
)
from ..specgen.components.reaction.reaction_decoder import ReactionDecoderOutput
from ..specgen.materialization import materialize_reaction_states
from ..specgen.fragment_spectrum_model import prepare_spectrum_model_batch


def _transition_chains(transitions_by_sequence, sequence):
    """Every recorded transition chain from Source to one reaction combination.

    The combination is applied to Source at once; the chains only supply the
    one-reaction-at-a-time steps the branching decoder is trained on.
    """
    if sequence is None:
        return (((None, None),),)
    chains = []
    for transition in transitions_by_sequence.get(sequence, ()):
        for prefix in _transition_chains(
            transitions_by_sequence, transition.parent_reaction_sequence
        ):
            chains.append((*prefix, (sequence, transition.added_reaction)))
    return tuple(dict.fromkeys(chains))


def _transitions_by_sequence(tree):
    result = {}
    for edge_index in range(tree.num_edges):
        for transition in tree.get_edge(edge_index).transitions:
            result.setdefault(transition.reaction_sequence, []).append(transition)
    return result


class _ChemistryPreparationBuilder:
    def __init__(
        self,
        generator: SourceAnchoredFragmentSpectrumGenerator,
        *,
        max_unique_fragment_smiles: int = -1,
        max_reaction_combinations: int = -1,
    ) -> None:
        self.generator = generator
        self.max_unique_fragment_smiles = max_unique_fragment_smiles
        self.max_reaction_combinations = max_reaction_combinations
        # Fragment-tree search statistics of the most recent build() call,
        # including one that stopped at a search limit or failed after the
        # tree was built; None when the tree search never ran.
        self.last_search_stats: dict[str, int] | None = None
        self.last_reaction_sequence_compounds = {}
        self.last_reactions = ()

    def build(
        self,
        source: Compound,
        precursor_types: Sequence[Adduct],
        collision_energy: Sequence[float],
        peaks_mz: Sequence[Sequence[float]],
        peaks_intensity: Sequence[Sequence[float]],
        *,
        seed_compounds: dict | None = None,
        addable_pattern_ids: frozenset[int] | None = None,
        precomputed_reactions: Sequence | None = None,
        precomputed_relations=None,
    ) -> tuple[SourceReactionStructure, tuple[int, ...]]:
        """Return the built structure plus which input sample indices it kept.

        A sample whose precursor ion is unreachable under the configured
        reaction patterns (e.g. an fragment-ion adduct the patterns cannot produce) is
        dropped rather than failing the whole group: real MS2 fragmentation
        never happens on an unreachable precursor, so such a sample has
        nothing to supervise. The caller uses the returned indices to report
        which of its original records were actually kept.
        """
        if not precursor_types or not len(precursor_types) == len(
            collision_energy
        ) == len(peaks_mz) == len(peaks_intensity):
            raise ValueError(
                "Every spectrum needs precursor, CE, peaks and intensities"
            )
        for mz, intensity in zip(peaks_mz, peaks_intensity):
            if len(mz) != len(intensity):
                raise ValueError("Peak m/z and intensity lengths differ")
        self.last_search_stats = None
        self.last_reaction_sequence_compounds = {}
        self.last_reactions = ()
        generator = self.generator
        fragmenter = generator.fragmenter
        # precomputed_reactions are the Source reactions this build may add: all
        # of them for a full build, or only the addable ones when seeded.
        if precomputed_reactions is not None:
            precomputed_reactions = tuple(
                sorted(set(precomputed_reactions), key=lambda reaction: reaction.key)
            )
        if addable_pattern_ids is None:
            reactions = (
                precomputed_reactions
                if precomputed_reactions is not None
                else fragmenter.fragment_ion_tree_builder.create_reaction_instances(
                    source
                )
            )
        else:
            added = (
                precomputed_reactions
                if precomputed_reactions is not None
                else fragmenter.fragment_ion_tree_builder.create_reaction_instances(
                    source, pattern_ids=addable_pattern_ids
                )
            )
            reactions = tuple(
                sorted(
                    set(added)
                    | {
                        reaction
                        for sequence in (seed_compounds or {})
                        for reaction in sequence.reactions
                    },
                    key=lambda reaction: reaction.key,
                )
            )
        from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations

        relations = (
            precomputed_relations
            if precomputed_relations is not None
            else ReactionRelations.from_reactions(tuple(reactions))
        )
        self.last_reactions = tuple(reactions)
        # Full chemistry is permitted here, never inside neural forward.
        requested_count = fragmenter.tree_max_reaction_count
        failures = []
        # Combination overflow is handled before chemistry by the domain search.
        # Only the structure-count limit can require a lower-depth rebuild.
        for completed_count in range(requested_count, -1, -1):
            try:
                tree = fragmenter.build_fragment_ion_tree(
                    source,
                    max_unique_fragment_smiles=self.max_unique_fragment_smiles,
                    max_reaction_combinations=self.max_reaction_combinations,
                    max_reaction_count=completed_count,
                    seed_reaction_sequences=tuple(seed_compounds or ()),
                    seed_compounds=seed_compounds,
                    addable_pattern_ids=addable_pattern_ids,
                    precomputed_reactions=reactions,
                    precomputed_relations=relations,
                )
                break
            except FragmentTreeLimitExceeded as error:
                if error.limit_name != "max_unique_fragment_smiles":
                    self.last_search_stats = error.stats
                    raise
                failures.append(
                    dict(
                        attempted_reaction_count=completed_count,
                        limit_name=error.limit_name,
                        limit=error.limit,
                        stats=error.stats,
                    )
                )
                self.last_search_stats = error.stats
                if completed_count == 0:
                    raise
            # A seed beyond the retained depth cannot be represented faithfully.
            if (
                seed_compounds
                and max(len(sequence.reactions) for sequence in seed_compounds)
                > completed_count - 1
            ):
                raise FragmentTreeLimitExceeded(
                    failures[-1]["limit_name"],
                    failures[-1]["limit"],
                    self.last_search_stats,
                )
        completed = tree.search_stats["completed_max_reaction_count"]
        truncated = bool(failures) or tree.search_stats["search_truncated"]
        self.last_search_stats = {
            **tree.search_stats,
            "requested_max_reaction_count": requested_count,
            "search_truncated": truncated,
            "discarded_from_reaction_count": completed + 1 if truncated else None,
            "limit_attempts": failures + tree.search_stats["limit_attempts"],
        }
        self.last_reaction_sequence_compounds = tree.reaction_sequence_compounds
        # Resolve the observed precursor against the same already-built tree so
        # unreachable records can be dropped before teacher assignment. These
        # paths are validation metadata, not decoder seeds: branching starts at
        # Source and is conditioned by normalized main adduct.
        precursor_reachable = [
            bool(fragmenter.resolve_precursor_reactions(tree, precursor_type))
            for precursor_type in precursor_types
        ]
        kept = tuple(
            index for index, reachable in enumerate(precursor_reachable) if reachable
        )
        if not kept:
            raise UnresolvedPrecursorError(
                "No sample in this group has a valid precursor reaction sequence from Source"
            )
        if len(kept) != len(precursor_types):
            precursor_types = [precursor_types[index] for index in kept]
            collision_energy = [collision_energy[index] for index in kept]
            peaks_mz = [peaks_mz[index] for index in kept]
            peaks_intensity = [peaks_intensity[index] for index in kept]
        assignments = fragmenter.annotate_peak_sets(
            tree, zip(precursor_types, peaks_mz)
        )
        transitions_by_sequence = _transitions_by_sequence(tree)
        teacher_paths = []
        for peak_annotations in assignments:
            peaks = []
            for explanations in peak_annotations:
                paths = []
                for sequence in dict.fromkeys(
                    annotation.reaction_sequence for annotation in explanations
                ):
                    paths.extend(_transition_chains(transitions_by_sequence, sequence))
                peaks.append(list(dict.fromkeys(paths)))
            teacher_paths.append(peaks)
        mains = [
            fragmenter._resolve_main_adduct_type(adduct) for adduct in precursor_types
        ]
        energies = torch.tensor(collision_energy, dtype=torch.float32)
        group_keys = {}
        sample_groups = []
        for main in mains:
            sample_groups.append(group_keys.setdefault(str(main), len(group_keys)))
        group_adduct = torch.tensor(
            [generator.adduct_type_strs.index(key) for key in group_keys],
            dtype=torch.long,
        )
        data = prepare_source_reactions(
            source=source,
            reactions=reactions,
            graph_builder=generator.mol_encoder.graph_builder,
            relations=relations,
            collision_energy_eV=energies,
            max_reaction_count=fragmenter.tree_max_reaction_count,
            teacher_paths=teacher_paths,
            sample_branch_group_index=torch.tensor(sample_groups),
            main_adduct_index=group_adduct,
            adducts=AdductData.from_fragmenter(fragmenter),
            valid_reaction_sequences=frozenset(self.last_reaction_sequence_compounds),
        )
        # Materialization adds its own Source bookkeeping root. Teacher depth
        # zero is the shared empty reaction state for each branch group.
        active_ptr, active_reactions = data.active_reactions()
        rows = [
            tuple(active_reactions[a:b].tolist())
            for a, b in zip(active_ptr[:-1], active_ptr[1:])
        ]
        samples = data.states.branch_group_index.tolist()
        dense_rows = []
        dense_sample = []
        parents = []
        added = []
        terminal = []
        teacher_dense = {}
        width = fragmenter.tree_max_reaction_count
        for sample in range(data.num_branch_groups):
            source_row = len(dense_rows)
            dense_rows.append(tuple([-1] * width))
            dense_sample.append(sample)
            parents.append(-1)
            added.append(-1)
            terminal.append(False)
            for node, (owner, ids) in enumerate(zip(samples, rows)):
                if owner != sample:
                    continue
                parent = int(data.states.parent_index[node])
                teacher_dense[node] = len(dense_rows)
                dense_rows.append((*ids, *([-1] * (width - len(ids)))))
                dense_sample.append(sample)
                parents.append(teacher_dense[parent] if parent >= 0 else source_row)
                added.append(int(data.states.added_reaction_index[node]))
                terminal.append(True)
        # Competition must include unobserved alternatives as well as the
        # positive teacher paths. Use every exhaustively enumerated state.
        for sample in range(data.num_branch_groups):
            existing = {
                tuple(sorted(i for i in row if i >= 0))
                for s, row in zip(dense_sample, dense_rows)
                if s == sample
            }
            root = next(
                i
                for i, (s, row) in enumerate(zip(dense_sample, dense_rows))
                if s == sample and all(r < 0 for r in row)
            )
            for sequence in sorted(
                self.last_reaction_sequence_compounds,
                key=lambda seq: len(seq.reactions),
            ):
                ids = tuple(sorted(reactions.index(r) for r in sequence.reactions))
                if ids in existing:
                    continue
                existing.add(ids)
                dense_rows.append((*ids, *([-1] * (width - len(ids)))))
                dense_sample.append(sample)
                parents.append(root)
                added.append(-1)
                terminal.append(True)
        synthetic = ReactionDecoderOutput(
            torch.tensor(dense_sample),
            torch.tensor(dense_rows, dtype=torch.long),
            torch.zeros(len(dense_rows)),
            torch.tensor(terminal),
            torch.tensor(parents),
            torch.tensor(added),
            SimpleNamespace(
                reaction_index=torch.arange(len(reactions)).expand(
                    data.num_branch_groups, -1
                )
            ),
        )
        from ..specgen.components.reaction.reaction_decoder import (
            build_reaction_pool,
            with_all_transitions,
        )

        synthetic = replace(
            synthetic, pool=build_reaction_pool(torch.zeros((len(reactions), 1)), data)
        )
        from ..specgen.components.reaction.reaction_compatibility import (
            ReactionCompatibilityEngine,
        )

        synthetic = with_all_transitions(synthetic, ReactionCompatibilityEngine(width))
        decoded = materialize_reaction_states(
            synthetic,
            (source,),
            (reactions,),
            data.branch_groups.source_compound_index,
            generator.mol_encoder.graph_builder,
        )
        node_by_state = {
            (dense_sample[row], tuple(i for i in dense_rows[row] if i >= 0)): node
            for row, node in zip(
                decoded.materialized_state_index.tolist(),
                decoded.materialized_node_index.tolist(),
            )
        }
        state_nodes = torch.tensor(
            [
                node_by_state.get((sample, tuple(row)), -1)
                for sample, row in zip(samples, rows)
            ],
            dtype=torch.long,
        )
        annotations = []
        # (node, str(adduct)) -> maximum matched relative peak intensity.  This
        # chemistry/assignment work is prepared once here; the training loop
        # only consumes tensors and never reruns Fragmenter or RDKit.
        for sample, (peak_annotations, adduct, energy, mzs, intensities) in enumerate(
            zip(
                assignments,
                precursor_types,
                collision_energy,
                peaks_mz,
                peaks_intensity,
            )
        ):
            branch_group = sample_groups[sample]
            main = fragmenter._resolve_main_adduct_type(adduct)
            precursor_mz = adduct.apply_to_formula(source.formula).normalized.exact_mass
            peaks = []
            for peak_index, (mz, intensity, explanations) in enumerate(
                zip(mzs, intensities, peak_annotations)
            ):
                matches = []
                for annotation in explanations:
                    sequence = annotation.reaction_sequence
                    state = (
                        branch_group,
                        (
                            tuple(
                                sorted(
                                    reactions.index(reaction)
                                    for reaction in sequence.reactions
                                )
                            )
                            if sequence
                            else ()
                        ),
                    )
                    if state not in node_by_state:
                        continue
                    node = node_by_state[state]
                    formula = annotation.formula
                    match = dict(
                        smiles=annotation.smiles,
                        formula=str(formula),
                        theoreticalMz=float(formula.exact_mass),
                        massErrorPpm=(float(mz) - formula.exact_mass)
                        / formula.exact_mass
                        * 1e6,
                        adduct=str(annotation.adduct),
                        fragmentIonAdduct=str(annotation.fragment_ion_adduct),
                        hydrogenShift=annotation.hydrogen_shift,
                    )
                    existing = next(
                        (
                            item
                            for item in matches
                            if {k: v for k, v in item.items() if k != "nodeIndices"}
                            == match
                        ),
                        None,
                    )
                    if existing is None:
                        matches.append(dict(nodeIndices=[node], **match))
                    elif node not in existing["nodeIndices"]:
                        existing["nodeIndices"] = sorted(
                            (*existing["nodeIndices"], node)
                        )
                peaks.append(
                    dict(
                        index=peak_index,
                        mz=float(mz),
                        intensity=float(intensity),
                        precursor=bool(
                            fragmenter.mass_tolerance.within(float(mz), precursor_mz)
                        ),
                        matches=matches,
                    )
                )

            def score(selected):
                total = sum(peak["intensity"] for peak in selected)
                return (
                    sum(peak["intensity"] for peak in selected if peak["matches"])
                    / total
                    if total > 0
                    else None
                )

            annotations.append(
                dict(
                    adduct=str(adduct),
                    mainAdduct=str(main),
                    collisionEnergy=float(energy),
                    precursorMz=float(precursor_mz),
                    peaks=peaks,
                    assignmentScore=score(peaks),
                    assignmentScoreWithoutPrecursor=score(
                        [peak for peak in peaks if not peak["precursor"]]
                    ),
                )
            )
        spectrum_batch = prepare_spectrum_model_batch(
            decoded,
            fragmenter,
            precursor_types,
            generator.tensorizer,
            sample_branch_group_index=data.sample_branch_group_index,
            ion_type_strs=getattr(generator, "ion_type_strs", None),
        )
        target = []
        valid_formulas = (
            (spectrum_batch.state_formula_tensor[:, :-1] >= 0).all(1).tolist()
        )
        for sample, mz, valid_formula in zip(
            spectrum_batch.state_sample_index.tolist(),
            spectrum_batch.state_mz.tolist(),
            valid_formulas,
        ):
            matches = [
                float(intensity)
                for observed, intensity in zip(
                    peaks_mz[sample], peaks_intensity[sample]
                )
                if valid_formula
                and mz > 0
                and fragmenter.mass_tolerance.within(observed, mz)
            ]
            target.append(max(matches, default=0.0))
        # The base state universe includes all materialized chemistry, not only
        # paths selected by the teacher or its one-step negative continuations.
        set_rows = data.reaction_sets.rows()
        candidate_rows = [
            set_rows[i] for i in data.states.candidate_reaction_set_index.tolist()
        ]
        candidate_rows = [tuple(row) for row in candidate_rows]
        candidate_groups = data.states.candidate_branch_group_index.tolist()
        known = set(zip(candidate_groups, candidate_rows))
        for group, row in zip(dense_sample, dense_rows):
            key = (group, tuple(i for i in row if i >= 0))
            if key not in known and key in node_by_state:
                known.add(key)
                candidate_groups.append(group)
                candidate_rows.append(key[1])
        reaction_sets, context_set_ids = ReactionSet.from_rows(candidate_rows)
        candidate_nodes = torch.tensor(
            [
                node_by_state.get((group, row), -1)
                for group, row in zip(candidate_groups, candidate_rows)
            ],
            dtype=torch.long,
        )
        return (
            replace(
                data,
                reaction_sets=reaction_sets,
                formula_elements=tuple(generator.tensorizer.element_order),
                observations=tuple(annotations),
                spectrum_batch=replace(
                    spectrum_batch, target_intensity=torch.tensor(target)
                ),
                states=replace(
                    data.states,
                    fragment_node_index=state_nodes,
                    candidate_fragment_index=candidate_nodes,
                    candidate_branch_group_index=torch.tensor(candidate_groups),
                    candidate_reaction_set_index=context_set_ids,
                ),
            ),
            kept,
        )


class ReactionStructureBuilder(_ChemistryPreparationBuilder):
    """Prepare chemistry, then normalize the shared training/prediction components."""

    def build(self, *args, **kwargs):
        from .source_reaction_structure import SourceReactionStructure

        prepared, kept = super().build(*args, **kwargs)
        return SourceReactionStructure.from_preparation(prepared), kept

    def build_prediction(self, source, precursor_types, collision_energy, **kwargs):
        from .source_reaction_structure import SourceReactionStructure

        prepared, kept = super().build(
            source,
            precursor_types,
            collision_energy,
            [[] for _ in precursor_types],
            [[] for _ in precursor_types],
            **kwargs,
        )
        return SourceReactionStructure.from_preparation(prepared).for_prediction(), kept
