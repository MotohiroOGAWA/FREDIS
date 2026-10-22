from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict, Set, Sequence
import json
from pathlib import Path
from collections import defaultdict

from ...libs.mmkit.mmkit import Adduct, Compound

from ..mass.tolerance import MassTolerance, parse_mass_tolerance, format_mass_tolerance
from ..formula import utils as formula_utils
from .tree import FragmentTree
from .ion_tree import FragmentIonTree, FragmentIonTreeBuilder, FragmentIonAdductRuleSet
from .annotation import PrecursorReaction, ReactionAnnotation, contains, reaction_count
from .reaction import ReactionPatternSet, ReactionSequenceResult, ReactionSequence


@dataclass(frozen=True)
class Fragmenter:
    fragment_ion_tree_builder: FragmentIonTreeBuilder
    mass_tolerance: MassTolerance
    # Precursor selection belongs to Fragmenter, not the tree builder.
    precursor_candidate_max_reaction_count: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.precursor_candidate_max_reaction_count) is not int
            or self.precursor_candidate_max_reaction_count < 0
        ):
            raise ValueError(
                "precursor_candidate_max_reaction_count must be a non-negative integer"
            )

    @property
    def adduct_types(self) -> Tuple[Adduct, ...]:
        return self.adduct_rule_set.adduct_types

    @property
    def adduct_rule_set(self) -> FragmentIonAdductRuleSet:
        return self.fragment_ion_tree_builder.fragment_ion_adduct_rule_set

    @property
    def tree_max_reaction_count(self) -> int:
        return self.fragment_ion_tree_builder.max_reaction_count

    @property
    def reaction_pattern_set(self) -> ReactionPatternSet:
        return self.fragment_ion_tree_builder.reaction_pattern_set

    @property
    def name(self) -> str:
        return self.fragment_ion_tree_builder.name

    def get_index_by_adduct_type(self, adduct_type: Adduct) -> int:
        main_adduct_type = self._resolve_main_adduct_type(adduct_type)
        mapper = self.adduct_rule_set.adduct_type_to_index
        if main_adduct_type not in mapper:
            raise ValueError(
                f"Adduct type not supported by this fragmenter: {adduct_type}"
            )
        return mapper[main_adduct_type]

    def build_fragment_tree(
        self,
        compound: Compound,
        *,
        max_unique_fragment_smiles: int = -1,
        max_reaction_combinations: int = -1,
        max_reaction_count: int | None = None,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        seed_compounds: dict[ReactionSequence, Compound] | None = None,
        addable_pattern_ids: frozenset[int] | None = None,
        print_info: bool = False,
    ) -> FragmentTree:
        return self.fragment_ion_tree_builder.build_fragment_tree(
            compound,
            max_unique_fragment_smiles=max_unique_fragment_smiles,
            max_reaction_combinations=max_reaction_combinations,
            max_reaction_count=max_reaction_count,
            seed_reaction_sequences=seed_reaction_sequences,
            seed_compounds=seed_compounds,
            addable_pattern_ids=addable_pattern_ids,
            print_info=print_info,
        )

    def build_fragment_ion_tree(
        self,
        compound: Compound,
        *,
        max_unique_fragment_smiles: int = -1,
        max_reaction_combinations: int = -1,
        max_reaction_count: int | None = None,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        seed_compounds: dict[ReactionSequence, Compound] | None = None,
        addable_pattern_ids: frozenset[int] | None = None,
        print_info: bool = False,
        precomputed_reactions: Sequence | None = None,
        precomputed_relations=None,
    ) -> FragmentIonTree:
        return self.fragment_ion_tree_builder.build(
            compound,
            max_unique_fragment_smiles=max_unique_fragment_smiles,
            max_reaction_combinations=max_reaction_combinations,
            max_reaction_count=max_reaction_count,
            seed_reaction_sequences=seed_reaction_sequences,
            seed_compounds=seed_compounds,
            addable_pattern_ids=addable_pattern_ids,
            print_info=print_info,
            precomputed_reactions=precomputed_reactions,
            precomputed_relations=precomputed_relations,
        )

    def resolve_precursor_reactions(
        self,
        fragment_ion_tree: FragmentIonTree,
        precursor_type: Adduct,
    ) -> frozenset[PrecursorReaction]:
        """The reaction combination(s) on Source that select precursor_type's ion.

        A node is a precursor when its compound, a shared hydrogen shift and
        the main adduct give the precursor formula. Every reaction combination
        reaching that node within precursor_candidate_max_reaction_count
        reactions is returned. An empty result means no precursor is reachable.
        """
        return self._resolve_precursor_reactions(
            fragment_ion_tree,
            precursor_type,
            self._reaction_sequences_by_node(fragment_ion_tree),
        )

    def annotate_peaks(
        self,
        fragment_ion_tree: FragmentIonTree,
        precursor_type: Adduct,
        peaks_mz: Iterable[float],
    ) -> Tuple[Tuple[ReactionAnnotation, ...], ...]:
        """Annotate each peak of one spectrum with reaction combinations."""
        return self.annotate_peak_sets(
            fragment_ion_tree, ((precursor_type, peaks_mz),)
        )[0]

    def annotate_peak_sets(
        self,
        fragment_ion_tree: FragmentIonTree,
        peak_sets: Iterable[Tuple[Adduct, Iterable[float]]],
    ) -> Tuple[Tuple[Tuple[ReactionAnnotation, ...], ...], ...]:
        """Annotate each peak of several spectra sharing one FragmentIonTree.

        A peak is explained by a formula candidate (fragment node, fragment-ion
        adduct, hydrogen shift) together with a reaction combination that
        reaches that node and contains a precursor reaction combination. Only
        the explanations with the fewest reactions are kept for each peak.
        """
        peak_sets_tuple = tuple(
            (precursor_type, tuple(peaks_mz)) for precursor_type, peaks_mz in peak_sets
        )
        sequences_by_node = self._reaction_sequences_by_node(fragment_ion_tree)
        precursors_by_type: Dict[Adduct, frozenset[PrecursorReaction]] = {}
        records_by_main_adduct: Dict[Adduct, List[int]] = defaultdict(list)
        for record_index, (precursor_type, _) in enumerate(peak_sets_tuple):
            if precursor_type not in precursors_by_type:
                precursors_by_type[precursor_type] = self._resolve_precursor_reactions(
                    fragment_ion_tree, precursor_type, sequences_by_node
                )
            records_by_main_adduct[
                self._resolve_main_adduct_type(precursor_type)
            ].append(record_index)

        candidates: List[List[List[ReactionAnnotation]]] = [
            [[] for _ in peaks_mz] for _, peaks_mz in peak_sets_tuple
        ]
        combinations_cache: Dict[
            Tuple[Adduct, int],
            Tuple[Tuple[ReactionSequence | None, PrecursorReaction], ...],
        ] = {}

        for main_adduct_type, record_indices in records_by_main_adduct.items():
            group = fragment_ion_tree.get_formula_candidate_group(main_adduct_type)
            peak_refs = [
                (record_index, peak_index)
                for record_index in record_indices
                for peak_index in range(len(peak_sets_tuple[record_index][1]))
            ]
            assigned_peaks = formula_utils.assign_formulas_to_peaks(
                peaks_mz=[
                    peak_sets_tuple[record][1][peak] for record, peak in peak_refs
                ],
                formula_candidates=group.formulas,
                mass_tolerance=self.mass_tolerance,
            )
            for (record_index, peak_index), info in zip(peak_refs, assigned_peaks):
                precursor_type = peak_sets_tuple[record_index][0]
                for formula_index in info["matched_formula_indices"]:
                    formula = group.formulas[formula_index]
                    for entry in range(
                        int(group.indptr[formula_index]),
                        int(group.indptr[formula_index + 1]),
                    ):
                        node_index = int(group.node_indices[entry])
                        key = (precursor_type, node_index)
                        if key not in combinations_cache:
                            combinations_cache[key] = (
                                self._reaction_combinations_for_node(
                                    sequences_by_node.get(node_index, ()),
                                    precursors_by_type[precursor_type],
                                )
                            )
                        for sequence, precursor in combinations_cache[key]:
                            candidates[record_index][peak_index].append(
                                ReactionAnnotation(
                                    node_index=node_index,
                                    smiles=fragment_ion_tree.get_node(
                                        node_index
                                    ).smiles,
                                    reaction_sequence=sequence,
                                    precursor=precursor,
                                    adduct=group.candidate_adducts[
                                        int(group.candidate_adduct_indices[entry])
                                    ],
                                    fragment_ion_adduct=group.fragment_ion_adducts[
                                        int(group.fragment_ion_adduct_positions[entry])
                                    ],
                                    hydrogen_shift=int(group.hydrogen_shifts[entry]),
                                    formula=formula,
                                )
                            )

        return tuple(
            tuple(
                self._fewest_reactions(peak_candidates)
                for peak_candidates in record_candidates
            )
            for record_candidates in candidates
        )

    def fragment_all(
        self,
        compound: Compound,
        *,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        max_reaction_count: int | None = None,
    ) -> Tuple[ReactionSequenceResult, ...]:
        """Apply reaction histories to the original Source compound."""
        return self.fragment_ion_tree_builder.react_all(
            compound,
            seed_reaction_sequences=seed_reaction_sequences,
            max_reaction_count=max_reaction_count,
        )

    def _resolve_main_adduct_type(
        self,
        precursor_type: Adduct,
    ) -> Adduct:
        if precursor_type.charge != 1 and precursor_type.charge != -1:
            raise ValueError(
                f"Only singly charged adducts are supported: {precursor_type}"
            )

        matched_adduct_flags, residual_component_adduct = (
            Adduct.split_by_reference_adducts(
                precursor_type,
                reference_adducts=self.adduct_types,
            )
        )

        if sum(matched_adduct_flags) == 0:
            raise ValueError(
                "Adduct does not contain any of the supported adduct types"
                f"({', '.join(str(adduct) for adduct in self.adduct_types)}): "
                f"{precursor_type}"
            )

        if sum(matched_adduct_flags) > 1:
            raise ValueError(
                "Adduct contains multiple supported adduct types"
                f"({', '.join(str(adduct) for adduct in self.adduct_types)}): "
                f"{precursor_type}"
            )

        return self.adduct_types[matched_adduct_flags.index(True)]

    @staticmethod
    def _reaction_sequences_by_node(
        fragment_ion_tree: FragmentIonTree,
    ) -> Dict[int, frozenset[ReactionSequence | None]]:
        """Every reaction combination that reaches each node; Source is reached by None."""
        sequences: Dict[int, Set[ReactionSequence | None]] = defaultdict(set)
        sequences[0].add(None)
        for edge_index in range(fragment_ion_tree.num_edges):
            edge = fragment_ion_tree.get_edge(edge_index)
            for transition in edge.transitions:
                sequences[edge.target_index].add(transition.reaction_sequence)
        return {
            node_index: frozenset(values) for node_index, values in sequences.items()
        }

    def _resolve_precursor_reactions(
        self,
        fragment_ion_tree: FragmentIonTree,
        precursor_type: Adduct,
        sequences_by_node: Dict[int, frozenset[ReactionSequence | None]],
    ) -> frozenset[PrecursorReaction]:
        main_adduct_type = self._resolve_main_adduct_type(precursor_type)
        precursor_formula = precursor_type.apply_to_formula(
            fragment_ion_tree.fragment_compounds[0].formula
        ).normalized
        precursors: Set[PrecursorReaction] = set()
        for node_index, compound in enumerate(fragment_ion_tree.fragment_compounds):
            if not any(
                main_adduct_type.apply_to_formula(
                    shift.apply_to_formula(compound.formula).normalized
                ).normalized
                == precursor_formula
                for shift in fragment_ion_tree.hydrogen_shift_adducts
            ):
                continue
            for sequence in sequences_by_node.get(node_index, ()):
                if (node_index == 0) != (sequence is None):
                    continue
                if (
                    reaction_count(sequence)
                    <= self.precursor_candidate_max_reaction_count
                ):
                    precursors.add(
                        PrecursorReaction(
                            node_index=node_index, reaction_sequence=sequence
                        )
                    )
        return frozenset(precursors)

    def _reaction_combinations_for_node(
        self,
        sequences: Iterable[ReactionSequence | None],
        precursors: frozenset[PrecursorReaction],
    ) -> Tuple[Tuple[ReactionSequence | None, PrecursorReaction], ...]:
        """Combinations reaching a node within the reaction limit, each paired
        with every precursor combination it contains."""
        return tuple(
            (sequence, precursor)
            for sequence in sequences
            if reaction_count(sequence) <= self.tree_max_reaction_count
            for precursor in precursors
            if contains(sequence, precursor.reaction_sequence)
        )

    @staticmethod
    def _fewest_reactions(
        annotations: List[ReactionAnnotation],
    ) -> Tuple[ReactionAnnotation, ...]:
        if not annotations:
            return ()
        fewest = min(annotation.reaction_count for annotation in annotations)
        kept = dict.fromkeys(
            annotation
            for annotation in annotations
            if annotation.reaction_count == fewest
        )
        return tuple(
            sorted(
                kept,
                key=lambda annotation: (
                    annotation.node_index,
                    (
                        annotation.reaction_sequence.key
                        if annotation.reaction_sequence is not None
                        else ()
                    ),
                    annotation.precursor.node_index,
                    (
                        annotation.precursor.reaction_sequence.key
                        if annotation.precursor.reaction_sequence is not None
                        else ()
                    ),
                    str(annotation.adduct),
                    str(annotation.fragment_ion_adduct),
                    annotation.hydrogen_shift,
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        builder_dict = self.fragment_ion_tree_builder.to_dict()

        return {
            "fragment_ion_tree_builder": builder_dict,
            "precursor_candidate_max_reaction_count": self.precursor_candidate_max_reaction_count,
            "mass_tolerance": format_mass_tolerance(self.mass_tolerance),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> Fragmenter:
        builder_dict = dict(data["fragment_ion_tree_builder"])

        return cls(
            fragment_ion_tree_builder=FragmentIonTreeBuilder.from_dict(builder_dict),
            mass_tolerance=parse_mass_tolerance(data["mass_tolerance"]),
            precursor_candidate_max_reaction_count=data.get(
                "precursor_candidate_max_reaction_count", 0
            ),
        )

    def to_json(self, path: str | Path) -> None:
        path = Path(path)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> Fragmenter:
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def copy(self) -> Fragmenter:
        """Return a copy of this fragmenter."""
        return Fragmenter(
            fragment_ion_tree_builder=self.fragment_ion_tree_builder.copy(),
            mass_tolerance=self.mass_tolerance,
            precursor_candidate_max_reaction_count=self.precursor_candidate_max_reaction_count,
        )
