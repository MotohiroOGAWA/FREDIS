from __future__ import annotations

from collections.abc import Sequence, Iterable
from dataclasses import dataclass
import time
from typing import Any

from rdkit import Chem
from ....libs.mmkit.mmkit import Compound
from ..reaction._ReactionPattern import _ReactionPattern
from ..reaction.ReactionPatternSet import ReactionPatternSet
from ..reaction.ReactionInstance import ReactionInstance, mapped_source
from ..reaction.ReactionInstanceGenerator import create_reaction_instances
from ..reaction.ReactionSequence import ReactionSequence
from ..reaction.ReactionSequenceResult import ReactionSequenceResult
from ..reaction.ReactionSequenceSearch import (
    ReactionSequenceSearch,
    FragmentTreeLimitExceeded,
    _ReactionSequenceCandidate,
    validate_search_limits,
)
from .ReactionSequenceTransition import ReactionSequenceTransition
from .FragmentEdge import FragmentEdge
from .FragmentNode import FragmentNode
from .FragmentTree import FragmentTree


@dataclass(frozen=True)
class _FragmentExpansionState:
    node_index: int
    reaction_sequence: ReactionSequence | None


@dataclass(frozen=True)
class FragmentTreeBuilder:
    """Build an in-memory tree from combinations anchored to Original Source.

    Primitive matching happens once per pattern on Source. Edges record reaction
    state transitions; generated fragments are never used as reaction inputs.
    """

    max_reaction_count: int
    reaction_pattern_set: ReactionPatternSet

    def __post_init__(self) -> None:
        if type(self.max_reaction_count) is not int or self.max_reaction_count <= 0:
            raise ValueError("max_reaction_count must be a positive integer.")
        if not isinstance(self.reaction_pattern_set, ReactionPatternSet):
            raise TypeError("reaction_pattern_set must be a ReactionPatternSet.")

    @property
    def reaction_patterns(self) -> tuple[_ReactionPattern, ...]:
        return self.reaction_pattern_set.patterns

    @property
    def name(self) -> str:
        return self.reaction_pattern_set.name

    def create_reaction_instances(
        self, source_compound: Compound, *, pattern_ids: frozenset[int] | None = None
    ) -> tuple[ReactionInstance, ...]:
        if not isinstance(source_compound, Compound):
            raise TypeError("source_compound must be a Compound.")
        pattern_set = self.reaction_pattern_set
        if pattern_ids is not None:
            pattern_set = ReactionPatternSet(
                pattern_set.name,
                tuple(
                    pattern
                    for pattern in pattern_set
                    if pattern.pattern_id in pattern_ids
                ),
            )
        return create_reaction_instances(source_compound, pattern_set)

    def _reaction_limit(self, max_reaction_count: int | None) -> int:
        if max_reaction_count is None:
            return self.max_reaction_count
        if type(max_reaction_count) is not int or max_reaction_count < 0:
            raise ValueError(
                "max_reaction_count must be None or a non-negative integer."
            )
        return max_reaction_count

    def _search(
        self,
        source_compound: Compound,
        *,
        seed_reaction_sequences: Sequence[ReactionSequence] | None,
        addable_pattern_ids: frozenset[int] | None = None,
        include_source_seed: bool = False,
        max_reaction_count: int | None,
        max_reaction_combinations: int = -1,
        precomputed_reactions: Sequence[ReactionInstance] | None = None,
        precomputed_relations=None,
    ) -> tuple[ReactionSequenceSearch, Iterable[_ReactionSequenceCandidate]]:
        if not isinstance(source_compound, Compound):
            raise TypeError("source_compound must be a Compound.")
        limit = self._reaction_limit(max_reaction_count)
        seeds = tuple(seed_reaction_sequences or ())
        if any(not isinstance(seed, ReactionSequence) for seed in seeds):
            raise TypeError(
                "seed_reaction_sequences must contain ReactionSequence instances"
            )
        if any(len(seed.reactions) > limit for seed in seeds):
            raise ValueError("Seed exceeds max_reaction_count")
        universe = frozenset(
            a.GetAtomMapNum() for a in mapped_source(source_compound).GetAtoms()
        )
        for seed in seeds:
            if any(
                a.retained_atom_maps | a.discarded_atom_maps != universe
                for a in seed.reactions
            ):
                raise ValueError("Seed reaction universe differs from Original Source")
        # Callers that already enumerated Source reactions (for example once per
        # pattern and compound) pass them here instead of matching again.
        if precomputed_reactions is not None:
            reactions = tuple(
                sorted(set(precomputed_reactions), key=lambda reaction: reaction.key)
            )
        else:
            reactions = (
                self.create_reaction_instances(source_compound)
                if addable_pattern_ids is None
                else self.create_reaction_instances(
                    source_compound, pattern_ids=addable_pattern_ids
                )
            )
        addable_reactions = None if addable_pattern_ids is None else reactions
        search = ReactionSequenceSearch(
            reactions,
            relations=precomputed_relations,
            max_reaction_count=limit,
            seed_reaction_sequences=seeds,
            addable_reactions=addable_reactions,
            include_source=include_source_seed,
            max_reaction_combinations=max_reaction_combinations,
        )
        return search, search.iter_candidates()

    def _materialize(self, source_compound, search, candidates):
        return {
            candidate.reaction_sequence: result
            for candidate, result in self._materialize_iter(
                source_compound, search, candidates
            )
        }

    def _materialize_iter(
        self,
        source_compound: Compound,
        search: ReactionSequenceSearch,
        candidates: Iterable[_ReactionSequenceCandidate],
        *,
        max_unique_fragment_smiles: int = -1,
        seed_compounds: dict[ReactionSequence, Compound] | None = None,
    ):
        """Compile/run each unique effect once, retaining every reaction history.

        Every newly generated fragment's canonical SMILES joins a set that
        excludes the Source SMILES. Once that set grows beyond
        max_unique_fragment_smiles materialization stops immediately, before
        the next retained candidate is executed. Combination enumeration has
        already finished in the tree-building pipeline.
        """
        results: dict[ReactionSequence, ReactionSequenceResult] = {}
        unique_fragment_smiles: set[str] = set()
        source_smiles = source_compound.smiles
        effect_cache: dict[tuple[object, ...], tuple[Compound, str] | None] = {}
        visited_effect_keys: set[tuple[object, ...]] = set()
        seed_compounds = seed_compounds or {}
        for candidate in candidates:
            sequence = candidate.reaction_sequence
            if sequence in results:
                yield candidate, results[sequence]
                continue
            effect = sequence.effect_key
            if candidate.is_seed and sequence in seed_compounds:
                cached = (seed_compounds[sequence], "")
                visited_effect_keys.add(effect)
                effect_cache[effect] = cached
            elif effect in visited_effect_keys:
                search.stats.num_duplicate_effect_pruned += 1
                cached = effect_cache[effect]
            else:
                visited_effect_keys.add(effect)
                try:
                    reaction = sequence.compile(source_compound)
                    search.stats.num_compiled_sequences += 1
                    search.stats.num_rdkit_run_reactants += 1
                    products = reaction.run(source_compound)
                    if len(products) != 1:
                        raise ValueError(
                            "Source-specific reaction must generate exactly one target"
                        )
                    cached = (Compound(products[0]), reaction.smirks)
                    search.stats.num_generated_fragments += 1
                    smiles = cached[0].smiles
                    if smiles != source_smiles and smiles not in unique_fragment_smiles:
                        unique_fragment_smiles.add(smiles)
                        search.stats.num_unique_fragment_smiles = len(
                            unique_fragment_smiles
                        )
                        if (
                            0
                            <= max_unique_fragment_smiles
                            < len(unique_fragment_smiles)
                        ):
                            raise FragmentTreeLimitExceeded(
                                "max_unique_fragment_smiles", max_unique_fragment_smiles
                            )
                except Chem.rdchem.MolSanitizeException:
                    # A graph-valid collection can still violate chemical valence.
                    if candidate.is_seed:
                        raise
                    cached = None
                effect_cache[effect] = cached
            if cached is not None:
                target, smirks = cached
                results[sequence] = ReactionSequenceResult(sequence, target, smirks)
                yield candidate, results[sequence]

    def react_all(
        self,
        source_compound: Compound,
        *,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        max_reaction_count: int | None = None,
    ) -> tuple[ReactionSequenceResult, ...]:
        """Iterate normalized unordered histories and react once per unique Source effect."""
        search, candidates = self._search(
            source_compound,
            seed_reaction_sequences=seed_reaction_sequences,
            max_reaction_count=max_reaction_count,
        )
        results = self._materialize(source_compound, search, candidates)
        return tuple(
            sorted(
                results.values(),
                key=lambda r: (
                    len(r.reaction_sequence.reactions),
                    r.reaction_sequence.key,
                ),
            )
        )

    def build(
        self,
        compound: Compound,
        *,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        seed_compounds: dict[ReactionSequence, Compound] | None = None,
        addable_pattern_ids: frozenset[int] | None = None,
        max_unique_fragment_smiles: int = -1,
        max_reaction_combinations: int = -1,
        max_reaction_count: int | None = None,
        print_info: bool = False,
        precomputed_reactions: Sequence[ReactionInstance] | None = None,
        precomputed_relations=None,
    ) -> FragmentTree:
        return self._build_result(
            compound,
            seed_reaction_sequences=seed_reaction_sequences,
            seed_compounds=seed_compounds,
            addable_pattern_ids=addable_pattern_ids,
            precomputed_reactions=precomputed_reactions,
            precomputed_relations=precomputed_relations,
            max_unique_fragment_smiles=max_unique_fragment_smiles,
            max_reaction_combinations=max_reaction_combinations,
            max_reaction_count=max_reaction_count,
            print_info=print_info,
        )["fragment_tree"]

    def _build_result(
        self,
        compound: Compound,
        *,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        seed_compounds: dict[ReactionSequence, Compound] | None = None,
        addable_pattern_ids: frozenset[int] | None = None,
        max_unique_fragment_smiles: int = -1,
        max_reaction_combinations: int = -1,
        max_reaction_count: int | None = None,
        print_info: bool = False,
        precomputed_reactions: Sequence[ReactionInstance] | None = None,
        precomputed_relations=None,
    ) -> dict[str, Any]:
        """Build one source tree; -1 disables either search limit.

        max_reaction_combinations bounds cumulative raw combination attempts.
        Enumeration finishes before RDKit materialization. Overflow at depth n
        discards depth n and retains only complete depths through n - 1.
        max_unique_fragment_smiles bounds distinct generated fragment SMILES,
        excluding the Source itself, and is enforced as each fragment is
        materialized. A FragmentTreeLimitExceeded for the structure-count limit
        carries the statistics observed up to that point in ``stats``.
        """
        validate_search_limits(max_unique_fragment_smiles, max_reaction_combinations)
        started = time.monotonic()
        source_compound = compound
        search, candidates = self._search(
            source_compound,
            seed_reaction_sequences=seed_reaction_sequences,
            max_reaction_count=max_reaction_count,
            addable_pattern_ids=addable_pattern_ids,
            include_source_seed=seed_compounds is not None,
            max_reaction_combinations=max_reaction_combinations,
            precomputed_reactions=precomputed_reactions,
            precomputed_relations=precomputed_relations,
        )
        # Exhaust the pure combination search before the first compile/run.
        # An incomplete depth is discarded without ever materializing it.
        try:
            candidates = search.enumerate_completed_depths()
            return self._build_from_candidates(
                source_compound,
                search,
                candidates,
                seed_reaction_sequences=seed_reaction_sequences,
                seed_compounds=seed_compounds,
                max_unique_fragment_smiles=max_unique_fragment_smiles,
                print_info=print_info,
                started=started,
            )
        except FragmentTreeLimitExceeded as error:
            error.stats = search.stats.to_dict()
            raise

    def _build_from_candidates(
        self,
        source_compound,
        search,
        candidates,
        *,
        seed_reaction_sequences,
        seed_compounds,
        max_unique_fragment_smiles,
        print_info,
        started,
    ) -> dict[str, Any]:
        state = _FragmentTreeBuildState(source_compound.smiles)
        compounds = {0: source_compound.copy()}
        expansion_by_sequence: dict[
            ReactionSequence | None, _FragmentExpansionState
        ] = {None: _FragmentExpansionState(0, None)}
        processed_states: set[tuple[int, tuple[tuple[object, ...], ...] | None]] = set()
        pending_predecessors = {}
        for candidate, result in self._materialize_iter(
            source_compound,
            search,
            candidates,
            max_unique_fragment_smiles=max_unique_fragment_smiles,
            seed_compounds=seed_compounds,
        ):
            sequence = candidate.reaction_sequence
            parent_sequence = candidate.parent_reaction_sequence
            added_reaction = candidate.added_reaction
            parent = expansion_by_sequence.get(parent_sequence)
            incremental_seed = candidate.is_seed and seed_compounds is not None
            if incremental_seed:
                # A cache-backed seed represents an existing baseline state,
                # not a new root. Recreate one of its original one-reaction
                # transitions so baseline-only assignments remain visible in
                # the augmented tree while its chemistry stays cached.
                parent = None
                for added in sequence.reactions:
                    remaining = tuple(
                        reaction for reaction in sequence.reactions if reaction != added
                    )
                    predecessor = ReactionSequence(remaining) if remaining else None
                    alternate = expansion_by_sequence.get(predecessor)
                    if alternate is not None:
                        parent = alternate
                        parent_sequence = predecessor
                        added_reaction = added
                        break
            if parent is None:
                # Canonical combination enumeration is not a chemical execution
                # order. A primitive target can fail valence while its combined
                # target is valid. Link to an already generated alternative
                # predecessor without re-enumerating or executing the effect.
                for added in sequence.reactions:
                    remaining = tuple(a for a in sequence.reactions if a != added)
                    predecessor = ReactionSequence(remaining) if remaining else None
                    alternate = expansion_by_sequence.get(predecessor)
                    if alternate is not None and (
                        predecessor is not None or not seed_reaction_sequences
                    ):
                        normalized = ReactionSequence(
                            (*(predecessor.reactions if predecessor else ()), added)
                        )
                        if normalized == sequence:
                            parent = alternate
                            parent_sequence = predecessor
                            added_reaction = added
                            break
                if parent is None:
                    continue
            target_index = state.get_or_create_node_index(result.compound.smiles)
            child = _FragmentExpansionState(target_index, sequence)
            expansion_by_sequence[sequence] = child
            compounds.setdefault(target_index, result.compound)
            processed_states.add((target_index, sequence.key))
            transition = ReactionSequenceTransition(
                added_reaction,
                sequence,
                parent_sequence,
                candidate.is_seed and not incremental_seed,
            )
            state.add_fragment_edge(parent.node_index, target_index, transition)
            # Add alternative routes immediately, including routes whose predecessor
            # arrives later. Count distinct transitions rather than node pairs.
            for target_sequence, added in pending_predecessors.pop(sequence, []):
                destination = expansion_by_sequence[target_sequence]
                state.add_fragment_edge(
                    child.node_index,
                    destination.node_index,
                    ReactionSequenceTransition(added, target_sequence, sequence),
                )
            for added in sequence.reactions:
                remaining = tuple(a for a in sequence.reactions if a != added)
                predecessor = ReactionSequence(remaining) if remaining else None
                if predecessor is None and seed_reaction_sequences:
                    continue
                if (
                    ReactionSequence(
                        (*(predecessor.reactions if predecessor else ()), added)
                    )
                    != sequence
                ):
                    continue
                alternate = expansion_by_sequence.get(predecessor)
                if alternate is None:
                    pending_predecessors.setdefault(predecessor, []).append(
                        (sequence, added)
                    )
                else:
                    state.add_fragment_edge(
                        alternate.node_index,
                        child.node_index,
                        ReactionSequenceTransition(added, sequence, predecessor),
                    )
        if not seed_reaction_sequences:
            processed_states.add((0, None))
        stats = {**search.stats.to_dict(), **search.completion_stats}
        if print_info:
            print(
                f"Source reaction search: {stats}; nodes={len(state.nodes)}, "
                f"edges={len(state.edges)}, elapsed={time.monotonic() - started:.2f}s"
            )
        return {
            "fragment_tree": state.to_fragment_tree(),
            "fragment_compound_by_index": compounds,
            "expansion_states": tuple(expansion_by_sequence.values()),
            "processed_expansion_states": frozenset(processed_states),
            "search_stats": stats,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_reaction_count": self.max_reaction_count,
            "reaction_pattern_set": self.reaction_pattern_set.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FragmentTreeBuilder:
        return cls(
            max_reaction_count=data["max_reaction_count"],
            reaction_pattern_set=ReactionPatternSet.from_dict(
                data["reaction_pattern_set"]
            ),
        )

    def copy(self) -> FragmentTreeBuilder:
        return FragmentTreeBuilder(
            self.max_reaction_count, self.reaction_pattern_set.copy()
        )


class _FragmentTreeBuildState:
    """Chemical nodes and in-memory edges, independent of expansion histories."""

    def __init__(self, root_smiles: str) -> None:
        self.root_smiles = root_smiles
        self.nodes: dict[int, FragmentNode] = {}
        self.edges: dict[tuple[int, int], FragmentEdge] = {}
        self.transition_count = 0
        self.smiles_to_node_index: dict[str, int] = {}
        self.get_or_create_node_index(root_smiles)

    def get_or_create_node_index(self, smiles: str) -> int:
        if smiles in self.smiles_to_node_index:
            return self.smiles_to_node_index[smiles]
        index = len(self.nodes)
        self.nodes[index] = FragmentNode(index, smiles)
        self.smiles_to_node_index[smiles] = index
        return index

    def add_fragment_edge(
        self,
        source_index: int,
        target_index: int,
        transition: ReactionSequenceTransition,
    ) -> None:
        key = (source_index, target_index)
        if key in self.edges and transition in self.edges[key].transitions:
            return
        self.transition_count += 1
        if key in self.edges:
            self.edges[key] = self.edges[key].with_transition(transition)
            return
        self.edges[key] = FragmentEdge(
            len(self.edges), source_index, target_index, transitions=(transition,)
        )

    def to_fragment_tree(self) -> FragmentTree:
        return FragmentTree.from_nodes_and_edges(
            smiles=self.root_smiles,
            nodes=tuple(self.nodes.values()),
            edges=tuple(self.edges.values()),
        )
