"""Pure Python combination search; RDKit is used only after normalization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import heapq

from .ReactionInstance import ReactionInstance
from .ReactionSequence import ReactionSequence
from .ReactionRelations import ReactionRelations


@dataclass
class ReactionSequenceSearchStats:
    num_primitive_reactions: int = 0
    num_raw_combinations: int = 0
    num_hard_conflict_pruned: int = 0
    num_invalidated_reaction_pruned: int = 0
    num_redundancy_pruned: int = 0
    num_duplicate_sequence_pruned: int = 0
    num_duplicate_effect_pruned: int = 0
    num_compiled_sequences: int = 0
    num_rdkit_run_reactants: int = 0
    num_generated_fragments: int = 0
    num_unique_fragment_smiles: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(vars(self))


class FragmentTreeLimitExceeded(ValueError):
    """A single source tree exceeds a configured search limit.

    ``stats`` holds the search statistics observed up to the moment the limit
    was hit (the builder attaches them), so a skipped source stays diagnosable.
    """

    def __init__(
        self, limit_name: str, limit: int, stats: dict[str, int] | None = None
    ) -> None:
        self.limit_name = limit_name
        self.limit = limit
        self.stats = stats
        super().__init__(f"{limit_name} exceeded: limit={limit}, observed>{limit}")

    def __reduce__(self):
        return type(self), (self.limit_name, self.limit, self.stats)


def validate_search_limits(
    max_unique_fragment_smiles: int = -1, max_reaction_combinations: int = -1
) -> None:
    """Both limits are -1 (unlimited) or a positive integer."""
    for name, value in (
        ("max_unique_fragment_smiles", max_unique_fragment_smiles),
        ("max_reaction_combinations", max_reaction_combinations),
    ):
        if type(value) is not int or not (value == -1 or value > 0):
            raise ValueError(f"{name} must be -1 or a positive integer.")


@dataclass(frozen=True)
class _ReactionSequenceCandidate:
    parent_reaction_sequence: ReactionSequence | None
    reaction_sequence: ReactionSequence
    added_reaction: ReactionInstance | None
    is_seed: bool = False


class ReactionSequenceSearch:
    """Explore unordered collections using increasing primitive indices.

    A cursor is separate from the normalized history, since normalization may
    remove a reaction. Different cursors can expose different remaining reactions;
    histories are deduplicated independently from these enumeration positions.
    For a repeated history only its smallest (most permissive) cursor expands.
    Seed reactions are fixed anchors; added indices increase from -1, allowing
    additions whose index is lower than a reaction already in the seed.
    """

    def __init__(
        self,
        reactions: tuple[ReactionInstance, ...],
        *,
        max_reaction_count: int,
        seed_reaction_sequences: Sequence[ReactionSequence] | None = None,
        addable_reactions: Sequence[ReactionInstance] | None = None,
        include_source: bool = False,
        max_reaction_combinations: int = -1,
    ) -> None:
        validate_search_limits(max_reaction_combinations=max_reaction_combinations)
        # Cached seeds are replayed into the tree in prefix order so their
        # original baseline transitions can be reconstructed without running
        # the baseline reactions again.
        seeds = tuple(
            sorted(
                set(seed_reaction_sequences or ()),
                key=lambda s: (len(s.reactions), s.key),
            )
        )
        if any(len(seed.reactions) > max_reaction_count for seed in seeds):
            raise ValueError("Seed exceeds max_reaction_count")
        self.reactions = tuple(
            sorted(
                set(reactions) | {a for seed in seeds for a in seed.reactions},
                key=lambda a: a.key,
            )
        )
        self.index_by_reaction = {a: i for i, a in enumerate(self.reactions)}
        self.addable_reactions = frozenset(
            reactions if addable_reactions is None else addable_reactions
        )
        if not self.addable_reactions <= set(self.reactions):
            raise ValueError("addable_reactions must belong to the reaction universe")
        self.relations = ReactionRelations.from_reactions(self.reactions)
        self.max_reaction_count = max_reaction_count
        self.seeds = seeds
        self.include_source = include_source
        self.max_reaction_combinations = max_reaction_combinations
        self.stats = ReactionSequenceSearchStats(
            num_primitive_reactions=len(self.addable_reactions)
        )
        self.visited_sequence_keys: set[tuple[tuple[object, ...], ...]] = set()

    def enumerate(self) -> tuple[_ReactionSequenceCandidate, ...]:
        return tuple(self.iter_candidates())

    def iter_candidates(self):
        """Yield each connection lazily so tree limits can stop combination search.

        Raises FragmentTreeLimitExceeded as soon as num_raw_combinations
        exceeds max_reaction_combinations. The count spans every size, from
        one reaction up to max_reaction_count, and is checked before a candidate
        is yielded, so no further candidate reaches RDKit materialization.
        """
        connections: set[_ReactionSequenceCandidate] = set()
        # Heap prioritizes the normalized reaction count, with a stable serial tie-break.
        frontier: list[tuple[int, int, ReactionSequence | None, int]] = []
        minimum_cursor: dict[ReactionSequence | None, int] = {}
        serial = 0
        for seed in self.seeds:
            rejection = self.relations.rejection(
                tuple(self.index_by_reaction[a] for a in seed.reactions)
            )
            if rejection:
                raise ValueError(f"Invalid seed reaction sequence: {rejection}")
            self.visited_sequence_keys.add(seed.key)
            candidate = _ReactionSequenceCandidate(None, seed, None, True)
            connections.add(candidate)
            yield candidate
            heapq.heappush(frontier, (len(seed.reactions), serial, seed, -1))
            minimum_cursor[seed] = -1
            serial += 1
        if not self.seeds or self.include_source:
            frontier.append((0, serial, None, -1))
            minimum_cursor[None] = -1
            serial += 1
        while frontier:
            _, _, parent, cursor = heapq.heappop(frontier)
            if minimum_cursor[parent] != cursor:
                continue
            previous = parent.reactions if parent else ()
            if len(previous) >= self.max_reaction_count:
                continue
            for i in range(cursor + 1, len(self.reactions)):
                reaction = self.reactions[i]
                if reaction in previous or reaction not in self.addable_reactions:
                    continue
                self.stats.num_raw_combinations += 1
                if (
                    0
                    <= self.max_reaction_combinations
                    < self.stats.num_raw_combinations
                ):
                    raise FragmentTreeLimitExceeded(
                        "max_reaction_combinations", self.max_reaction_combinations
                    )
                indices = tuple(self.index_by_reaction[a] for a in previous) + (i,)
                rejection = self.relations.rejection(indices)
                if rejection:
                    if rejection == "hard_conflict":
                        self.stats.num_hard_conflict_pruned += 1
                    else:
                        self.stats.num_invalidated_reaction_pruned += 1
                    continue
                sequence = ReactionSequence((*previous, reaction))
                if len(sequence.reactions) < len(previous) + 1:
                    self.stats.num_redundancy_pruned += 1
                if (
                    not sequence.retained_atom_maps
                    or len(sequence.reactions) > self.max_reaction_count
                ):
                    continue
                if sequence == parent:
                    self.stats.num_duplicate_sequence_pruned += 1
                    continue
                candidate = _ReactionSequenceCandidate(parent, sequence, reaction)
                if sequence.key in self.visited_sequence_keys:
                    self.stats.num_duplicate_sequence_pruned += 1
                else:
                    self.visited_sequence_keys.add(sequence.key)
                if candidate not in connections:
                    connections.add(candidate)
                    yield candidate
                if sequence not in minimum_cursor or i < minimum_cursor[sequence]:
                    minimum_cursor[sequence] = i
                    heapq.heappush(
                        frontier, (len(sequence.reactions), serial, sequence, i)
                    )
                    serial += 1
