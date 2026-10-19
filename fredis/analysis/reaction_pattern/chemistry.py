"""Worker-side chemistry: every tree condition of one compound in one task.

A task carries one compound and the conditions to calculate for it, parents
first.  Source reactions are enumerated once per (compound, pattern) and cached
for the whole task, so a pattern set used at several tree nodes is matched
only once.  A root condition is a full fragment-tree build from its reactions;
a child condition is seeded with its parent's reaction-sequence compounds and
may only add reactions of the patterns its pattern set introduces (every
history containing a new reaction has a parent-only prefix shorter than
``max_reaction_count``, so this equals a full build of the child condition).
Every saved tree keeps its reaction-sequence compounds so that later runs can
add children below it without recalculating it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import time
import traceback

from .constants import (
    COMPLETED,
    FAILED,
    SKIP_ERROR,
    SKIP_LIMIT_EXCEEDED,
    SKIP_PARENT_UNAVAILABLE,
    SKIP_UNRESOLVED_PRECURSOR,
    SKIPPED,
)


@dataclass
class CompoundInput:
    compound_id: str
    smiles: str
    source_positions: list[int]
    sample_ids: list[int]
    dataset_ids: list[int]
    record_indexes: list[int]
    adducts: list[str]
    collision_energies: list[float]
    peaks_mz: list[list[float]]
    peaks_intensity: list[list[float]]


@dataclass
class ConditionStep:
    """One condition to calculate; ``parent_condition_id`` is None for a root."""

    condition_id: int
    parent_condition_id: int | None
    pattern_ids: tuple[int, ...]
    added_pattern_ids: tuple[int, ...]
    output_file: str


@dataclass
class ResolvedCondition:
    """A condition already available for this compound (e.g. from a previous result)."""

    condition_id: int
    status: str
    kept: tuple[int, ...]
    seed_file: str = ""


@dataclass
class ChemistryTask:
    compound: CompoundInput
    steps: list[ConditionStep]
    resolved: dict[int, ResolvedCondition] = field(default_factory=dict)


@dataclass
class ChemistryOptions:
    chemistry_config: dict
    max_unique_fragment_smiles: int = -1
    max_reaction_combinations: int = -1


_STATE: dict = {}
_SEED_CACHE_SIZE = 8


def initialize(options: ChemistryOptions) -> None:
    """Create one builder per process over the analysis-wide pattern library."""
    from fredis.ml.data_preparation.fragment_tree.context import (
        create_preparation_context,
    )
    from fredis.ml.input.structure_builder import ReactionStructureBuilder

    generator = create_preparation_context(options.chemistry_config)
    _STATE.clear()
    _STATE["builder"] = ReactionStructureBuilder(
        generator,
        max_unique_fragment_smiles=options.max_unique_fragment_smiles,
        max_reaction_combinations=options.max_reaction_combinations,
    )
    _STATE["max_reaction_count"] = int(
        options.chemistry_config["fragmenter_params"]["fragment_ion_tree_builder"].get(
            "max_reaction_count", 2
        )
    )
    _STATE["seeds"] = OrderedDict()


def run_tasks(tasks: list[ChemistryTask]) -> list[dict]:
    return [run_task(task) for task in tasks]


def _classify(error: Exception) -> tuple[str, str, str]:
    from fredis.domain.fragment.tree.FragmentTreeBuilder import (
        FragmentTreeLimitExceeded,
    )
    from fredis.ml.input.source_reaction_structure import UnresolvedPrecursorError

    if isinstance(error, FragmentTreeLimitExceeded):
        return SKIPPED, SKIP_LIMIT_EXCEEDED, ""
    if isinstance(error, UnresolvedPrecursorError):
        return SKIPPED, SKIP_UNRESOLVED_PRECURSOR, ""
    return FAILED, SKIP_ERROR, traceback.format_exc()


class _CompoundReactions:
    """Source reactions of one compound, enumerated once per pattern."""

    def __init__(self, smiles: str):
        from fredis.libs.mmkit.mmkit import Compound

        self.source = Compound.from_smiles(smiles)
        self.by_pattern: dict[int, tuple] = {}

    def __call__(self, pattern_ids) -> tuple:
        from fredis.domain.fragment.reaction import (
            ReactionPatternSet,
            create_reaction_instances,
        )

        pattern_set = _STATE[
            "builder"
        ].generator.fragmenter.fragment_ion_tree_builder.reaction_pattern_set
        result = []
        for pattern_id in pattern_ids:
            if pattern_id not in self.by_pattern:
                pattern = pattern_set.get_pattern(int(pattern_id))
                self.by_pattern[pattern_id] = create_reaction_instances(
                    self.source, ReactionPatternSet(pattern.name, (pattern,))
                )
            result.extend(self.by_pattern[pattern_id])
        return tuple(result)


def _seeds_from_file(seed_file: str) -> dict | None:
    """Reaction-sequence compounds below max depth, cached per process."""
    from fredis.analysis.prepared_result import load_prepared_structure
    from fredis.libs.mmkit.mmkit import Compound

    cache: OrderedDict = _STATE["seeds"]
    if seed_file in cache:
        cache.move_to_end(seed_file)
        return cache[seed_file]
    _, metadata = load_prepared_structure(seed_file)
    stored = metadata.get("reaction_sequence_compounds")
    limit = _STATE["max_reaction_count"]
    seeds = (
        None
        if not stored
        else {
            sequence: Compound.from_smiles(mapped_smiles)
            for sequence, mapped_smiles in stored
            if len(sequence.reactions) < limit
        }
    )
    cache[seed_file] = seeds
    while len(cache) > _SEED_CACHE_SIZE:
        cache.popitem(last=False)
    return seeds


def run_task(task: ChemistryTask) -> dict:
    compound = task.compound
    reactions = _CompoundReactions(compound.smiles)
    # condition_id -> (status, kept positions, in-memory seeds or None, seed file)
    state: dict[int, tuple[str, tuple[int, ...], dict | None, str]] = {
        key: (value.status, tuple(value.kept), None, value.seed_file)
        for key, value in task.resolved.items()
    }
    results = []
    for step in task.steps:
        started = time.monotonic()
        base = dict(
            compound_id=compound.compound_id,
            condition_id=step.condition_id,
            parent_condition_id=step.parent_condition_id,
        )
        builder = _STATE["builder"]
        builder.last_search_stats = None
        incremental = False
        try:
            if step.parent_condition_id is None:
                positions, seeds = list(compound.source_positions), None
            else:
                parent = state.get(step.parent_condition_id)
                if parent is None or parent[0] != COMPLETED:
                    results.append(
                        dict(
                            base,
                            status=SKIPPED,
                            skip_category=SKIP_PARENT_UNAVAILABLE,
                            reason="The parent condition is unavailable for this compound.",
                            error="",
                            kept=[],
                            counts=[],
                            file="",
                            incremental=False,
                            search_stats=None,
                            seconds=0.0,
                        )
                    )
                    state[step.condition_id] = (SKIPPED, (), None, "")
                    continue
                positions = list(parent[1])
                seeds = (
                    parent[2]
                    if parent[2] is not None
                    else (_seeds_from_file(parent[3]) if parent[3] else None)
                )
                incremental = seeds is not None
            result = _build(
                builder,
                compound,
                step,
                positions,
                reactions,
                seeds if incremental else None,
            )
        except Exception as error:  # one compound/condition never aborts the run
            status, category, detail = _classify(error)
            results.append(
                dict(
                    base,
                    status=status,
                    skip_category=category,
                    reason=f"{type(error).__name__}: {error}",
                    error=detail,
                    kept=[],
                    counts=[],
                    file="",
                    incremental=incremental,
                    search_stats=builder.last_search_stats,
                    seconds=time.monotonic() - started,
                )
            )
            state[step.condition_id] = (status, (), None, "")
            continue
        limit = _STATE["max_reaction_count"]
        seeds_out = {
            sequence: value
            for sequence, value in builder.last_reaction_sequence_compounds.items()
            if len(sequence.reactions) < limit
        }
        state[step.condition_id] = (
            COMPLETED,
            tuple(result["kept"]),
            seeds_out,
            result["file"],
        )
        results.append(
            dict(
                base,
                status=COMPLETED,
                skip_category="",
                reason="",
                error="",
                incremental=incremental,
                search_stats=builder.last_search_stats,
                seconds=time.monotonic() - started,
                **result,
            )
        )
    return {"compound_id": compound.compound_id, "steps": results}


def _build(
    builder,
    compound: CompoundInput,
    step: ConditionStep,
    positions: list[int],
    reactions: _CompoundReactions,
    seeds: dict | None,
) -> dict:
    from fredis.libs.mmkit.mmkit import Adduct, Compound
    from fredis.ml.data_preparation.fragment_tree.create_training_data import (
        _saved_reaction_registry,
    )
    from fredis.ml.input.source_reaction_structure import save_fragment_tree_structure

    index = {
        position: offset for offset, position in enumerate(compound.source_positions)
    }
    rows = [index[position] for position in positions]
    pick = lambda values: [values[row] for row in rows]
    if seeds is None:
        options = dict(precomputed_reactions=reactions(step.pattern_ids))
    else:
        options = dict(
            seed_compounds=seeds,
            addable_pattern_ids=frozenset(step.added_pattern_ids),
            precomputed_reactions=reactions(step.added_pattern_ids),
        )
    structure, relative_kept = builder.build(
        Compound.from_smiles(compound.smiles),
        [Adduct.parse(value) for value in pick(compound.adducts)],
        pick(compound.collision_energies),
        pick(compound.peaks_mz),
        pick(compound.peaks_intensity),
        **options,
    )
    relative_kept = [int(value) for value in relative_kept]
    kept = [int(positions[value]) for value in relative_kept]
    if step.parent_condition_id is not None and kept != positions:
        raise RuntimeError(
            "The child condition did not retain the complete sample population of its parent"
        )
    annotations = structure.sample_annotations
    metadata = dict(
        smiles=compound.smiles,
        compound_id=compound.compound_id,
        detail_kind="condition",
        condition_id=int(step.condition_id),
        pattern_ids=list(step.pattern_ids),
        added_pattern_ids=list(step.added_pattern_ids),
        incremental=seeds is not None,
        sample_ids=[pick(compound.sample_ids)[value] for value in relative_kept],
        dataset_ids=[pick(compound.dataset_ids)[value] for value in relative_kept],
        record_indexes=[
            pick(compound.record_indexes)[value] for value in relative_kept
        ],
        sample_annotations=annotations,
        # Seeds for children, including ones added by a later run.
        reaction_sequence_compounds=[
            (sequence, value.mapped_smiles)
            for sequence, value in builder.last_reaction_sequence_compounds.items()
        ],
        reaction_registry=_saved_reaction_registry(
            builder.last_reactions, builder.generator.fragmenter
        ),
    )
    output = Path(step.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_fragment_tree_structure(
        structure=structure, output_file=output, metadata=metadata
    )
    counts = [
        [len(peak.get("matches", ())) for peak in annotation.get("peaks", ())]
        for annotation in annotations
    ]
    return dict(file=str(output), kept=kept, counts=counts)
