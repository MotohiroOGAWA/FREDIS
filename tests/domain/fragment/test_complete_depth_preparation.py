"""A completed pure search precedes chemistry; overflowing depths are atomic."""

from collections import Counter
from itertools import combinations
from unittest.mock import patch

import pytest

from fredis.domain.fragment.reaction import (
    ReactionSequence,
    ReactionSequenceSearch,
    FragmentTreeLimitExceeded,
)
from fredis.libs.mmkit.mmkit import Compound, Adduct
from tests.domain.fragment.TestFragmentTreeBuilder import builder, pattern, reaction
from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from tests.ml.reaction.TestReactionPipeline import chemistry


def independent_reactions():
    return tuple(
        reaction(i, edge, matched=(edge,), cuts=(edge,))
        for i, edge in enumerate(combinations((1, 2, 3, 4), 2))
    )


@pytest.mark.parametrize("failed_depth", range(1, 7))
def test_budget_mid_depth_retains_only_complete_lower_depths(failed_depth):
    reactions = independent_reactions()
    lower = failed_depth - 1
    expected = ReactionSequenceSearch(reactions, max_reaction_count=lower)
    expected_rows = expected.enumerate()
    upper = ReactionSequenceSearch(reactions, max_reaction_count=failed_depth)
    upper.enumerate()
    limit = max(
        1,
        expected.stats.num_raw_combinations
        + (upper.stats.num_raw_combinations - expected.stats.num_raw_combinations) // 2,
    )
    actual = ReactionSequenceSearch(
        reactions, max_reaction_count=6, max_reaction_combinations=limit
    )
    with patch.object(
        ReactionSequence,
        "compile",
        side_effect=AssertionError("Chemistry during enumeration"),
    ):
        retained = actual.enumerate_completed_depths()
    assert {c.reaction_sequence for c in retained} == {
        c.reaction_sequence for c in expected_rows
    }
    assert actual.stats.num_raw_combinations == expected.stats.num_raw_combinations
    assert actual.completion_stats["completed_max_reaction_count"] == lower
    assert actual.completion_stats["discarded_from_reaction_count"] == failed_depth
    attempt = actual.completion_stats["limit_attempts"][0]
    assert attempt["stats"]["num_raw_combinations"] == limit + 1
    assert attempt["stats"]["num_rdkit_run_reactants"] == 0


def test_exact_budget_and_unlimited_search_are_identical():
    reactions = independent_reactions()
    baseline = ReactionSequenceSearch(reactions, max_reaction_count=6)
    expected = baseline.enumerate_completed_depths()
    exact = ReactionSequenceSearch(
        reactions,
        max_reaction_count=6,
        max_reaction_combinations=baseline.stats.num_raw_combinations,
    )
    assert exact.enumerate_completed_depths() == expected
    assert exact.stats == baseline.stats
    assert exact.completion_stats == baseline.completion_stats


@pytest.mark.parametrize("failed_depth", range(1, 5))
def test_tree_never_compiles_discarded_depth_or_retries_retained_effects(failed_depth):
    build = builder(
        *(
            pattern(f"[C:1]-[{element}:2]", "[C:1]")
            for element in ("O", "F", "Cl", "Br")
        ),
        limit=4,
    )
    source = Compound.from_smiles("C(O)(F)(Cl)Br")
    reference = build._build_result(source, max_reaction_count=failed_depth - 1)
    # Allow one attempt in the next depth; a depth with one member overflows immediately.
    limit = max(1, reference["search_stats"]["num_raw_combinations"])
    finished = []
    compiled = []
    enumerate_original = ReactionSequenceSearch.iter_candidates
    compile_original = ReactionSequence.compile

    def enumerate_tracked(search):
        try:
            yield from enumerate_original(search)
        finally:
            finished.append(True)

    def compile_tracked(sequence, *args, **kwargs):
        assert finished, "Compilation began before search finished"
        assert len(sequence.reactions) < failed_depth
        compiled.append(sequence.effect_key)
        return compile_original(sequence, *args, **kwargs)

    with (
        patch.object(ReactionSequenceSearch, "iter_candidates", enumerate_tracked),
        patch.object(ReactionSequence, "compile", compile_tracked),
    ):
        result = build._build_result(source, max_reaction_combinations=limit)
    assert len(finished) == 1
    assert all(n == 1 for n in Counter(compiled).values())
    actual = result["fragment_tree"]
    expected = reference["fragment_tree"]
    signature = lambda t: (
        t.node_smiles.tolist(),
        t.source_indices.tolist(),
        t.target_indices.tolist(),
        t.num_transitions,
    )
    assert signature(actual) == signature(expected)
    assert result["search_stats"]["completed_max_reaction_count"] == failed_depth - 1


def test_preparation_calls_tree_builder_once_on_combination_overflow():
    context = create_preparation_context(chemistry())
    source = Compound.from_smiles("CCCCCO")
    reference = context.fragmenter.build_fragment_ion_tree(source, max_reaction_count=2)
    prepare = ReactionStructureBuilder(
        context,
        max_reaction_combinations=reference.search_stats["num_raw_combinations"],
    )
    with patch.object(
        type(context.fragmenter),
        "build_fragment_ion_tree",
        autospec=True,
        side_effect=type(context.fragmenter).build_fragment_ion_tree,
    ) as calls:
        data, _ = prepare.build(
            source, [Adduct.parse("[M+H]+")], [20.0], [[103.0759]], [[1.0]]
        )
    assert calls.call_count == 1
    assert prepare.last_search_stats["completed_max_reaction_count"] == 2
    data.validate()


def test_incremental_seed_beyond_completed_depth_is_not_silently_dropped():
    reactions = independent_reactions()
    search = ReactionSequenceSearch(
        reactions,
        max_reaction_count=6,
        max_reaction_combinations=1,
        seed_reaction_sequences=(ReactionSequence(reactions[:3]),),
        include_source=True,
    )
    with pytest.raises(FragmentTreeLimitExceeded):
        search.enumerate_completed_depths()
    assert search.stats.num_rdkit_run_reactants == 0
