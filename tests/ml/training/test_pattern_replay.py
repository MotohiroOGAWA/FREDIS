from types import SimpleNamespace

import pytest
import torch
from fredis.ml.training.fragment_tree_training.pattern_replay import RarePatternReplayGroups


def data(pattern):
    return SimpleNamespace(reactions=SimpleNamespace(reaction_type=torch.tensor([[pattern, 0, 0]])))


def test_inserts_unseen_pattern_after_100_batches_before_ordinary_pass_ends():
    groups = [(i,) for i in range(200)]
    lookup = lambda group: {1 if group[0] == 150 else 0}
    sampler = RarePatternReplayGroups(groups, 100, [0, 1, 2], lookup)
    seen = []
    for group in sampler:
        seen.append(group)
        sampler.observe(data(next(iter(lookup(group)))))
    assert seen[:100] == groups[:100]
    assert seen[100] == (150,)
    assert seen[101] == (100,)
    assert len(seen) == len(sampler) == 201
    assert sampler.replayed_patterns[1] == 1
    assert sampler.state['unavailable'] == {2}
    assert seen.count((150,)) == 2  # Forced batch does not replace normal data.


def test_normal_occurrences_reset_gap_without_forced_sampling():
    groups = [(i,) for i in range(200)]
    sampler = RarePatternReplayGroups(groups, 100, [0, 1])
    seen = []
    for group in sampler:
        seen.append(group)
        sampler.observe(data(1 if group[0] % 99 == 0 else 0))
    assert seen == groups
    assert sampler.replayed_batches == 0


def test_gap_continues_across_epochs_and_resume_state():
    state = {}
    first = RarePatternReplayGroups([(0,)], 1, [0, 1], state=state)
    for group in first:
        first.observe(data(0))
    # State is also what is persisted in checkpoints.
    second = RarePatternReplayGroups([(0,), (1,)], 1, [0, 1],
        pattern_ids_for_group=lambda group: {group[0]}, state=state)
    iterator = iter(second)
    assert next(iterator) == (1,)
    second.observe(data(1))
    assert next(iterator) == (0,)


def test_zero_disables_insertion_without_reading_reactions():
    sampler = RarePatternReplayGroups([(1,), (0,)], 0, [0, 1])
    seen = []
    for group in sampler:
        seen.append(group)
        sampler.observe(None)
    assert seen == [(1,), (0,)]


@pytest.mark.parametrize('gap', [-1, 0.5, float('nan'), True])
def test_invalid_gap(gap):
    with pytest.raises(ValueError, match='non-negative integer'):
        RarePatternReplayGroups([], gap)
