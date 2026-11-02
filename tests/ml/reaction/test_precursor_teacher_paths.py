"""Alternative reaction orders must retain an actual precursor state."""
from types import SimpleNamespace

from fredis.ml.input.structure_builder import _precursor_teacher_paths


def transition(parent, added):
    return SimpleNamespace(parent_reaction_sequence=parent, added_reaction=added)


def annotation(sequence, precursor):
    return SimpleNamespace(reaction_sequence=sequence, precursor=SimpleNamespace(reaction_sequence=precursor))


def test_combination_inclusion_does_not_allow_precursor_bypass():
    # The same final combination can be reached through either reaction order.
    transitions = {
        'loss': [transition(None, 'loss')],
        'cleavage': [transition(None, 'cleavage')],
        'both': [transition('loss', 'cleavage'), transition('cleavage', 'loss')],
    }
    paths = _precursor_teacher_paths(transitions, [annotation('both', 'loss')])
    assert paths == [((None, None), ('loss', 'loss'), ('both', 'cleavage'))]
    assert _precursor_teacher_paths(transitions, [annotation('cleavage', 'loss')]) == []
    assert _precursor_teacher_paths(transitions, [annotation('loss', 'loss')]) == [((None, None), ('loss', 'loss'))]
    assert len(_precursor_teacher_paths(transitions, [annotation('both', None)])) == 2
