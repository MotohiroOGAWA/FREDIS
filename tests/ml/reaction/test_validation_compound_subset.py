"""Intermediate checks sample complete compounds, across files and chunks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from fredis.ml.training.fragment_tree_training.prepared_dataset import (
    PreparedPiece,
    PreparedPieceSequence,
    _LoaderState,
)
from fredis.ml.training.fragment_tree_training.training import (
    validation_subset,
    validation_compound_groups,
)


def indexed_pieces():
    # A occurs in different files and chunks; its sample count must not change
    # its chance of selection. Filtering has already removed missing row IDs.
    return PreparedPieceSequence(
        [
            PreparedPiece(Path("a1"), (0, 2), "A"),
            PreparedPiece(Path("b"), (0,), "B"),
            PreparedPiece(Path("a1"), (3, 4), "A"),
            PreparedPiece(Path("c"), (0, 1, 2), "C"),
            PreparedPiece(Path("a2"), (0,), "A"),
            PreparedPiece(Path("d"), (0,), "D"),
        ],
        state=_LoaderState(False),
        total_unfiltered_samples=12,
    )


def test_compound_fraction_keeps_all_retained_spectra_across_chunks_without_loading():
    pieces = indexed_pieces()
    groups = {"A": [0, 1, 3, 4, 8], "B": [2], "C": [5, 6, 7], "D": [9]}
    with patch.object(pieces._state, "load", side_effect=AssertionError("eager load")):
        assert validation_compound_groups(pieces) == groups
        for seed in range(10):
            selected, indices = validation_subset(pieces, 0.25, seed)
            chosen = [name for name, rows in groups.items() if set(rows) & set(indices)]
            assert len(chosen) == 1
            assert indices == groups[chosen[0]]
            assert selected.total_samples == len(indices)
            assert set(selected.compound_sample_groups()) == set(chosen)
            assert validation_subset(pieces, 0.25, seed)[1] == indices
        # Rounding is over the compound count, and fraction=1 retains every row.
        assert len(validation_subset(pieces, 0.26, 42)[0].compound_sample_groups()) == 2
        assert validation_subset(pieces, 1.0, 42)[1] == list(range(10))


def test_older_manifest_uses_actual_source_identity_not_filename():
    data = SimpleNamespace(
        source_smiles=("A", "B"),
        sample_tree_index=torch.tensor([0, 1, 0]),
    )
    pieces = PreparedPieceSequence(
        [
            PreparedPiece(Path("first"), (0, 1)),
            PreparedPiece(Path("second"), (2,)),
        ],
        state=_LoaderState(False),
        total_unfiltered_samples=3,
    )
    with patch.object(pieces._state, "load", return_value=data):
        assert pieces.compound_sample_groups() == {"A": [0, 2], "B": [1]}
        _, indices = validation_subset(pieces, 0.5, 42)
        assert indices in ([0, 2], [1])


def test_in_memory_selection_also_groups_across_pieces_and_sources():
    pieces = [
        SimpleNamespace(
            num_samples=3,
            source_smiles=("A", "B"),
            sample_tree_index=torch.tensor([0, 1, 0]),
        ),
        SimpleNamespace(
            num_samples=1, source_smiles=("A",), sample_tree_index=torch.tensor([0])
        ),
    ]
    assert validation_compound_groups(pieces) == {"A": [0, 2, 3], "B": [1]}
    with patch(
        "fredis.ml.input.reaction_batching.select_samples",
        side_effect=lambda piece, ids: ids,
    ):
        _, indices = validation_subset(pieces, 0.5, 42)
    assert indices in ([0, 2, 3], [1])


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_compound_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        validation_subset(indexed_pieces(), fraction, 42)
