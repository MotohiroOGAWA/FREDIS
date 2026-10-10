"""Lazy access to prepared fragment-tree tensors.

The preparation sidecars contain enough information to build stable sample
chunks without deserializing the tensor payloads.  Streaming therefore opens
each payload only when its batch is used.  The optional RAM mode changes only
the loader cache policy: files are still loaded on first use, never eagerly.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fredis.ml.input.reaction_batching import select_samples
from fredis.ml.input.source_reaction_structure import SourceReactionStructure


@dataclass(frozen=True)
class PreparedPiece:
    path: Path
    sample_indices: tuple[int, ...]

    @property
    def num_samples(self) -> int:
        return len(self.sample_indices)


@dataclass
class _LoaderState:
    cache_enabled: bool
    cache: dict[Path, SourceReactionStructure] = field(default_factory=dict)
    recent_path: Path | None = None
    recent_data: SourceReactionStructure | None = None

    def load(self, path: Path) -> SourceReactionStructure:
        if self.cache_enabled and path in self.cache:
            return self.cache[path]
        if self.recent_path == path and self.recent_data is not None:
            return self.recent_data
        data = SourceReactionStructure.load(path)
        if data.num_samples < 1:
            raise ValueError(f"Structure contains no samples: {path}")
        if data.spectrum_batch is None:
            raise ValueError(f"Full training requires stored target molecular graphs: {path}")
        self.recent_path, self.recent_data = path, data
        if self.cache_enabled:
            self.cache[path] = data
        return data


class PreparedPieceSequence(Sequence[SourceReactionStructure]):
    """A sequence of sample chunks backed by lazily loaded tensor files."""

    def __init__(self, pieces: Sequence[PreparedPiece], *, state: _LoaderState,
                 total_unfiltered_samples: int, statistics: dict | None = None):
        self._pieces = tuple(pieces)
        self._state = state
        self.total_unfiltered_samples = int(total_unfiltered_samples)
        self.statistics = dict(statistics or {})

    def __len__(self) -> int:
        return len(self._pieces)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        piece = self._pieces[index]
        data = self._state.load(piece.path)
        if piece.sample_indices and max(piece.sample_indices) >= data.num_samples:
            raise ValueError(
                f"Prepared index is stale for {piece.path}; regenerate the dataset so manifest.tsv and "
                "assignment_scores.tsv match the tensor files.")
        if len(piece.sample_indices) == data.num_samples and piece.sample_indices == tuple(range(data.num_samples)):
            return data
        return select_samples(data, piece.sample_indices)

    @property
    def total_samples(self) -> int:
        return sum(piece.num_samples for piece in self._pieces)

    @property
    def file_count(self) -> int:
        return len({piece.path for piece in self._pieces})

    @property
    def cache_enabled(self) -> bool:
        return self._state.cache_enabled

    def sample_count(self, index: int) -> int:
        return self._pieces[index].num_samples

    def ordered_indices(self, shuffle: bool) -> list[int]:
        """Shuffle source files while keeping a file's chunks adjacent.

        Adjacency lets streaming reuse its one-file transient cache and avoids
        deserializing a large source once per chunk.
        """
        blocks: list[list[int]] = []
        for index, piece in enumerate(self._pieces):
            if not blocks or self._pieces[blocks[-1][0]].path != piece.path:
                blocks.append([])
            blocks[-1].append(index)
        if shuffle:
            random.shuffle(blocks)
        return [index for block in blocks for index in block]

    def sample_subset(self, indices: Sequence[int]) -> "PreparedPieceSequence":
        selected = list(indices)
        pieces: list[PreparedPiece] = []
        cursor = offset = 0
        for piece in self._pieces:
            end = offset + piece.num_samples
            local = []
            while cursor < len(selected) and selected[cursor] < end:
                local.append(selected[cursor] - offset)
                cursor += 1
            if local:
                pieces.append(PreparedPiece(piece.path, tuple(piece.sample_indices[index] for index in local)))
            offset = end
        return PreparedPieceSequence(pieces, state=self._state,
            total_unfiltered_samples=self.total_unfiltered_samples, statistics=self.statistics)


def _score(value: str | None) -> float | None:
    return None if value is None or value.strip() == '' else float(value)


def _passes(row: dict[str, str], minimum: float, minimum_without_precursor: float) -> bool:
    full = _score(row.get('assignment_score'))
    without = _score(row.get('assignment_score_without_precursor'))
    return (full is None or full >= minimum) and (without is None or without >= minimum_without_precursor)


def prepared_piece_sequence(directory: str | Path, *, max_samples: int,
                            minimum_assignment_score: float,
                            minimum_assignment_score_without_precursor: float,
                            cache_in_memory: bool) -> PreparedPieceSequence:
    directory = Path(directory)
    manifest_path = directory / 'manifest.tsv'
    if not manifest_path.is_file():
        raise ValueError(
            f'{manifest_path} is required for streaming training. Regenerate this prepared dataset with the current '
            'create-fragment-tree-data command.')
    files = {path.name: path for path in directory.rglob('*.preft.pt')}
    if not files:
        raise ValueError(f'No prepared .preft.pt files in {directory}')
    counts: dict[str, int] = {}
    with manifest_path.open(newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            name = row.get('file', '')
            if name and row.get('status') == 'completed':
                counts[name] = int(row['num_valid_samples'])
    missing = sorted(set(files) - set(counts))
    if missing:
        raise ValueError(
            f'{manifest_path} does not describe {len(missing)} prepared tensor file(s), including {missing[0]}; '
            'regenerate the dataset before streaming training.')

    selected: dict[str, list[int]]
    filtering = minimum_assignment_score > 0 or minimum_assignment_score_without_precursor > 0
    if filtering:
        score_path = directory / 'assignment_scores.tsv'
        if not score_path.is_file():
            raise ValueError(
                f'{score_path} is required when assignment-score filtering is enabled; regenerate the dataset.')
        selected = defaultdict(list)
        score_counts: dict[str, int] = defaultdict(int)
        with score_path.open(newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                name = row['structure_file']
                score_counts[name] += 1
                if _passes(row, minimum_assignment_score, minimum_assignment_score_without_precursor):
                    selected[name].append(int(row['sample_index']))
        stale_scores = sorted(name for name, count in counts.items() if score_counts.get(name) != count)
        if stale_scores:
            raise ValueError(
                f'{score_path} does not match {len(stale_scores)} prepared tensor file(s), including '
                f'{stale_scores[0]}; regenerate the dataset before filtered streaming training.')
        for indices in selected.values():
            indices.sort()
    else:
        selected = {name: list(range(count)) for name, count in counts.items()}

    pieces = []
    for name, path in sorted(files.items(), key=lambda item: str(item[1])):
        indices = selected.get(name, ())
        for start in range(0, len(indices), max_samples):
            pieces.append(PreparedPiece(path, tuple(indices[start:start + max_samples])))
    if not pieces:
        raise ValueError(f'No samples in {directory} meet the configured assignment score thresholds')
    try:
        statistics = json.loads((directory / 'reaction_statistics.json').read_text())
    except (OSError, json.JSONDecodeError):
        statistics = {}
    return PreparedPieceSequence(pieces, state=_LoaderState(cache_in_memory),
        total_unfiltered_samples=sum(counts.values()), statistics=statistics)
