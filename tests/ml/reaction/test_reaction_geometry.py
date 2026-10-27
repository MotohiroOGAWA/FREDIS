"""Minimum substructure distances and bounded device-side graph propagation."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from fredis.ml.specgen.components.reaction.reaction_transformer import reaction_geometry


def csr(rows):
    counts = torch.tensor([len(row) for row in rows], dtype=torch.long)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))), torch.tensor(
        [value for row in rows for value in row], dtype=torch.long
    )


def fixture():
    # Large matched regions overlap although selected internal centers could be far apart.
    matches = [[10], [0, 1, 2, 3, 4], [4, 5, 6, 7], [14], [9], [1]]
    retained = [[10, 11], [0, 1, 2], [], [12, 13, 14], [7, 8, 9], [0, 1]]
    bonds = [(i, i + 1) for i in range(9)] + [(10, 11), (12, 13), (13, 14)]
    edges = torch.tensor(bonds + [(v, u) for u, v in bonds], dtype=torch.long).T
    ptr, atoms = csr(matches)
    rptr, ratoms = csr(retained)
    registry = SimpleNamespace(
        reaction_type=torch.zeros((6, 3), dtype=torch.long),
        reaction_source_compound_index=torch.tensor([1, 0, 0, 1, 0, 0]),
        reaction_source_atom_ptr=ptr,
        reaction_source_atom_index=atoms,
        reaction_retained_atom_ptr=rptr,
        reaction_retained_atom_index=ratoms,
        reaction_conflict_index=torch.tensor([[1, 2], [2, 1]]),
        reaction_invalidation_index=torch.tensor([[2], [4]]),
    )
    data = SimpleNamespace(
        reactions=registry,
        source_graph=SimpleNamespace(
            edge_index=edges, ptr=torch.tensor([0, 10, 15, 16]), num_nodes=16
        ),
    )
    return data, matches, retained, bonds


def reference(data, matches, retained, bonds, limit):
    owners = data.reactions.reaction_source_compound_index.tolist()
    rows = [
        [r for r, owner in enumerate(owners) if owner == source] for source in range(3)
    ]
    width = max(map(len, rows))
    ids = torch.full((3, width), -1, dtype=torch.long)
    distance = torch.full((3, width, width), limit + 1, dtype=torch.long)
    relations = torch.zeros((3, width, width, 4), dtype=torch.bool)
    neighbors = [[] for _ in range(16)]
    for u, v in bonds:
        neighbors[u].append(v)
        neighbors[v].append(u)
    conflict = set(map(tuple, data.reactions.reaction_conflict_index.T.tolist()))
    invalid = set(map(tuple, data.reactions.reaction_invalidation_index.T.tolist()))
    for source, members in enumerate(rows):
        ids[source, : len(members)] = torch.tensor(members)
        for i, r in enumerate(members):
            shortest = {u: 0 for u in matches[r]}
            queue = deque(matches[r])
            while queue:
                u = queue.popleft()
                for v in neighbors[u]:
                    if v not in shortest:
                        shortest[v] = shortest[u] + 1
                        queue.append(v)
            for j, t in enumerate(members):
                distance[source, i, j] = min(
                    limit + 1, min(shortest.get(v, limit + 1) for v in matches[t])
                )
                relations[source, i, j] = torch.tensor(
                    [
                        (r, t) in conflict,
                        (r, t) in invalid,
                        bool(set(matches[r]) & set(matches[t])),
                        bool(set(retained[r]) & set(matches[t])),
                    ]
                )
    return ids, distance, relations


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("limit", [1, 2, 4])
def test_geometry_matches_independent_bfs(device, limit):
    data, matches, retained, bonds = fixture()
    expected = reference(data, matches, retained, bonds, limit)
    # Inputs intentionally remain on CPU, as in incremental GPU preparation.
    with patch.object(
        torch.Tensor, "tolist", side_effect=AssertionError("CPU topology expansion")
    ):
        actual = reaction_geometry(data, limit, device=device)
    for result, target in zip(actual, expected):
        assert result.device.type == device
        torch.testing.assert_close(result.cpu(), target)
    # Overlapping large SMARTS matches have minimum distance zero.
    assert actual[1][0, 0, 1] == 0
    # The two regions of source 1 are disconnected.
    assert actual[1][1, 0, 1] == limit + 1


def test_geometry_handles_no_reactions():
    data, _, _, _ = fixture()
    for name, tensor in vars(data.reactions).items():
        if name.endswith("_ptr"):
            setattr(data.reactions, name, torch.zeros(1, dtype=torch.long))
        elif name.endswith("_index") and tensor.ndim == 2:
            setattr(data.reactions, name, torch.empty((2, 0), dtype=torch.long))
        else:
            setattr(data.reactions, name, tensor[:0])
    ids, distance, relations, endpoints = reaction_geometry(data)
    assert ids.shape == (3, 0)
    assert endpoints.shape == (2, 0)
    assert distance.shape == (3, 0, 0)
    assert relations.shape == (3, 0, 0, 4)


def test_frontier_expands_each_atom_reaction_pair_at_most_once():
    data, _, _, _ = fixture()
    frontiers = []
    original = torch.sparse.mm

    def record(matrix, active):
        if matrix.shape == (16, 16):
            frontiers.append(active.clone().bool())
        return original(matrix, active)

    with patch.object(torch.sparse, "mm", side_effect=record):
        reaction_geometry(data, max_distance=4)
    assert len(frontiers) == 4
    seen = torch.zeros_like(frontiers[0])
    for frontier in frontiers:
        assert not (seen & frontier).any()
        seen |= frontier


def test_large_overlapping_matches_do_not_shortcut_atom_paths():
    data, _, _, _ = fixture()
    _, distance, _, _ = reaction_geometry(data, max_distance=6)
    # [0..4] overlaps [4..7], but cannot teleport through it to atom 9.
    assert distance[0, 0, 1] == 0
    assert distance[0, 1, 2] == 2
    assert distance[0, 0, 2] == 5


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_geometry_is_safe_under_autocast(device):
    data, matches, retained, bonds = fixture()
    expected = reference(data, matches, retained, bonds, 4)
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device, dtype=dtype):
        actual = reaction_geometry(data, device=device)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), target)
