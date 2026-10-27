"""Rectangular geometry matches shortest-path references without all-atom matrices."""

from collections import deque

import pytest
import torch
from torch_geometric.data import Batch, Data

from fredis.ml.common.layers.graphormer import SpatialEncoding, EdgeEncoding
from fredis.ml.specgen.components.reaction.reactant_atom_attention import (
    rectangular_structural_bias, build_molecular_batch, ReactionAtomEncoder,
)


def reference_paths(nodes, bonds, features, query, target):
    adjacency = [[] for _ in range(nodes)]
    for index, (a, b) in enumerate(bonds):
        adjacency[a].append((b, index)); adjacency[b].append((a, index))
    paths = []
    frontier = deque([(query, [], {query})])
    best = nodes + 1
    while frontier:
        current, edges, visited = frontier.popleft()
        if len(edges) > best:
            continue
        if current == target:
            best = len(edges); paths.append(edges); continue
        for neighbor, edge in adjacency[current]:
            if neighbor not in visited:
                frontier.append((neighbor, edges + [edge], visited | {neighbor}))
    return paths


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_distances_and_all_tied_shortest_bond_paths(device, monkeypatch):
    torch.manual_seed(12)
    bonds = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    edges = torch.tensor(bonds + [(b, a) for a, b in bonds], device=device).T
    features = torch.randn(len(bonds), 9, device=device, requires_grad=True)
    attr = torch.cat((features, features))
    atoms = torch.randn(6, 4, device=device)
    molecular = build_molecular_batch(atoms, edges, attr)
    spatial = SpatialEncoding(2, 3).to(device)
    edge = EdgeEncoding(9, 2, 3).to(device)
    queries = torch.tensor([0, 3], device=device)
    monkeypatch.setattr(molecular, 'build_adj', lambda *a, **k: pytest.fail('All-atom adjacency is unnecessary'))
    monkeypatch.setattr(edge, 'forward', lambda *a, **k: pytest.fail('All-atom edge bias is unnecessary'))
    actual = rectangular_structural_bias(molecular, queries, spatial, edge, 3)
    expected = torch.zeros_like(actual)
    for row, query in enumerate(queries.tolist()):
        for target in range(6):
            paths = reference_paths(6, bonds, features, query, target)
            distance = len(paths[0]) if paths else 4
            value = spatial.dist_emb.weight[min(distance, 3)]
            if 0 < distance <= 3:
                values = [sum(features[index] @ edge.w_pos[position]
                              for position, index in enumerate(path)) / distance for path in paths]
                value = value + torch.stack(values).mean(0) * (2 if edge.undirected else 1)
            expected[row, :, target] = value
    torch.testing.assert_close(actual, expected)
    (actual * torch.randn_like(actual)).sum().backward()
    assert features.grad.abs().sum() > 0 and edge.w_pos.grad.abs().sum() > 0
    assert spatial.dist_emb.weight.grad.abs().sum() > 0


def test_duplicate_query_rows_are_reused_across_reactions_and_molecules(monkeypatch):
    graphs = [Data(x=torch.randn(n, 4), edge_index=torch.tensor([[0, 1], [1, 0]]),
                   edge_attr=torch.randn(2, 2)) for n in (4, 3)]
    source = Batch.from_data_list(graphs)
    molecular = build_molecular_batch(source.x, source.edge_index, source.edge_attr, source.ptr)
    model = ReactionAtomEncoder('CC', atom_dim=4, bond_dim=2, hidden_dim=8, output_dim=8,
                                num_heads=2, num_layers=1).eval()
    sites = torch.tensor([[0, 1], [1, 0], [4, 5], [0, 1]])
    from fredis.ml.specgen.components.reaction import reactant_atom_attention as module
    original = module.rectangular_structural_bias
    rows = []
    def capture(molecule, queries, *args):
        rows.append(queries.clone())
        return original(molecule, queries, *args)
    monkeypatch.setattr(module, 'rectangular_structural_bias', capture)
    together = model(molecular, sites)
    assert len(rows) == 1
    torch.testing.assert_close(rows[0], torch.tensor([0, 1, 4, 5]))
    assert together.shape == (4, 8)
    for index in range(4):
        torch.testing.assert_close(together[index:index+1], model(molecular, sites[index:index+1]))
