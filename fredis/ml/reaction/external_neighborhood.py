"""Pattern-owned inward propagation through external atoms, excluding SMARTS paths."""

from collections import deque
import torch
from torch import nn


class ExternalNeighborhoodEncoder(nn.Module):
    def __init__(self, atom_dim: int, hidden_dim: int, hops: int, *, bond_dim: int = 0):
        super().__init__()
        if type(hops) is not int or hops < 0:
            raise ValueError("neighborhood_hops must be a non-negative integer")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("neighborhood_hidden_dim must be a positive integer")
        if type(bond_dim) is not int or bond_dim < 0:
            raise ValueError("bond_dim must be a non-negative integer")
        self.hidden_dim, self.hops, self.bond_dim = hidden_dim, hops, bond_dim
        self.dropout = nn.Dropout(0.0)
        self.atom_projection = nn.Linear(atom_dim, hidden_dim, bias=False)
        self.messages = nn.ModuleList(
            nn.Linear(hidden_dim + bond_dim, hidden_dim, bias=False)
            for _ in range(hops)
        )

    def forward(self, atoms, edge_index, site_atoms, edge_attr=None):
        """Return [matches, SMARTS atoms, hidden_dim], with exact zeros for isolated roles.

        The discrete subgraph is assembled without RDKit. For each SMARTS role,
        other SMARTS atoms are blocked during traversal: an external substituent
        on one role cannot reach a second role through the SMARTS backbone.
        Messages move from distance k to k-1, including raw chemical information
        from every shell up to the configured radius.
        """
        if site_atoms.ndim != 2:
            raise ValueError("site_atoms must be [matches, SMARTS atoms]")
        shape = (*site_atoms.shape, self.hidden_dim)
        if not self.hops or not site_atoms.numel():
            return atoms.new_zeros(shape)
        if self.bond_dim and (
            edge_attr is None or edge_attr.shape != (edge_index.shape[1], self.bond_dim)
        ):
            raise ValueError("Bond-aware neighborhood requires source edge attributes")
        edge_by_pair = {}
        adjacency = [set() for _ in range(atoms.shape[0])]
        for i, (a, b) in enumerate(edge_index.detach().cpu().T.tolist()):
            edge_by_pair[(a, b)] = i
            adjacency[a].add(b)
            adjacency[b].add(a)
        atom_ids, depths, parents, children, roots, bonds = [], [], [], [], [], []
        for match in site_atoms.detach().cpu().tolist():
            internal = set(match)
            for root in match:
                distances = {root: 0}
                queue = deque([root])
                while queue:
                    current = queue.popleft()
                    if distances[current] == self.hops:
                        continue
                    for neighbor in sorted(adjacency[current] - internal):
                        if neighbor not in distances:
                            distances[neighbor] = distances[current] + 1
                            queue.append(neighbor)
                local = {atom: len(atom_ids) + i for i, atom in enumerate(distances)}
                roots.append(local[root])
                atom_ids.extend(distances)
                depths.extend(distances.values())
                for atom, depth in distances.items():
                    for neighbor in sorted(adjacency[atom]):
                        if distances.get(neighbor) == depth + 1:
                            parents.append(local[atom])
                            children.append(local[neighbor])
                            bonds.append(
                                edge_by_pair.get(
                                    (neighbor, atom), edge_by_pair.get((atom, neighbor))
                                )
                            )
        device = atoms.device
        ids = torch.tensor(atom_ids, device=device, dtype=torch.long)
        depth = torch.tensor(depths, device=device, dtype=torch.long)
        parent = torch.tensor(parents, device=device, dtype=torch.long)
        child = torch.tensor(children, device=device, dtype=torch.long)
        bond_ids = torch.tensor(bonds, device=device, dtype=torch.long)
        state = self.dropout(torch.nn.functional.gelu(self.atom_projection(atoms[ids])))
        # Root chemistry has its own input channel, so it must not leak into the
        # neighborhood channel, including through projection biases.
        state = state * (depth > 0).unsqueeze(-1)
        for distance in range(self.hops, 0, -1):
            select = depth[child] == distance
            targets, sources = parent[select], child[select]
            inputs = state[sources]
            if self.bond_dim:
                inputs = torch.cat((inputs, edge_attr[bond_ids[select]]), dim=-1)
            message = self.messages[distance - 1](inputs)
            pooled = state.new_zeros(state.shape).index_add(0, targets, message)
            counts = state.new_zeros(state.shape[0]).index_add(
                0, targets, state.new_ones(targets.numel())
            )
            state = state + self.dropout(torch.nn.functional.gelu(
                pooled / counts.clamp_min(1).unsqueeze(-1)
            ))
        return state[torch.tensor(roots, device=device)].reshape(shape)
