"""Prepared peak/depth branch supervision over shared execution-state features."""

from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor
from .tensor_table import TensorTable
from .compound_reaction_graph import ReactionExecutionGraph


@dataclass(frozen=True)
class BranchTeacher(TensorTable):
    graph: ReactionExecutionGraph
    state_depth: Tensor
    edge_label: Tensor
    peak_index: Tensor
    peak_branch_group_index: Tensor
    node_state_index: Tensor
    node_peak_index: Tensor
    target_node_index: Tensor
    node_depth: Tensor
    root_index: Tensor
    source_index: Tensor
    target_index: Tensor
    edge_index: Tensor
    bag_peak_index: Tensor
    bag_depth: Tensor
    node_index: Tensor
    node_bag_index: Tensor
    negative_edge_index: Tensor
    negative_bag_index: Tensor

    def validate(self, data):
        if self.edge_label.shape != self.graph.source_index.shape:
            raise ValueError(
                "Branch teacher labels must align with execution transitions"
            )
        if self.state_depth.shape != self.graph.compound_index.shape:
            raise ValueError("Branch teacher depths must align with execution states")
        sizes = {
            "peak_index": data.annotations.spectra.mz.numel(),
            "peak_branch_group_index": data.num_branch_groups,
            "node_state_index": self.state_depth.numel(),
            "node_peak_index": self.peak_index.numel(),
            "root_index": self.node_state_index.numel(),
            "target_node_index": self.node_state_index.numel(),
            "source_index": self.node_state_index.numel(),
            "target_index": self.node_state_index.numel(),
            "edge_index": self.edge_label.numel(),
            "bag_peak_index": self.peak_index.numel(),
            "node_index": self.node_state_index.numel(),
            "node_bag_index": self.bag_peak_index.numel(),
            "negative_edge_index": self.edge_label.numel(),
            "negative_bag_index": self.bag_peak_index.numel(),
        }
        for name, size in sizes.items():
            value = getattr(self, name)
            lower = -1 if name == "peak_index" else 0
            if (
                value.dtype != torch.long
                or value.ndim != 1
                or bool(((value < lower) | (value >= size)).any())
            ):
                raise ValueError(f"Invalid branch teacher {name}")
        if (
            self.peak_index.shape != self.peak_branch_group_index.shape
            or self.root_index.shape != self.peak_index.shape
        ):
            raise ValueError("Branch teacher requires one root and group per peak unit")
        if not (
            self.node_state_index.shape
            == self.node_peak_index.shape
            == self.node_depth.shape
        ):
            raise ValueError("Invalid branch teacher node columns")
        if not (
            self.source_index.shape == self.target_index.shape == self.edge_index.shape
        ):
            raise ValueError("Invalid branch teacher transition columns")
        if (
            self.bag_peak_index.shape != self.bag_depth.shape
            or self.node_index.shape != self.node_bag_index.shape
            or self.negative_edge_index.shape != self.negative_bag_index.shape
        ):
            raise ValueError("Invalid branch teacher bag columns")
        if bool((self.edge_label[self.edge_index] != 1).any()) or bool(
            (self.edge_label[self.negative_edge_index] != 0).any()
        ):
            raise ValueError("Branch teacher positive/negative memberships disagree")
        if not torch.equal(self.node_depth, self.state_depth[self.node_state_index]):
            raise ValueError("Branch teacher node depths disagree")
        if not torch.equal(
            self.node_state_index[self.source_index],
            self.graph.source_index[self.edge_index],
        ) or not torch.equal(
            self.node_state_index[self.target_index],
            self.graph.target_index[self.edge_index],
        ):
            raise ValueError("Branch teacher reaction history graph does not match shared transitions")
        if not torch.equal(
            self.node_depth[self.target_index], self.node_depth[self.source_index] + 1
        ):
            raise ValueError("Branch teacher transitions must advance one depth")
        if not torch.equal(
            self.node_depth[self.node_index], self.bag_depth[self.node_bag_index]
        ):
            raise ValueError("Branch teacher bag depths disagree")
        if not torch.equal(
            self.node_peak_index[self.node_index],
            self.bag_peak_index[self.node_bag_index],
        ):
            raise ValueError("Branch teacher bag peak membership disagrees")
        if not torch.equal(
            self.graph.branch_group_index[self.node_state_index],
            self.peak_branch_group_index[self.node_peak_index],
        ):
            raise ValueError("Branch teacher states belong to another branch group")
        if not torch.equal(
            self.state_depth[self.graph.source_index[self.negative_edge_index]] + 1,
            self.bag_depth[self.negative_bag_index],
        ):
            raise ValueError("Branch teacher negative bag depths disagree")
        if not torch.equal(
            self.graph.branch_group_index[
                self.graph.source_index[self.negative_edge_index]
            ],
            self.peak_branch_group_index[self.bag_peak_index[self.negative_bag_index]],
        ):
            raise ValueError("Branch teacher negative belongs to another group")
        for owner in (
            self.node_peak_index,
            self.bag_peak_index,
            self.node_peak_index[self.source_index],
            self.bag_peak_index[self.node_bag_index],
            self.bag_peak_index[self.negative_bag_index],
            self.node_peak_index[self.target_node_index],
        ):
            if bool((owner[1:] < owner[:-1]).any()):
                raise ValueError("Branch teacher rows must be packed by peak unit")
        actual = self.peak_index >= 0
        sample = torch.repeat_interleave(
            torch.arange(data.num_samples, device=data.device),
            data.annotations.spectra.peak_ptr.diff(),
        )
        if not torch.equal(
            data.sample_branch_group_index[sample[self.peak_index[actual]]],
            self.peak_branch_group_index[actual],
        ):
            raise ValueError("Branch teacher peak belongs to another branch group")
        return self

    @classmethod
    def from_peak_edges(
        cls, graph, state_depth, peak_edges, *, negative_candidates=None
    ):
        """Pack already assigned peak paths as reaction history graphs; never enumerate new paths."""
        device = state_depth.device

        def vector(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        src, dst = graph.source_index.tolist(), graph.target_index.tolist()
        depths, groups = state_depth.tolist(), graph.branch_group_index.tolist()
        positive = {edge for _, _, edges, _ in peak_edges for edge in edges}
        supported = (
            set(graph.root_index.tolist())
            | {src[e] for e in positive}
            | {dst[e] for e in positive}
        )
        negative = {
            e
            for e, parent in enumerate(src)
            if parent in supported and e not in positive
        }
        if negative_candidates is not None:
            negative.intersection_update(negative_candidates)
        outgoing = {}
        for edge in negative:
            outgoing.setdefault(src[edge], []).append(edge)
        # An unassigned group still supervises its root's negative alternatives.
        assigned = {group for _, group, _, _ in peak_edges}
        peak_edges = list(peak_edges) + [
            (-1, group, [], [])
            for group in range(len(graph.root_index))
            if group not in assigned and outgoing.get(int(graph.root_index[group]))
        ]
        columns = {
            f.name: []
            for f in fields(cls)
            if f.name not in ("graph", "state_depth", "edge_label")
        }
        for peak, group, edges, targets in peak_edges:
            edges = sorted(set(edges))
            states = sorted(
                {int(graph.root_index[group])}
                | {src[e] for e in edges}
                | {dst[e] for e in edges},
                key=lambda i: (depths[i], i),
            )
            unit = len(columns["peak_index"])
            mapping = {
                state: len(columns["node_state_index"]) + i
                for i, state in enumerate(states)
            }
            columns["peak_index"].append(peak)
            columns["peak_branch_group_index"].append(group)
            columns["target_node_index"].extend(
                mapping[i] for i in sorted(set(targets))
            )
            columns["node_state_index"].extend(states)
            columns["node_peak_index"].extend([unit] * len(states))
            columns["node_depth"].extend(depths[i] for i in states)
            columns["root_index"].append(mapping[int(graph.root_index[group])])
            columns["source_index"].extend(mapping[src[e]] for e in edges)
            columns["target_index"].extend(mapping[dst[e]] for e in edges)
            columns["edge_index"].extend(edges)
            positive_by_depth, parents_by_depth = {}, {}
            for edge in edges:
                d = depths[dst[edge]]
                positive_by_depth.setdefault(d, set()).add(mapping[dst[edge]])
                parents_by_depth.setdefault(d, set()).add(src[edge])
            # Terminal alternatives remain negative-only next-step bags, weighted
            # at the same peak/depth granularity as positive prediction bags.
            continuing = {src[e] for e in edges}
            for state in states:
                if state not in continuing and outgoing.get(state):
                    parents_by_depth.setdefault(depths[state] + 1, set()).add(state)
            for d in sorted(parents_by_depth):
                bag = len(columns["bag_peak_index"])
                nodes = sorted(positive_by_depth.get(d, ()))
                negatives = sorted(
                    {
                        e
                        for parent in parents_by_depth[d]
                        for e in outgoing.get(parent, ())
                    }
                )
                if not nodes and not negatives:
                    continue
                columns["bag_peak_index"].append(unit)
                columns["bag_depth"].append(d)
                columns["node_index"].extend(nodes)
                columns["node_bag_index"].extend([bag] * len(nodes))
                columns["negative_edge_index"].extend(negatives)
                columns["negative_bag_index"].extend([bag] * len(negatives))
        labels = torch.full((len(src),), -1, dtype=torch.int8, device=device)
        labels[vector(sorted(positive))] = 1
        labels[vector(sorted(negative))] = 0
        return cls(
            graph,
            state_depth,
            labels,
            **{key: vector(value) for key, value in columns.items()},
        )

    @classmethod
    def from_preparation(cls, data, raw):
        graph = data.execution_graph
        h, t = raw.states, raw.teacher
        active = raw.annotations.contexts.compound_index[h.candidate_index]
        compound_map = {smiles: i for i, smiles in enumerate(data.compounds.smiles)}
        state_lookup = {
            (g, c, s): i
            for i, (g, c, s) in enumerate(
                zip(
                    graph.branch_group_index.tolist(),
                    graph.compound_index.tolist(),
                    graph.reaction_set_index.tolist(),
                )
            )
        }
        state_ids = []
        for group, compound, set_id in zip(
            h.branch_group_index.tolist(),
            active.tolist(),
            h.candidate_reaction_set_index[h.candidate_index].tolist(),
        ):
            canonical = raw.annotations.compounds.smiles[compound]
            state_ids.append(state_lookup[(group, compound_map[canonical], set_id)])
        edge_lookup = {
            (a, r): i
            for i, (a, r) in enumerate(
                zip(graph.source_index.tolist(), graph.reaction_index.tolist())
            )
        }
        peaks = []
        for peak, group, first, last in zip(
            t.peak_index.tolist(),
            t.peak_branch_group_index.tolist(),
            t.peak_path_ptr[:-1].tolist(),
            t.peak_path_ptr[1:].tolist(),
        ):
            edges, targets = set(), set()
            for path in range(first, last):
                a, b = t.path_step_ptr[path : path + 2].tolist()
                for state, reaction in zip(
                    t.path_step_state_index[a:b].tolist(),
                    t.path_step_reaction_index[a:b].tolist(),
                ):
                    edge = edge_lookup[(state_ids[state], reaction)]
                    edges.add(edge)
                targets.add(int(graph.target_index[edge]))
            if edges:
                peaks.append((peak, group, edges, targets))
        return cls.from_peak_edges(graph, data.state_reaction_count, peaks)

    @classmethod
    def collate(cls, items, compound_maps, set_maps, reaction_offsets):
        graph_columns = {f.name: [] for f in fields(ReactionExecutionGraph)}
        columns = {f.name: [] for f in fields(cls) if f.name != "graph"}
        state_offset = edge_offset = group_offset = peak_offset = unit_offset = (
            node_offset
        ) = bag_offset = 0
        for data, cm, sm, ro in zip(items, compound_maps, set_maps, reaction_offsets):
            t = data.branch_teacher
            g = t.graph
            graph_columns["compound_index"].append(cm[g.compound_index])
            graph_columns["branch_group_index"].append(
                g.branch_group_index + group_offset
            )
            graph_columns["reaction_set_index"].append(sm[g.reaction_set_index])
            graph_columns["reaction_index"].append(g.reaction_index + ro)
            for key in ("source_index", "target_index", "root_index"):
                graph_columns[key].append(getattr(g, key) + state_offset)
            offsets = dict(
                node_state_index=state_offset,
                peak_branch_group_index=group_offset,
                node_peak_index=unit_offset,
                root_index=node_offset,
                target_node_index=node_offset,
                source_index=node_offset,
                target_index=node_offset,
                node_index=node_offset,
                edge_index=edge_offset,
                negative_edge_index=edge_offset,
                bag_peak_index=unit_offset,
                node_bag_index=bag_offset,
                negative_bag_index=bag_offset,
            )
            for key in columns:
                value = getattr(t, key)
                if key == "peak_index":
                    value = torch.where(value >= 0, value + peak_offset, value)
                else:
                    value = value + offsets.get(key, 0)
                columns[key].append(value)
            state_offset += t.state_depth.numel()
            edge_offset += t.edge_label.numel()
            group_offset += data.num_branch_groups
            peak_offset += data.annotations.spectra.mz.numel()
            unit_offset += t.peak_index.numel()
            node_offset += t.node_state_index.numel()
            bag_offset += t.bag_peak_index.numel()
        graph = ReactionExecutionGraph(
            **{k: torch.cat(v) for k, v in graph_columns.items()}
        )
        return cls(graph, **{k: torch.cat(v) for k, v in columns.items()})

    def select(self, peak_indices, selected_groups, group_map):
        """Remap saved memberships, including repeated/reordered samples."""
        g = self.graph
        device = self.state_depth.device
        keep = group_map[g.branch_group_index] >= 0
        state_ids = keep.nonzero().flatten()
        state_map = torch.full_like(g.compound_index, -1)
        state_map[state_ids] = torch.arange(len(state_ids), device=device)
        edge_ids = keep[g.source_index].nonzero().flatten()
        edge_map = torch.full_like(g.source_index, -1)
        edge_map[edge_ids] = torch.arange(len(edge_ids), device=device)
        graph = ReactionExecutionGraph(
            g.compound_index[state_ids],
            group_map[g.branch_group_index[state_ids]],
            state_map[g.source_index[edge_ids]],
            state_map[g.target_index[edge_ids]],
            g.reaction_index[edge_ids],
            state_map[g.root_index[selected_groups]],
            g.reaction_set_index[state_ids],
        )
        peak_units = {}
        for unit, peak in enumerate(self.peak_index.tolist()):
            if peak >= 0:
                peak_units.setdefault(peak, []).append(unit)
        selection = [
            (unit, new)
            for new, old in enumerate(peak_indices.tolist())
            for unit in peak_units.get(old, ())
        ]
        selection.extend(
            (unit, -1)
            for unit in (self.peak_index < 0).nonzero().flatten().tolist()
            if int(group_map[self.peak_branch_group_index[unit]]) >= 0
        )
        columns = {
            f.name: []
            for f in fields(self)
            if f.name not in ("graph", "state_depth", "edge_label")
        }

        # Rows are packed by peak unit. Build boundaries once, then copy only
        # selected slices instead of rescanning every table for every peak.
        def boundaries(owner):
            counts = torch.bincount(owner, minlength=self.peak_index.numel())
            return torch.cat((counts.new_zeros(1), counts.cumsum(0))).tolist()

        pointers = [
            boundaries(owner)
            for owner in (
                self.node_peak_index,
                self.node_peak_index[self.source_index],
                self.bag_peak_index,
                self.bag_peak_index[self.node_bag_index],
                self.bag_peak_index[self.negative_bag_index],
                self.node_peak_index[self.target_node_index],
            )
        ]
        node_offset = bag_offset = 0
        for new_unit, (unit, peak) in enumerate(selection):
            nodes, transitions, bags, positives, negatives, targets = [
                slice(ptr[unit], ptr[unit + 1]) for ptr in pointers
            ]
            node_shift = node_offset - nodes.start
            bag_shift = bag_offset - bags.start
            block = dict(
                peak_index=self.peak_index.new_tensor([peak]),
                peak_branch_group_index=group_map[
                    self.peak_branch_group_index[unit : unit + 1]
                ],
                node_state_index=state_map[self.node_state_index[nodes]],
                node_peak_index=torch.full_like(self.node_peak_index[nodes], new_unit),
                node_depth=self.node_depth[nodes],
                target_node_index=self.target_node_index[targets] + node_shift,
                root_index=self.root_index[unit : unit + 1] + node_shift,
                source_index=self.source_index[transitions] + node_shift,
                target_index=self.target_index[transitions] + node_shift,
                edge_index=edge_map[self.edge_index[transitions]],
                bag_peak_index=torch.full_like(self.bag_peak_index[bags], new_unit),
                bag_depth=self.bag_depth[bags],
                node_index=self.node_index[positives] + node_shift,
                node_bag_index=self.node_bag_index[positives] + bag_shift,
                negative_edge_index=edge_map[self.negative_edge_index[negatives]],
                negative_bag_index=self.negative_bag_index[negatives] + bag_shift,
            )
            for key, value in block.items():
                columns[key].append(value)
            node_offset += nodes.stop - nodes.start
            bag_offset += bags.stop - bags.start
        empty = self.peak_index.new_empty(0)
        return type(self)(
            graph,
            self.state_depth[state_ids],
            self.edge_label[edge_ids],
            **{k: torch.cat(v) if v else empty for k, v in columns.items()},
        )
