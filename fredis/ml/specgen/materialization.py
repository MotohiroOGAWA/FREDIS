"""Explicit post-decoding RDKit boundary; never called by training forward."""

from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Sequence
from rdkit import Chem
import torch
from torch import Tensor
from torch_geometric.data import Batch
from fredis.libs.mmkit.mmkit import Compound
from fredis.domain.fragment.reaction import ReactionInstance, ReactionSequence
from fredis.domain.fragment.tree.FragmentTree import FragmentTree
from fredis.domain.fragment.tree.FragmentNode import FragmentNode
from fredis.domain.fragment.tree.FragmentEdge import FragmentEdge
from fredis.domain.fragment.tree.ReactionSequenceTransition import (
    ReactionSequenceTransition,
)
from ..mol.graph_builder import MolGraphBuilder
from .components.reaction.reaction_decoder import ReactionDecoderOutput


@dataclass(frozen=True)
class DecodedFragmentTreeBatch:
    node_graph: Batch
    edge_index: Tensor
    node_sample_index: Tensor
    edge_added_reaction_index: Tensor
    terminal_node_index: Tensor
    node_log_score: Tensor
    trees: tuple[FragmentTree, ...]
    compounds: tuple[Compound, ...]
    rdkit_run_count: int
    failed_effect_count: int
    reaction_state_graph: bool = field(default=False, kw_only=True)
    materialized_state_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )
    materialized_node_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long), kw_only=True
    )

    def to(self, device: str | torch.device) -> DecodedFragmentTreeBatch:
        from dataclasses import replace

        return replace(
            self,
            node_graph=self.node_graph.to(device),
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "edge_index",
                    "node_sample_index",
                    "edge_added_reaction_index",
                    "terminal_node_index",
                    "node_log_score",
                    "materialized_state_index",
                    "materialized_node_index",
                )
            },
        )


def materialize_reaction_states(
    output: ReactionDecoderOutput,
    sources: Sequence[Compound],
    reactions_by_tree: Sequence[tuple[ReactionInstance, ...]],
    sample_tree_index: Tensor,
    graph_builder: MolGraphBuilder,
    *,
    effect_cache: dict | None = None,
) -> DecodedFragmentTreeBatch:
    # CPU transfer and domain object reconstruction happen only after decoding.
    sample = output.state_sample_index.detach().cpu().tolist()
    state = output.state_reaction_index.detach().cpu().tolist()
    parent = output.parent_state_index.detach().cpu().tolist()
    score = output.state_log_score.detach().cpu().tolist()
    terminal = output.terminal.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    if output.transition_parent_index is None:
        raise ValueError("Decoded states must retain all reaction transitions")
    pool = output.pool.reaction_index.detach().cpu().tolist()
    tree_by_sample = sample_tree_index.detach().cpu().tolist()
    closure = set(terminal)
    pending = list(terminal)
    while pending:
        ancestor = parent[pending.pop()]
        if ancestor >= 0 and ancestor not in closure:
            closure.add(ancestor)
            pending.append(ancestor)
    reaction_offsets = [
        sum(len(row) for row in reactions_by_tree[:tree])
        for tree in range(len(reactions_by_tree))
    ]
    effects: dict[tuple[int, object], Compound | None] = (
        effect_cache if effect_cache is not None else {}
    )
    nodes_by_sample = [
        [FragmentNode(0, sources[tree].smiles)] for tree in tree_by_sample
    ]
    compounds_by_sample = [[sources[tree]] for tree in tree_by_sample]
    edges_by_sample: list[dict[tuple[int, int], FragmentEdge]] = [
        {} for _ in tree_by_sample
    ]
    state_node, sequences = {}, {}
    index_by_smiles = [{sources[tree].smiles: 0} for tree in tree_by_sample]
    runs = failures = 0
    terminal_set = set(terminal)
    terminal_nodes: list[tuple[int, int]] = []
    scores_by_sample = [[-float("inf")] for _ in tree_by_sample]
    for row in sorted(closure):
        s, tree = sample[row], tree_by_sample[sample[row]]
        ids = [pool[s][i] - reaction_offsets[tree] for i in state[row] if i >= 0]
        sequence = (
            ReactionSequence(reactions_by_tree[tree][i] for i in ids) if ids else None
        )
        sequences[row] = sequence
        if sequence is None:
            node = 0
        else:
            effect = (tree, sequence.effect_key)
            if effect not in effects:
                try:
                    compiled = sequence.compile(sources[tree])
                    runs += 1
                    products = compiled.run(sources[tree])
                    if len(products) != 1:
                        raise ValueError(
                            "Source reaction sequence must materialize one molecule"
                        )
                    effects[effect] = Compound(products[0])
                except Chem.rdchem.MolSanitizeException:
                    failures += 1
                    effects[effect] = None
            compound = effects[effect]
            if compound is None:
                continue
            node = index_by_smiles[s].get(compound.smiles)
            if node is None:
                node = len(nodes_by_sample[s])
                nodes_by_sample[s].append(FragmentNode(node, compound.smiles))
                compounds_by_sample[s].append(compound)
                index_by_smiles[s][compound.smiles] = node
                scores_by_sample[s].append(-float("inf"))
        state_node[row] = node
        scores_by_sample[s][node] = max(scores_by_sample[s][node], score[row])
        if row in terminal_set:
            terminal_nodes.append((s, node))
    for p, c, r in zip(
        output.transition_parent_index.tolist(),
        output.transition_child_index.tolist(),
        output.transition_reaction_index.tolist(),
    ):
        if p not in state_node or c not in state_node:
            continue
        s = sample[c]
        src, dst = state_node[p], state_node[c]
        if src == dst:
            continue
        tree = tree_by_sample[s]
        reaction = reactions_by_tree[tree][pool[s][r] - reaction_offsets[tree]]
        transition = ReactionSequenceTransition(
            reaction, sequences[c], sequences[p], is_seed=False
        )
        key = (src, dst)
        edges = edges_by_sample[s]
        if key not in edges:
            edges[key] = FragmentEdge(len(edges), src, dst)
        edges[key] = edges[key].with_transition(transition)
    trees = []
    all_compounds = []
    sample_rows, edge_rows, edge_reactions, node_scores = [], [], [], []
    offsets = []
    for s, nodes in enumerate(nodes_by_sample):
        offset = len(all_compounds)
        offsets.append(offset)
        edges = tuple(edges_by_sample[s].values())
        trees.append(
            FragmentTree.from_nodes_and_edges(
                smiles=nodes[0].smiles, nodes=tuple(nodes), edges=edges
            )
        )
        all_compounds.extend(compounds_by_sample[s])
        sample_rows.extend([s] * len(nodes))
        node_scores.extend(scores_by_sample[s])
        for edge in edges:
            for transition in edge.transitions:
                edge_rows.append(
                    (edge.source_index + offset, edge.target_index + offset)
                )
                reaction = transition.added_reaction
                edge_reactions.append(
                    -1
                    if reaction is None
                    else reaction_offsets[tree_by_sample[s]]
                    + reactions_by_tree[tree_by_sample[s]].index(reaction)
                )
    graph = Batch.from_data_list(
        [graph_builder.build(compound) for compound in all_compounds]
    )
    return DecodedFragmentTreeBatch(
        graph,
        torch.tensor(edge_rows, dtype=torch.long).reshape(-1, 2).T,
        torch.tensor(sample_rows, dtype=torch.long),
        torch.tensor(edge_reactions, dtype=torch.long),
        torch.tensor(
            sorted({offsets[s] + node for s, node in terminal_nodes}), dtype=torch.long
        ),
        torch.tensor(node_scores, dtype=torch.float32),
        tuple(trees),
        tuple(all_compounds),
        runs,
        failures,
        reaction_state_graph=True,
        materialized_state_index=torch.tensor(sorted(state_node), dtype=torch.long),
        materialized_node_index=torch.tensor(
            [offsets[sample[row]] + state_node[row] for row in sorted(state_node)],
            dtype=torch.long,
        ),
    )


def compound_graph_display(data):
    """Domain tree views for display only; model indices remain CompoundBatch IDs."""
    from ..input.source_reaction_structure import rows

    graph = data.compound_graph
    execution = data.execution_graph
    trees = []
    for group, edges in enumerate(
        rows(data.reaction_contexts.edge_ptr, data.reaction_contexts.edge_index)
    ):
        root = int(
            data.source_compounds.compound_index[
                data.reaction_contexts.root_source_compound_index[group]
            ]
        )
        ids = [root] + sorted(
            set(
                execution.compound_index[execution.branch_group_index == group].tolist()
            )
            - {root}
        )
        local = {cid: i for i, cid in enumerate(ids)}
        nodes = tuple(
            FragmentNode(i, data.compounds.smiles[cid]) for i, cid in enumerate(ids)
        )
        domain_edges = []
        for edge in edges:
            src, dst = int(graph.source_compound_index[edge]), int(
                graph.target_compound_index[edge]
            )
            if src in local and dst in local and src != dst:
                domain_edges.append(
                    FragmentEdge(len(domain_edges), local[src], local[dst])
                )
        trees.append(
            FragmentTree.from_nodes_and_edges(
                smiles=data.compounds.smiles[root],
                nodes=nodes,
                edges=tuple(domain_edges),
            )
        )
    return DecodedFragmentTreeBatch(
        data.compounds.graph,
        torch.stack((graph.source_compound_index, graph.target_compound_index)),
        torch.full(
            (len(data.compounds.compounds),), -1, dtype=torch.long, device=data.device
        ),
        graph.reaction_index,
        data.source_compounds.compound_index,
        torch.zeros(len(data.compounds.compounds), device=data.device),
        tuple(trees),
        data.compounds.compounds,
        0,
        0,
        reaction_state_graph=True,
    )
