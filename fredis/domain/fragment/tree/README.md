# Source-anchored FragmentTreeBuilder

`FragmentTreeBuilder` matches every pattern once on Original Source, inspects product templates once, and enumerates unordered, simultaneous primitive reaction combinations in Python. Every reaction must preserve every other reaction's concrete Source match. After mutual-retention, changed-bond conflict, redundancy, and duplicate checks, each unique final effect is compiled and applied once to Original Source. Intermediate fragments are never matched or reacted.

```python
builder = FragmentTreeBuilder(
    max_reaction_count=3,
    reaction_pattern_set=patterns,
)
reactions = builder.create_reaction_instances(source_compound)
seed = ReactionSequence((reactions[0],))
tree = builder.build(
    source_compound,
    seed_reaction_sequences=(seed,),
    max_reaction_count=3,
)
```

Without seeds (including an empty seed collection), exploration starts at Source, node 0. With seeds, Source remains node 0 and exploration starts only at the seed histories. Seed reaction counts are included in the normalized reaction limit. Seeds with multiple reactions attach their whole sequence in a transition with `is_seed=True` and `added_reaction=None`.

Nodes merge by canonical chemical SMILES. Expansion states retain each different Source reaction history even when the chemical node is shared. Equal final effects reuse a generated compound; equal SMILES alone do not identify equal effects because the retained Source atoms can differ.

If a primitive target fails chemical valence but its combined target is valid, the builder can connect that already computed target through a generated alternative predecessor. Canonical combination enumeration remains separate from this tree presentation; the effect is never executed twice.

`FragmentEdge.transitions` stores `ReactionSequenceTransition` values. Normal transitions represent adding one primitive reaction and normalizing the child history; they do not describe a reaction on the parent compound. `FragmentTree.num_transitions` counts them. Transition data is retained by in-memory edge access and copying; it is not serialized.

The retained set of a primitive reaction is the Source connected component containing all selected product anchors after removing the cuts of the entire reaction. Products whose anchors span components are unsupported. New atoms/bonds, element/charge updates, and aromatic bond order changes are omitted. Existing unspecified charges and unchanged aromatic bonds are preserved; single/double/triple bond order changes are supported.

Builder dictionaries require `max_reaction_count` and `reaction_pattern_set`. Every distinct reaction history that reaches a node is kept as a transition on its edge. Graph traversal depth is the number of edges, which can differ from reaction count for multi-reaction seeds.

`builder._build_result(source_compound)` returns `search_stats`, `expansion_states`, and `processed_expansion_states` alongside the tree and node compounds. Counters include primitive reactions, raw combinations, hard conflicts, invalidated reaction centers, normalization reductions, duplicate histories/effects, compiled sequences, RDKit executions, and generated effects.

`FragmentIonTreeBuilder` and `Fragmenter` forward `max_reaction_count` and `seed_reaction_sequences`. `Fragmenter.precursor_candidate_max_reaction_count` limits the normalized Source reaction count of precursor candidates, including every reaction in a seed. Candidate selection never uses graph depth. Peak annotation uses the reaction combinations that reach each node and checks precursor and total reaction budgets independently of edge distance. `tree_max_reaction_count` exposes the builder limit.

Canonical enumeration computes each unordered combination once, but the tree can retain multiple valid one-reaction predecessors of that already generated history. Batched peak annotation distinguishes precursor contexts. A Source reaction that overlaps the precursor's changed bonds is a hard conflict, so such a peak stays unassigned.

```bash
python -m unittest tests.domain.fragment.TestFragmentTreeBuilder tests.domain.fragment.TestReactionInstance tests.domain.fragment.TestReactionPattern tests.domain.fragment.TestReactionPatternSet tests.domain.fragment.TestFragmentTree
```
