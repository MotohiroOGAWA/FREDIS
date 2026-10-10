# Reaction instances and sequences

A `ReactionInstance` binds one `ReactionPattern` rule to one concrete Source match. A `ReactionSequence` combines compatible instances into one molecule-specific reaction.

```python
instance = ReactionInstance.from_match(
    source=source,
    reaction_pattern=pattern,
    reaction_pattern_id=0,
    reaction_rule_id=0,
    product_molecule_id=0,  # Zero-based ProductMolecule ID.
    source_atom_maps=(2, 3),  # Reactant query atom order.
    retained_atom_maps=frozenset({1, 2}),
    cut_bond_maps=(),
)
sequence = ReactionSequence.from_reactions((instance,))
compiled = sequence.compile(source)
products = compiled.run(source)
```

`source` accepts a Compound or an RDKit Mol whose atoms have unique positive atom maps. `from_match` validates the selected match and records only bonds explicitly present in the outer reactant SMARTS query graph. Bonds inside recursive SMARTS are outside this scope.

Supply retained atoms manually. `product_molecule_id` is required and selects a zero-based product template within the selected reaction rule. `source_atom_maps` follows RDKit reactant query atom order, regardless of template atom map numbers.

`from_match` extracts cuts and single/double/triple bond order changes from the selected product template. Additional cuts may be supplied manually. `bond_updates` records `(source_map_u, source_map_v, target_bond_type)` triples. `changed_bond_maps` includes cuts, order changes, and Source bonds removed through atom deletion, including bonds outside the matched query. Direct dataclass callers must register those additional deleted Source bonds themselves. Charge updates and atom/bond creation are outside this scope.

Sequences reject overlapping `changed_bond_maps` after duplicate removal and before redundancy removal. Shared unchanged context bonds are allowed. Redundancy removal preserves surviving cuts and bond order updates. Compilation applies the combined edits and recalculates hydrogen counts at affected atoms.

`CompositeReaction.run` accepts the original mapped Source, calls RunReactants once, and restores Source atom maps. RDKit atom maps alone do not constrain match locations, so the compiled `rxn` includes additional queries matching Source atom map properties. Reconstructing a reaction from the `smirks` string loses these additional queries. Use `run` to execute the reaction at its concrete locations.

`create_reaction_instances` (used by `FragmentTreeBuilder`) generates every primitive instance in bulk on Original Source. Unlike `from_match`, it computes retention from the connected Source component after the entire rule's cuts and performs no per-instance SMARTS validation or reaction execution. See [the tree workflow](../tree/README.md) for seeds, effect caching, transition storage and performance counters.

```bash
python -m unittest tests.domain.fragment.TestReactionInstance tests.domain.fragment.TestReactionPattern tests.domain.fragment.TestReactionPatternSet
```
