# Source-anchored fragment-tree model

Primitive reactions are encoded once from the source molecular graph.
Each branch group is identified by `(compound, normalized main adduct)` and
starts from the empty Source reaction state. `ReactionSequenceEncoder` represents the
selected reaction set without positional encoding. `BranchingReactionDecoder`
scores every chemically compatible next reaction from the complete primitive
reaction universe; collision energy is not an input to this scorer.

Reactions in a state are an unordered, simultaneous reaction set. There is no
"run A before B" interpretation. For candidate `B` and every selected reaction
`A`, compatibility requires both:

```text
B.source_atom_maps ⊆ A.retained_atom_maps
A.source_atom_maps ⊆ B.retained_atom_maps
A.changed_bond_maps ∩ B.changed_bond_maps = ∅
```

The tensor candidate mask is therefore:

```text
pool_valid(B)
AND B not already selected
AND mutual_source_retention(A, B) for every selected A
AND no_changed_bond_conflict(A, B) for every selected A
AND intersection(retained_atom_maps of the normalized set) is non-empty
AND normalized set differs from the parent set
AND normalized reaction count <= max_reaction_count
```

Preparation stores the directional invalidation relation `A -> B` when
`B.source_atom_maps` is not a subset of `A.retained_atom_maps`. The mask checks
all ordered pairs in the proposed set, so either `A -> B` or `B -> A` rejects
the composite reaction. A changed bond merely appearing in another reaction's
matched SMARTS context does not establish ordering; only two reactions changing
the same Source bond is a conflict.

Search accumulates `logsigmoid` transition scores, prunes on cumulative path
probability when enabled, deduplicates equal normalized states by maximum score,
and enforces `max_reaction_count` plus one shared `max_fragment_nodes` budget per
branch group. There is no learned absolute reaction filter, top-k pool, beam, or
EOS classifier.

After search, selected states cross the explicit chemistry boundary and are
materialized. The spectrum model encodes the resulting fragment tree. Prepared ion
explanations embed the ion adduct and hydrogen shift;
normalized attention combines explanations of the same physical ion. An
independent fragment logit and state logits produce physical probability in log
space (`logsigmoid(fragment) + logsigmoid(state)`). Intensity is predicted from
fragment, physical ion, main adduct, and collision-energy features and is not
used as either classification logit.

Training uses prepared graphs and packed transition/path/candidate tensors only.
It does not parse SMILES, match SMARTS, materialize fragments, calculate
formulas, generate ion candidates, or call RDKit.
