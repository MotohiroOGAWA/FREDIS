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

Reactant matches are encoded by an independent `ReactantGraphormerEncoder` for
each Reaction Pattern, shared across that pattern's rules and product choices.
Its fixed capacity is the number of atoms returned by `MolFromSmarts`; no
`start_cap`, `cap_growth` or attention-routing parameter is exposed. The
training encoder batches matches by pattern, uses the SMARTS bond topology,
and adds a learned SMARTS-position embedding to each source atom token. The
existing source atom, transformation-status and hop-context features remain
inputs. `reactant_num_layers` (default 2) and `reactant_dropout` (default 0)
in `reaction_model_params` control the pattern encoders. `reactant_node_dim`
and `reactant_graph_dim` optionally set their token and graph widths; both
default to `hidden_dim`. Attention heads follow `num_heads`.

The standalone reactant encoder accepts `reactant_smarts`, input `atom_dim` and
`bond_dim`, token `node_dim`, output `graph_dim`, `num_layers`, `num_heads` and
`dropout`. It returns node and graph embeddings. `graph_dim` may be any positive
integer: `ceil(graph_dim / node_dim)` independent learned graph tokens are
concatenated and projected by one Linear layer to exactly `graph_dim`.
Generic Graphormer calls use `graph_dim` instead of a caller-supplied graph
token count; their token width is named `hidden_dim`. Omitting `graph_dim`
retains masked mean pooling for node-only uses such as the spectrum tree.

This architecture changes checkpoint parameter names/shapes (pattern encoders
and graph readout projections). Checkpoints from the previous GNN/shared
SetTransformer architecture require retraining. Prepared training structures
remain usable when their pattern definitions match the model configuration.

Generic Graphormer accepts external context through `context_dim` and
`forward(data, context_repr=...)`. No context token count is supplied by the
caller: it is `ceil(context_dim / hidden_dim)`. Context is projected to the
resulting token width and kept separate from graph/node tokens. Each block
runs graph/node self-attention, reads context through a dedicated cross-attention
sublayer, and applies its FFN. Queries come from graph/node features; keys and
values come from context. Every valid graph/node token receives the residual
update; only padded positions are masked. Context is never updated from graph
contents. With one context token, attention weights are always one, so the
context update is shared, but the subsequent FFN and layers still act on each
node's distinct features. No cross-attention parameters are created when
context is disabled. Previous context-enabled checkpoints need retraining;
context-disabled molecular/reactant models keep their parameter layout.
