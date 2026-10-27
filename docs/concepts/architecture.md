# Neural architecture and probability flow

FREDIS has a shared molecular encoder, independent pattern-owned reaction atom
encoders, a CE-independent branch Transformer, and CE-conditioned spectrum heads.
The branch network selects reachable histories; the spectrum network distributes
probability over fragments and ion conditions.

## End-to-end view

```{mermaid}
flowchart TB
    SOURCE["Source and Compound graphs"] --> MOL["Shared MolEncoder<br/>Molecular Graphormer"]
    RAW["Full source atom/bond features<br/>Concrete reaction matches"] --> REACTION["Pattern-owned ReactionInstanceEncoder"]
    REACTION --> LOCAL["Local reaction features"]
    LOCAL --> MEMORY["Reaction memory Transformer<br/>ReactionRelationEncoding"]
    MEMORY --> DECODER["State Transformer Decoder<br/>BOS, main adduct, root and applied set"]
    MOL --> DECODER
    DECODER --> BRANCH["Branch query / reaction key scores<br/>Compatible-history search"]
    MOL --> GAT["Spectrum fragment GAT<br/>Root + main adduct + CE"]
    LOCAL --> GAT
    BRANCH --> TOPOLOGY["Reachable Compound / history topology"]
    TOPOLOGY --> GAT
    GAT --> FRAGMENT["Fragment Head<br/>Child / Self / Sink softmax"]
    LOCAL --> SET["Applied-set SetTransformer"]
    SET --> STATE["State Head<br/>Ion / hydrogen-shift softmax"]
    MOL --> STATE
    FRAGMENT --> FLOW["Log-space history propagation"]
    FLOW --> JOINT["Retained mass x set-conditioned state probability"]
    STATE --> JOINT
    JOINT --> FORMULA["Sum formula-equivalent contributions<br/>Annotate and normalize output peaks"]
```

Spectrum scoring uses local reaction features before the branch Transformer.
Its GAT and heads are independent of the branch memory/decoder outputs. Molecular
features and raw reaction encoders are shared trainable dependencies.

## Full chemical inputs

Let `F_atom = len(symbols) + 26`. With nine configured elements, each atom has
35 channels:

| Feature group | Channels |
|---|---:|
| Element symbol | `len(symbols)` |
| Formal charge | 3 |
| Ring type | 5 |
| Hybridization | 5 |
| Hydrogen count | 5 |
| Explicit valence | 8 |

The code's `valence_electrons` field contains `GetExplicitValence()` categories.
Bond features contain four bond-type and five ring-type channels. These are all
features currently defined by MolGraphBuilder; the builder does not define
additional stereo/conjugation channels.

## Molecular encoder

MolEncoder projects atom channels to `node_dim`, adds degree embeddings, and
runs molecular Graphormer layers with shortest-path and bond-path attention bias.
Learned graph tokens provide whole-molecule readout: the tokens are concatenated
and linearly projected to `graph_dim`.

`node_dim=16` and `graph_dim=64` use four graph tokens. `graph_dim` belongs to
molecular readout and can differ from the atom width. Degree clipping defaults
to 4; molecular spatial/edge limits default to 3.

## One reaction instance

For one pattern, let `R` be the actual match count across the batch, `P` its
SMARTS atom count, `N_max` the largest source molecule, and `H` its internal width.

```{mermaid}
flowchart TB
    Q["Reactant atom channels + SMARTS-position one-hot<br/>Linear: F_atom + P to H"]
    KV["All source atoms<br/>One-hop GNN with neighbor atoms and bonds"]
    CENTRALITY["Add CentralityEncoding to both sides"]
    BIAS["SpatialEncoding + EdgeEncoding<br/>Reactant atom to every source atom"]
    CROSS["Cross attention<br/>R x heads x P x N_max"]
    NORM["Residual + LayerNorm<br/>FFN + residual + LayerNorm"]
    READ["Flatten P tokens in SMARTS order<br/>Linear: P*H to shared reaction width"]
    OUTPUT["Add rule/product embeddings<br/>Dropout"]
    Q --> CENTRALITY
    KV --> CENTRALITY
    CENTRALITY --> CROSS
    BIAS --> CROSS
    CROSS --> NORM
    NORM -->|next layer| CROSS
    NORM --> READ --> OUTPUT
```

| Encoding | Contribution |
|---|---|
| Centrality | Source-molecule degrees added to Query and memory; `max_degree=4` by default |
| Spatial | Shortest reactant-atom-to-all-atom distances added to attention bias |
| Edge | Bond features along those shortest paths added to attention bias |

Each pattern owns its projections, one-hop GNN, structural embeddings, attention
layers, rule/product embeddings, and readout. Same-pattern matches from different
molecules are batched; each attends only to its own molecule with padding masked.
Duplicate queried source atoms reuse structural rows. The molecular key/value
memory is shared by a source's matches and remains fixed across attention layers.

The final reactant tokens are flattened directly in SMARTS order and passed
through Linear to the common reaction width. For `P=2`, `H=64`, and shared output
128, the readout is `Linear(128,128)`. The local encoder consumes raw chemistry
independently of MolEncoder outputs.

## Relations between complete reactions

ReactionMemoryEncoder batches each source's reactions, padding to the batch's
maximum reaction count. ReactionRelationEncoding uses minimum distances between
matched substructures and conflict, invalidation, shared-match, and shared-retained
atom relations. Its default cutoff is four bonds; farther/unreachable pairs share
a separate distance bucket.

Each pattern also projects SMARTS-position one-hot vectors to the common reaction
width. Every equally nearest endpoint is projected and summed, with directed
Query/Key and Value contributions. This preserves endpoint identity without a
separate role-width parameter.

The state decoder takes BOS, main-adduct, root-molecule, and applied-reaction-set
tokens, with self attention and cross attention to source reaction memory. It has
no absolute positions, so its BOS readout is invariant to applied-token order.
Branch queries combine that readout and main-adduct features; keys come from
reaction memory. Branch scores never use CE.

## Spectrum heads and probability

Sparse spectrum GAT nodes are unique `(spectrum, Compound)` pairs. Physical
reaction edges are deduplicated, and both message directions include direction
features. Independent GAT blocks apply attention and FFN with residual LayerNorm.
Main-adduct conditions select independent Fragment and State Head banks.

The Fragment Head scores Child, Self, and Sink events in one softmax per parent
history. Only Child mass propagates; Self mass is retained and Sink mass is lost.
All parents contribute before the next dependency level is processed.

The State Head consumes the current Compound, root molecule, CE, and an applied
reaction-set encoding. Histories sharing `(spectrum, Compound, applied set)` sum
retained mass and share one state distribution. Different sets are scored
separately:

```text
P(ion | fragment, set) = softmax(ion/shift logits)
P(fragment, ion) = sum_set P(retained fragment, set) * P(ion | fragment, set)
P(formula) = sum_explanations_with_same_formula P(explanation)
```

The implementation propagates and aggregates in log space for numerical stability.
Output filtering and maximum-intensity normalization occur after physical formula
aggregation. Formula annotations are chemical quantities, rather than neural
transition inputs.

## Normalization and efficient batching

Attention and FFN blocks use **post-norm**: attention, residual addition,
LayerNorm, then FFN, residual addition, LayerNorm. Layer/depth loops keep sequential
dependencies; atoms, edges, matches, and same-level histories use device tensors.

Static spectrum topology and condition lookups are cached without learned features
or autograd graphs. MolGraphormer packs PyG batches directly into size buckets.
Decoder batching bounds attention memory and reuses reaction-ID mappings.

Continue with [training objectives and detailed reaction settings](../guides/reaction-prediction-training.md),
[configuration](../reference/configuration.md), and [input tensor ownership](../reference/input-schema.md).
