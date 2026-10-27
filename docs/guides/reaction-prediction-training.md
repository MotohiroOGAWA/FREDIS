# Branch prediction and spectrum training

## 1. Prepare experimental supervision

Group records by source molecule and normalize main adducts. Enumerate reactions,
materialize valid structures, build ion/formula candidates and assign measured
peaks. Save [BranchTeacher](../reference/input-schema.md#observations-and-learning-targets) units
for each observed peak and depth, plus the existing spectrum intensity targets.
See [data preparation](data-preparation.md) for input columns and search limits.

## 2. Encode molecules and local reaction features

Each reaction pattern owns a `ReactionAtomEncoder` built from raw atom and bond
features. MolEncoder supplies independent whole-molecule features for branch-state
and spectrum prediction.

The per-pattern reaction path is:

1. Concatenate each matched atom's complete MolGraphBuilder feature vector with
   its SMARTS-position one-hot label, then apply a Linear projection to
   `hidden_dim`. All channels are preserved: element, formal charge, ring type,
   hybridization, hydrogen count, and explicit valence. No symbol-only slicing
   is performed.
2. Build molecular memory with exactly one simultaneous GNN message-passing step.
   The memory combines each atom's complete raw chemical features with mean
   incoming messages from raw neighbor atoms and bond features, followed by
   GELU and LayerNorm. This GNN has independent weights for each reaction pattern.
3. Add full-molecule degree embeddings to both reactant queries and all-atom
   memory. Compute pattern-owned spatial and multi-hop edge-path encodings
   directly for unique reactant-to-all-atom rows. Sparse bond propagation avoids
   constructing full all-atom bias matrices, and identical query atoms reuse
   their rows across reaction matches. Existing encoding parameters are reused
   without extending the common Graphormer implementation.
4. Repeat `num_layers` independent cross-attention blocks. Every block applies
   attention, residual addition and LayerNorm, then an FFN, residual addition and
   LayerNorm. Only reactant tokens are updated; the one-hop molecular memory is
   fixed across layers. Keys and values are projected per molecule and layer,
   then shared across all matches. Padded atoms and other molecules are masked.
   Real atoms remain attendable regardless of structural distance limits.
   Every actual reaction match in the batch is processed together, with no fixed
   match limit or chunk setting. Reaction-memory padding uses the maximum
   reaction count in the current batch.
5. Flatten final reactant tokens in SMARTS-position order and apply one Linear
   readout directly to the shared `reaction_model_params.hidden_dim`. Add
   pattern-owned rule/product embeddings and apply dropout to produce the
   reaction feature. No separately configured intermediate readout width is used.

The attention and both structural biases have shape
`[reaction_matches, heads, reactant_atoms, source_atoms_padded]`. Matches from
different compounds are processed together within each pattern; every match
attends all real atoms of its own compound only. Different patterns retain
independent learned weights and may have different hidden widths.

```{mermaid}
flowchart TD
    R["Reactions from multiple compounds, grouped by pattern"] --> Q["Reactant atom features + SMARTS-position one-hot"]
    Q --> QH["Linear + Centrality: R x P x D"]
    A["All atoms and bonds of source compounds"] --> G["Pattern-owned one-hop GNN + Centrality"]
    G --> KV["Shared per-compound keys and values"]
    R --> B["Reactant-to-all-atom Spatial + Edge bias"]
    A --> B
    QH --> C["Cross Attention: R x heads x P x N"]
    KV --> C
    B --> C
    C --> L["Residual + LayerNorm; FFN + Residual + LayerNorm; repeat L times"]
    L --> F["Flatten reactant tokens + Linear + rule/product embeddings"]
    F --> H["Reaction features: R x H"]
```

The downstream `ReactionRelationEncoding` retains every tied nearest SMARTS
position for each directed reaction pair. Each pattern owns a bias-free Linear
projection from its position one-hot to the existing shared reaction width.
Every nearest position is projected separately and contributions are **summed**,
never averaged. Endpoint discovery uses the existing
`reaction_transformer.max_reaction_distance` cutoff; more distant/disconnected
pairs keep their far-distance bucket and have no endpoint contribution.
Sparse endpoint indices feed query/key biases and value updates;
a dense `[compound, reaction, reaction, hidden]` endpoint tensor is unnecessary.
Position projections are saved, remapped by pattern identity, and frozen with
imported pattern features. No separate role-width setting is introduced.

```{mermaid}
flowchart LR
    P["Reaction pair: all tied nearest SMARTS positions"] --> O["One-hot for each position"]
    O --> W["Pattern-owned Linear: P to H"]
    W --> S["Sum all tied contributions"]
    D["Minimum atom distance + chemical relations"] --> E["ReactionRelationEncoding"]
    S --> E
    E --> M["Reaction-memory attention bias and value updates"]
```

Configure each pattern through
`reaction_model_params.pattern_model_params["PATTERN_ID"]`:

| Parameter | Default | Meaning |
|---|---|---|
| `hidden_dim` | Shared reaction width, normally 128 | Query, GNN and attention width |
| `num_layers` | 2 | Number of cross-attention/FFN blocks |
| `num_heads` | Shared reaction heads, normally 4 | Attention heads; must divide `hidden_dim` |
| `ffn_mult` | 4 | FFN intermediate width multiplier |
| `max_degree` | 4 | Degree embedding clipping limit |
| `max_spatial_dist` | 4 | Distance embedding clipping limit |
| `max_edge_dist` | 4 | Maximum encoded bond-path length |

All values above are positive integers. Each parameter also has a CLI flag of
form `--pattern-hidden-dim ID=VALUE`, `--pattern-num-layers ID=VALUE`,
`--pattern-max-edge-dist ID=VALUE`, etc. Workbench exposes the same fields in
shared and individual reaction architecture settings. `dropout` is controlled by
the existing shared training `--dropout` setting; frozen imported patterns retain
their saved dropout. All bond channels produced by MolGraphBuilder are used in both GNN messages and
edge-path encodings: four bond-type channels and five ring-type channels. Bond
stereochemistry and conjugation are not separate MolGraphBuilder features.
The GNN always has exactly one layer. Atom-label vocabulary,
SMARTS-position count and bond-feature width are derived from chemistry inputs.
`reaction_model_params.hidden_dim` controls the final shared reaction width and
remains independent of each pattern's attention hidden width.

Example:

```json
{
  "reaction_model_params": {
    "hidden_dim": 128,
    "pattern_model_params": {
      "0": {
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "ffn_mult": 4,
        "max_degree": 4,
        "max_spatial_dist": 4,
        "max_edge_dist": 4
      }
    }
  }
}
```

This is a new reaction architecture and requires newly trained reaction weights.
Earlier reaction encoders are not silently converted on checkpoint load. Pattern
export/import, architecture remapping and freezing include the complete query,
GNN, structural encoding, attention and readout parameters. Frozen reaction
features are independent of the trainable shared molecular encoder.

ReactionRelationEncoding represents minimum matched-substructure distance,
chemical pair relations, and projections of every equally nearest SMARTS endpoint
in the downstream reaction memory encoder.

The branch TransformerEncoder computes reaction memory once per source. The
TransformerDecoder receives BOS, main-adduct and root-molecule tokens, followed
by applied-reaction tokens. Its BOS readout represents the current state and
scores candidate reactions against the shared memory. Branch prediction uses
main adduct and molecular structure; CE conditions spectrum scoring separately.

## 3. Train branch and spectrum objectives

The branch loss evaluates connected prefix log scores using saved peak/depth
memberships. At depth two it sums both transition log probabilities. Class balance,
negative averaging and depth/peak averaging happen during training. Negatives
are grouped across the alternative parents for that peak/depth and exclude
transitions supporting other observed peaks.

Spectrum scoring uses local ReactionInstanceEncoder outputs directly, before
branch Transformer encoding. It does not use branch memory or state-decoder
outputs. These reaction-instance features include all-atom cross-attention context and remain separate from branch memory.

Build one graph per spectrum with canonical Compound structures as nodes and
known reactions as edges. Identical structures and repeated physical reaction
edges across histories are merged before GAT evaluation. Initialize each node
from its molecular features, root molecular features, main-adduct embedding and
CE features. Each sparse GAT block aggregates messages in both reaction
directions, distinguished by direction features, then applies residual,
LayerNorm and FFN updates. No dense adjacency or per-node softmax loop is built.
The GAT updates Compound features; local reaction features are reused at every
block. `spectrum_model_params.gat_num_layers` defaults to 2 and must be positive
(training CLI: `--spectrum-gat-layers`; Workbench: **Fragment Graph Attention**).

The common layer implementation follows the
[Arms GAT](https://github.com/MotohiroOGAWA/Arms/blob/main/arms/layers/gat.py)
structure. `GATLayer` exposes `directed`, `softmax_per` (`dst` or `graph`),
positive temperature `T`, and `add_self_loops`. `GATBlock` applies GAT,
residual/dropout/LayerNorm, FFN, then residual/dropout/LayerNorm; its
`ffn_hidden_mult` defaults to 4. `GAT` repeats independent blocks using
`nn.ModuleList`. Spectrum scoring uses `directed=True` because both message
directions, with distinct direction attributes, are supplied explicitly;
attention normalizes per destination (`softmax_per="dst"`, `T=1`). Graph-wide
attention normalization is available in the common layer but is not the final
source-wise event probability normalization.

The Fragment Head scores updated parent/child features, local reaction features
and CE, with a separate bank per main adduct. Child, Stay and Sink events compete
through a source-grouped log softmax. Attention self loops only update features;
they are separate from the learned Stay event. Histories keep their own valid
next reactions, so merging structures does not permit an incompatible reaction.
Log probability propagates along histories and retained contributions merge by
physical Compound before the existing ion/shift State Head.

The State Head concatenates fragment molecular features, root molecular
features, CE and an applied-reaction-set feature. A spectrum-owned common
`SetTransformer` encodes each unique applied set from local reaction features,
without positional encoding. Empty root sets use its learned PMA seed. Unique
set encodings are reused across spectra and processed in bounded batches.
State scoring groups histories by `(spectrum, Compound, applied reaction set)`.
For example, `A then B` and `B then A` share the same set representation and
state distribution; their retained masses are summed in log space. A different
set such as `{C,D}` receives a separate State Head evaluation, using shared MLP
weights. Ion/shift combinations normalize within each group. Joint probabilities
are `P(Compound, set) * P(ion, shift | Compound, set)` and are summed across sets
only after state scoring. No set-feature averaging or mixture-weight input is
used. Physical `state_probability` output is a reporting marginal, computed from
the summed joint probabilities and physical fragment mass.
Configure `state_set_num_heads` (default 1, must divide reaction width) and
`state_set_num_layers` (default 2) in `spectrum_model_params`, or use CLI
`--state-set-heads` and `--state-set-layers`. Workbench exposes **State Reaction Set**.
These are separate from the branch state decoder and do not alter branch scores.

Formula aggregation, cross-entropy and cosine losses remain unchanged. Spectrum loss trains the GAT, molecular encoder and trainable local
reaction features; it has no gradient path through the branch Transformer.
Training does not apply learned branch pruning before computing spectrum loss.

## 4. Fine-tune or resume

Select imported reaction features during preparation and new reaction architecture
during training. Imported local reaction modules remain frozen; MolEncoder and
predictors train. Destination pattern IDs can change; import bindings preserve the
complete pattern feature modules. Resume restores the checkpoint optimizer and
epoch. See [fine-tuning](fine-tuning.md) and [training](training.md).

## 5. Generate and inspect spectra

Start at the empty reaction set, score compatible next reactions, sum path log
probabilities and apply `branch_path_threshold` and `max_fragment_nodes`.
Materialize selected fragments, run the condition-aware fragment GAT, and evaluate
the Fragment and State Heads. Propagate fragment probabilities through reaction histories, aggregate
formula-equivalent contributions and export spectra. Prediction reads no teachers.
Single and batch prediction share encodings and restore original input order
using `sample_input_index`. See [prediction](prediction.md).

All attention-based residual blocks use post-normalization: attention, residual
addition, LayerNorm, then FFN, residual addition, LayerNorm. This applies to the
common Transformer encoder/decoder, Graphormer self/cross attention, reaction
atom cross attention, SetTransformer, and GAT blocks. No pre-attention memory
normalization or additional final stack normalization is applied by the common
Transformer encoder/decoder.
