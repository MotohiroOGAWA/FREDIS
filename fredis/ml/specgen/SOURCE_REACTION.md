# Source-anchored competitive fragmentation model

The architecture is `fragment-tree-competitive-ion`. Prepared training tensors retain
schema `fredis.annotated-spectrum-training`; prepared data includes the required peak/depth BranchTeacher. Regenerate
datasets without this teacher. Prediction heads must be trained with the current architecture.

## Primitive reaction features

Each reaction pattern owns a complete raw-atom encoder. Reactant queries use all
MolGraphBuilder atom features plus SMARTS-position one-hot labels. A one-hop GNN
with full atom/bond features provides molecular keys and values. Pattern-owned
centrality, spatial and edge-path encodings condition repeated post-LayerNorm
cross-attention/FFN blocks. A direct Linear readout and additive rule/product
embeddings produce each reaction feature. These modules do not depend on the
whole-molecule MolEncoder, so updating MolEncoder does not change frozen
primitive reaction features.

Reaction-memory relative encoding uses minimum matched-substructure distances,
chemical relations, and the SMARTS positions of all tied nearest endpoints.
For each pattern, a bias-free Linear maps a position one-hot directly to the
shared reaction hidden width. All tied positions are projected separately and
summed. Endpoint discovery is bounded by the existing reaction-memory spatial
cutoff; distant/disconnected pairs receive no endpoint contribution.
The implementation uses equivalent weight-column lookups and sparse
pair/position indices instead of materializing repeated one-hot vectors or dense
pair embeddings. Both source and destination endpoint positions affect the
query/key biases and value updates. Position projection weights are included in
pattern export/import, identity remapping and fine-tuning freezing. There is no
separate role-width parameter or expansion adapter.

Primitive cross attention is rectangular: each reaction's reactant atoms query
all atoms of its own molecule. Same-pattern reactions from different molecules
are batched together. Spatial/edge biases are computed only for unique query
atoms and reused across matches, using sparse bond propagation rather than an
all-atom square matrix.

## Reaction memory, state decoding and search

The branch TransformerEncoder contextualizes reaction tokens
with GRPE spatial, edge, and value encodings from the original source graph.
Spatial distance is the minimum atom distance between the complete matched
substructures, rather than between selected reaction centers. Distances and
shared/retained-atom relations are computed with device-side tensor reductions
and cached. The default distance cutoff is four bonds.
The branch TransformerDecoder encodes BOS, main-adduct and root-molecule
condition tokens, and the applied reaction set with bidirectional self-attention
and cross-attention to that memory. Learned type embeddings distinguish the
four token kinds. Root tokens reuse trainable MolEncoder graph features;
condition projections are computed once per source or branch group. It has no
absolute positions. BOS readout is invariant to applied-reaction ordering.
Branch queries combine this state with main-adduct features, and dot-product
scoring against encoded reaction keys produces independent log probabilities
using `logsigmoid`. Path scores are sums of transition log probabilities.
The main precursor adduct conditions branch selection; CE does not change the
structural search. Each (molecule, main adduct) shares a branch graph across CE.

Batch prediction computes conflict and invalidation relations in
CPU preparation workers, once per unique molecule. Inference reuses them.
Chemistry and formula candidates are prepared before neural scoring, using the
same components as training. Edge selection keeps reachable admissible histories
within the path threshold and unique-Compound budget. See [shared inputs](../input/README.md).

## Fragment Head and topological probability flow

Each main precursor adduct owns an independent Fragment Head, shared by its
fragments. A separate sparse GAT first updates Compound graph features using
known reaction edges, root structure, main adduct and CE. Repeated full GATBlocks
include attention, residual/dropout/LayerNorm and FFN/residual/LayerNorm.
`spectrum_model_params.gat_num_layers` defaults to 2. The common GAT classes
expose direction, destination/graph normalization, temperature and self-loop
settings from the Arms implementation. Spectrum supplies explicit forward and
reverse message edges with direction flags, using destination normalization.

Canonical structures and physical reaction edges are deduplicated within each
spectrum before the GAT; separate histories reuse these updated representations.
Reaction edge features are local ReactionInstanceEncoder outputs, before the
branch Transformer. The Fragment Head inputs are updated parent/child features,
local reaction features, and CE. Optional signed-log spectrum descriptors
are appended to molecular embeddings. The main adduct selects the bank; ion
adduct and hydrogen shift do not enter this head.

`ConditionHead` supplies three independent MLP slices: Child (0), Self (1), and
Sink (2). Self and Sink use two copies of the parent's molecular embedding and
zero reaction features. Each parent has one Self, one Sink, and its structural
Child events, including at terminal fragments. Spectrum scoring does not use
branch memory or state-decoder outputs. All events compete in one parent-history
log softmax; exponentiating its outputs gives probabilities:

```text
P(Self | i) + sum_j P(j | i) + P(Sink | i) = 1
```

Initialize incoming mass to 1 at each sample's original Source fragment. Process
its fragment history graph in topological order, after every parent has contributed:

```python
final_fragment_prob[i] = incoming[i] * p_self[i]
sink_prob[i] = incoming[i] * p_sink[i]
for child in children[i]:
    incoming[child] += incoming[i] * p_child[i, child]
```

Self and Sink mass are final and never propagated again. Sink is unobserved
additional fragmentation or disappearance. Multiple-parent contributions add.
Terminal fragments predict Self and Sink with the same head. For each sample,
`sum(final_fragment_prob) + sum(sink_prob) = 1`. Structural cycles are expanded into finite compatible reaction histories before
propagation; molecular structures remain shared.

## State Head and joint probabilities

Only after fragment propagation finishes, each main-adduct-specific State Head
predicts the full Cartesian product of fragment-ion adducts and hydrogen shifts.
The head is shared across all fragments. Inputs are the fragment molecular
embedding, the original Source molecular embedding, CE features and an applied
reaction-set embedding. A separate common SetTransformer encodes local reaction
features once per unique applied set, with padding masks and no positional
encoding; empty sets use the learned PMA seed. Its defaults are one attention
head and two layers (`state_set_num_heads`, `state_set_num_layers`). Multiple
histories with the same Compound and applied set share one state distribution;
their retained masses are summed in log space. Different sets are evaluated
separately with shared State Head weights and their joint ion probabilities are
summed afterwards. Set features are never averaged.
The main
precursor adduct selects the bank. `ConditionHead` has one MLP slice per ion/shift
combination. Softmax is over all states of each sample/fragment/reaction-set group:

```text
sum_k P(State_k | Fragment_i, Set_s) = 1
P(Fragment_i, State_k) = sum_s P(Fragment_i, Set_s) * P(State_k | Fragment_i, Set_s)
```

State probabilities do not enter fragment events or reaction history graph propagation. Formula,
mass and charge annotations do not control either softmax. Different ion/shift
conditions remain separate even when their final formulas match. Formula-level
probabilities are summed only after the joint probabilities have been computed.

Output `fragment_probability`, `fragment_node_index`, and `fragment_sample_index`
describe sample-specific fragment rows. `state_fragment_index` maps each state
row to its fragment. `state_probability` reports the physical-fragment conditional
marginal after summing the separate reaction-set joint probabilities; it is not
a distribution predicted from averaged reaction-set features.
`ion_probability` remains the joint probability used by loss and peak export.
`sink_probability` is total sink mass per sample. Transition diagnostics index
fragment rows, with target `-1` for Sink and reaction indices `-1` for Self and
`-2` for Sink. `probability_mass_error` includes sink mass.

## Formula and m/z annotations

Inference computes mmkit annotations after scoring and propagation:

```python
shift = Adduct.from_dict({'H': hydrogen_shift})
formula = fragment_ion_adduct.apply_to_formula(
    shift.apply_to_formula(fragment_compound.formula)
).normalized
mz = formula.exact_mass / abs(fragment_ion_adduct.charge)
```

These values are output annotations, not learned transition inputs. Shared preparation computes the same annotations for training and prediction;
training additionally assigns observed peak targets. Equivalent formulas are merged only for supervision and output,
not to merge neural states before propagation. Invalid negative-element
formulas and non-positive masses are omitted from exported peaks. The internal
probability mass on such states remains visible and is not redistributed during
propagation. Display normalization uses valid formula peaks only.

## Loss

Branch supervision uses prepared peak/depth path MIL with positive/weak-negative/ignored
labels (1/0/-1). `negative_weight` controls weak-negative weight.
The spectrum objective sums probabilities by sample/formula, compares them with
sum-normalized observed intensities by cross entropy, and adds a weighted cosine
loss. The spectrum objective sums equally weighted precursor-inclusive and
precursor-excluded terms. The excluded term removes the formula-level peak of
Source with the declared precursor adduct and its hydrogen-shift variants,
then separately sum-normalizes the
remaining prediction and target. Samples with no remaining positive targets
contribute only the inclusive term. Teacher intensities for formula-equivalent states use maximum, not sum.
The complete probability distribution, including invalid/unobserved states,
competes for mass; there is no independent intensity or confidence classifier.
Confidence is one in the viewer interface.

Training uses prepared structural graphs. Spectrum loss differentiates through
the Fragment Head, State Head, CE, reaction features and molecular encoders, but not
through discrete proposal selection or RDKit materialization.

## CLI and Workbench

| CLI argument | Setting | Default |
|---|---|---|
| `--fragment-fc-dims` | `fragment_fc_dims` | `128,64` |
| `--state-fc-dims` | `state_fc_dims` | `128,64` |
| `--fragment-dropout` | `fragment_dropout` | `0` |
| `--state-dropout` | `state_dropout` | `0` |
| `--ce-fc-dims` | `ce_fc_dims` | `32` |
| `--collision-energy-feature-dim` | `collision_energy_dim` | `16` |
| `--ce-center` | `ce_center` | `0` |
| `--ce-scale` | `ce_scale` | `100` |
| `--adduct-embedding-dim` | branch-search `adduct_embedding_dim` | `32` |

CE is a dropout-free ReLU MLP of `(CE - ce_center) / ce_scale`. Workbench exposes
Fragment/State Head widths and dropout; branch adduct and CE settings start collapsed.

Both heads have configurable hidden widths and dropout in Workbench and CLI.
Retired coupled Transition Head checkpoints are incompatible: the parameter
banks and softmax meanings have changed. Train a new checkpoint. Branch Decoder,
ReactionEncoder, MolEncoder, prepared Teacher Graphs and Learned Graph selection
keep their existing roles.
