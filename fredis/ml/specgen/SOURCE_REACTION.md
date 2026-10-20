# Source-anchored competitive fragmentation model

The current architecture is `fragment-tree-competitive-ion`. Prepared tensors,
analysis artifacts and fine-tuning metadata start at schema/version **1**.
There is one supported model path; older encodings and format conversions are
not implemented. Model configurations are plain `model_config` dictionaries.

## Reaction features and candidate graph

Each source reaction instance has a pattern-owned Reactant Graphormer. Its
inputs are raw chemical atom features, fixed SMARTS-position one-hot features,
and optionally external-neighborhood and whole-molecule features. External
neighborhoods use the original source graph, block paths through other SMARTS
atoms and support any non-negative `neighborhood_hops`, including 4 and above.
No fixed 1/2/3-hop CSR tensors are prepared or stored.

The pattern readout produces `h_reaction` in a common feature space. The
reaction-set Set Transformer summarizes selected reactions with a BOS token;
that state summary is separate from `h_reaction`. Main adduct conditions the
proposal decoder, while CE does not change proposal features.

The decoder proposes bounded reaction sets, retaining the best cumulative
proposal score for duplicate sets. It then retains **all chemically valid
transitions between accepted sets**, including alternate parents. RDKit
materializes accepted reaction sets after decoding. Distinct sets keep distinct
internal states even when their product SMILES match; molecular embeddings can
still be shared. Preparation includes every exhaustively enumerated state,
including alternatives without observed peaks.

## Competitive ion transitions

Each structural state has physical ion candidates with deduplicated
`fragment_ion_adduct × hydrogen_shift` explanations. Both fields are embedded
and passed through a configurable combination MLP. Equivalent explanations
share a physical parent state, whose features are their mean; outgoing child
explanations retain their individual combination embeddings.

For every structural edge and valid parent/child ion pair, a transition MLP
scores the parent and child molecular embeddings, optional fragment descriptors,
reused `h_reaction`, parent ion features, child combination features, main adduct
and CE. Transitions must retain charge and have elementwise nonnegative neutral
loss. These constraints supplement structural compatibility and adduct rules;
they are not a complete model of ion chemistry.

All outgoing events compete with a learned self transition in a source-wise
softmax. Duplicate enumerated events are excluded. Equivalent explanations and
alternative paths contribute by **addition**, not by maximum-path selection.

Starting from the observed precursor ion, probabilities propagate for a fixed
number of steps:

```text
p[t + 1, child] = sum_parent p[t, parent] * T[parent, child]
```

Initialization prefers a matching source-root ion. If a neutral-loss precursor
matches only downstream structural states, its initial mass is distributed
uniformly among those matches. Self transitions mean persistence for one step;
leaf states persist with probability one. `propagation_steps` includes self
transitions and is separate from the maximum reaction count.

Final probabilities are summed by sample and output formula before spectrum
comparison or peak-intensity filtering. Candidate intensities are divided by the
maximum formula-grouped probability in that sample. The spectrum objective is
cross entropy against sum-normalized target intensities plus a weighted cosine
term; proposal branch loss remains separate. There is no independent intensity
regressor, fragment/state BCE classifier or ion-confidence threshold. Output
confidence is one for the viewer interface, not a learned presence probability.
Transition outputs retain each event's source, target, explanation, reaction and
probability; self events use explanation/reaction index -1.

## Configurable MLPs

All three MLPs use `build_fc_layers` from `common/torch_utils/nn_utils.py`:
Linear/ReLU/Dropout hidden blocks followed by a linear output. Widths are
comma-separated positive integers; an empty list selects a linear map.

CE is exactly `MLP((CE - ce_center) / ce_scale)`. There is no sinusoidal encoding
or post-MLP LayerNorm. Defaults center 0 and scale 100 map CE 0..100 to 0..1;
normalization does not convert instrument units or fit dataset statistics.

| CLI argument | Spectrum model setting | Default |
|---|---|---|
| `--transition-fc-dims` | `transition_fc_dims` | `128,64` |
| `--transition-dropout` | `transition_dropout` | `0` |
| `--propagation-steps` | `propagation_steps` | `6` |
| `--ion-state-fc-dims` | `state_fc_dims` | `128` |
| `--ion-state-hidden-dim` | `state_hidden_dim` (output width) | `128` |
| `--ce-fc-dims` | `ce_fc_dims` | `32` |
| `--adduct-embedding-dim` | `adduct_embedding_dim` | `32` |
| `--collision-energy-feature-dim` | `collision_energy_dim` | `16` |
| `--ce-center` | `ce_center` | `0` |
| `--ce-scale` | `ce_scale` | `100` |

Workbench exposes these fields under Spectrum Model Params. Architecture and
normalization are saved with the checkpoint. Candidate proposal pruning limits
the states available to probability propagation, so coverage and predictive
accuracy need evaluation on real datasets.

Precursor adduct identity uses `adduct_embedding_dim` before separate branch and
spectrum projections. The CE MLP has no dropout setting. In Workbench, adduct
and CE settings are collapsed by default.
