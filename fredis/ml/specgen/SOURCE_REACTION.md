# Source-anchored competitive fragmentation model

The architecture is `fragment-tree-competitive-ion`. Prepared training tensors
use schema `fredis.fragment-tree-training`. Regenerate prepared data and
train a new checkpoint for this architecture; retired ion-embedding models are
not converted or silently accepted.

## Source reactions and search

Each reaction instance is encoded from the original source molecule using a
pattern-owned external-neighborhood encoder, reactant Graphormer, and
pattern readout. Reaction-set attention uses a BOS token.
The main precursor adduct conditions branch selection; CE does not change the
structural search. Each (molecule, main adduct) shares a branch graph across CE.

Batch prediction computes conflict, invalidation, and dominance relations in
CPU preparation workers, once per unique molecule. Inference reuses them.
Search retains the best proposal score per reaction set, then restores all
valid edges between accepted sets before materializing chemical structures.

## Fragment Head and topological probability flow

Each main precursor adduct owns an independent Fragment Head, shared by its
fragments. Its inputs are parent molecular embedding, child molecular embedding,
added reaction embedding, and CE features. Optional signed-log spectrum descriptors
are appended to molecular embeddings. The main adduct selects the bank; ion
adduct and hydrogen shift do not enter this head.

`ConditionHead` supplies three independent MLP slices: Child (0), Self (1), and
Sink (2). Self and Sink use two copies of the parent's molecular embedding and
zero reaction features. Each parent has one Self, one Sink, and its structural
Child events, including at terminal fragments. All compete in one parent-wise
softmax:

```text
P(Self | i) + sum_j P(j | i) + P(Sink | i) = 1
```

Initialize incoming mass to 1 at each sample's original Source fragment. Process
its Fragment DAG in topological order, after every parent has contributed:

```python
final_fragment_prob[i] = incoming[i] * p_self[i]
sink_prob[i] = incoming[i] * p_sink[i]
for child in children[i]:
    incoming[child] += incoming[i] * p_child[i, child]
```

Self and Sink mass are final and never propagated again. Sink is unobserved
additional fragmentation or disappearance. Multiple-parent contributions add.
Terminal fragments predict Self and Sink with the same head. For each sample,
`sum(final_fragment_prob) + sum(sink_prob) = 1`. Cyclic break graphs are rejected.

## State Head and joint probabilities

Only after fragment propagation finishes, each main-adduct-specific State Head
predicts the full Cartesian product of fragment-ion adducts and hydrogen shifts.
The head is shared across all fragments. Inputs are the fragment molecular
embedding, the original Source molecular embedding, and CE features. The main
precursor adduct selects the bank. `ConditionHead` has one MLP slice per ion/shift
combination. Softmax is over all states of each sample/fragment:

```text
sum_k P(State_k | Fragment_i) = 1
P(Fragment_i, State_k) = final_fragment_prob[i] * P(State_k | Fragment_i)
```

State probabilities do not enter fragment events or DAG propagation. Formula,
mass and charge annotations do not control either softmax. Different ion/shift
conditions remain separate even when their final formulas match. Formula-level
probabilities are summed only after the joint probabilities have been computed.

Output `fragment_probability`, `fragment_node_index`, and `fragment_sample_index`
describe sample-specific fragment rows. `state_fragment_index` maps each state
row to its fragment, and `state_probability` is its conditional probability.
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

These values are output annotations, not learned transition inputs. Training
preparation computes the same annotations early solely to assign observed
peak targets. Equivalent formulas are merged only for supervision and output,
not to merge neural states before propagation. Invalid negative-element
formulas and non-positive masses are omitted from exported peaks. The internal
probability mass on such states remains visible and is not redistributed during
propagation. Display normalization uses valid formula peaks only.

## Loss

Branch supervision is positive path MIL plus weighted weak-negative softplus.
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
