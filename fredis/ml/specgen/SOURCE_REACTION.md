# Source-anchored competitive fragmentation model

The architecture is `fragment-tree-competitive-ion`. Prepared training tensors
use schema `fredis.fragment-tree-training`. Regenerate prepared data and
train a new checkpoint for this architecture; retired ion-embedding models are
not converted or silently accepted.

## Source reactions and search

Each reaction instance is encoded from the original source molecule using a
pattern-owned external-neighborhood encoder, reactant Graphormer, optional
global attention, and pattern readout. Reaction-set attention uses a BOS token.
The main precursor adduct conditions branch selection; CE does not change the
structural search. Each (molecule, main adduct) shares a branch graph across CE.

Batch prediction computes conflict, invalidation, and dominance relations in
CPU preparation workers, once per unique molecule. Inference reuses them.
Search retains the best proposal score per reaction set, then restores all
valid edges between accepted sets before materializing chemical structures.

## Independent condition heads

There is one independent transition-head bank per main adduct. Its conditions
are the Cartesian product of that rule's `fragment_ion_adducts` and the inclusive
`hydrogen_shift_min ... hydrogen_shift_max` range. Conditions are ordered by ion
first, then increasing hydrogen shift.

For example, `[M+Na]+` with two fragment-ion adducts and six shifts has 12
conditions. Each layer has `weight[12, output_width, input_width]` and
`bias[12, output_width]`. `[M+H]+` uses another bank with entirely separate
parameters. No learned fragment-adduct, hydrogen-shift, or ion-state embedding
is used. Hidden layers use GELU followed by configurable dropout; the last
layer produces an event logit.

Inputs are parent molecular embedding, child molecular embedding, the added
reaction embedding, and CE features. Optional spectrum descriptors are appended
to each molecular embedding after signed-log scaling. The learned precursor
adduct embedding remains a condition of branch search, not an input to these
transition heads. The head bank itself supplies main-adduct specificity.

With graph width 64, reaction width 128, CE width 16, no descriptors, and hidden
widths `[128,64]`, each bank contains weights `[12,128,272]`, `[12,64,128]`, and
`[12,1,64]`. These widths are configurable. The source molecule in the transition
input means the immediate parent structural node; the child is the product of
that structural transition.

## State graph and probability flow

Every structural node has the full ion/shift condition grid, including conditions
that later give invalid formulas. Every structural edge connects its parent
conditions to all child conditions. Child condition selects the scoring weights.
No formula, mass, charge, or elemental-neutral-loss test controls neural states,
edges, or logits. Every state also has a self event scored using the same bank,
the state's condition, two copies of its molecular embedding, zero reaction
features, and CE. Outgoing events and persistence compete in a parent-wise
softmax. Leaf states have only their self event.

Initial probability is placed on the original source-root state with fragment
adduct equal to the main adduct and hydrogen shift zero, using categorical
labels only. The ion rules must include that state. Initialization no longer
matches downstream neutral-loss states by formula. The declared precursor
adduct continues to be validated in CPU input preparation, but does not change
this main-adduct source-root initialization.

`propagation_steps` applications of the transition matrix produce final state
occupancies. Self events count as propagation steps. Alternative pathways add
probability. Intermediate occupancy is not accumulated as intensity.

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
loss. Teacher intensities for equivalent candidates use maximum, not sum.
The complete probability distribution, including invalid/unobserved states,
competes for mass; there is no independent intensity or confidence classifier.
Confidence is one in the viewer interface.

Training uses prepared structural graphs. Spectrum loss differentiates through
the transition heads, CE, reaction features and molecular encoders, but not
through discrete proposal selection or RDKit materialization.

## CLI and Workbench

| CLI argument | Setting | Default |
|---|---|---|
| `--transition-fc-dims` | `transition_fc_dims` | `128,64` |
| `--transition-dropout` | `transition_dropout` | `0` |
| `--propagation-steps` | `propagation_steps` | `6` |
| `--ce-fc-dims` | `ce_fc_dims` | `32` |
| `--collision-energy-feature-dim` | `collision_energy_dim` | `16` |
| `--ce-center` | `ce_center` | `0` |
| `--ce-scale` | `ce_scale` | `100` |
| `--adduct-embedding-dim` | branch-search `adduct_embedding_dim` | `32` |

CE is a dropout-free ReLU MLP of `(CE - ce_center) / ce_scale`. Workbench exposes
condition-head widths and dropout; branch adduct and CE settings start collapsed.
