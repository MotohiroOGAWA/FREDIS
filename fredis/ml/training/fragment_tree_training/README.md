# Fragment-tree training

Training has four independently weighted objectives. The branch model predicts reaction paths once per
`(compound, main adduct)` group, without collision energy. The intensity model
then predicts each prepared physical-ion candidate independently for every
spectrum, using main adduct and collision energy as separate inputs. Separate
fragment-presence and `(adduct, hydrogen shift)` state heads determine the
physical-candidate probability.

```bash
python -m fredis.cli train fragment-tree \
  --train-dir prepared/train_structures \
  --val-dir prepared/validation_structures \
  --output-dir training/run \
  --node-dim 32 --graph-dim 128 \
  --epochs 20 --device cuda \
  --branch-weight 1 --negative-weight 0.2 \
  --branch-mil-temperature 0.1 \
  --fragment-loss-weight 0.5 --state-loss-weight 0.5 \
  --intensity-loss-weight 1 --cosine-loss-weight 0.5 \
  --precursor-free-loss-weight 0.5 --intensity-power 0.5 \
  --branch-path-threshold 0 --max-fragment-nodes 100
```

This command initializes the Mol Encoder from its architecture options
and trains it jointly with the fragment-tree model.

`max_reaction_count` is inherited from the prepared dataset. A zero branch path
threshold disables score pruning. Search always deduplicates normalized states,
keeps the highest cumulative log probability, and applies the fragment-node
budget once to the shared branch group. The branch path threshold, node budget,
ion prediction threshold and peak intensity threshold form `generation_params`,
which is saved with the model separately from the architecture sections and can
be overridden at prediction. Loss weights and `intensity_power` are training
settings stored in the checkpoint, not in `model_config`.

## Branch objective

Every transition log probability is `logsigmoid(logit)`. A path score is the
sum of its transition log probabilities. Alternative explanations for one peak
use a normalized smooth maximum:

```text
T * (logsumexp(path_score / T) - log(number_of_paths))
```

Preparation unions positive paths over every collision energy in the branch
group. At each teacher-supported depth, a chemically valid continuation absent
from that union is stored as a weak negative. The branch loss combines positive
MIL with `negative_weight * mean(softplus(negative_logit))`.

## Fragment, ion-state, and intensity objectives

Preparation groups equivalent final formula/adduct/charge states into one
physical candidate while preserving every `(adduct, hydrogen shift)`
explanation. The model embeds both fields and aggregates equivalent
explanations with normalized attention. A fragment logit is predicted once per
fragment and a state logit once per explanation. The inference probability is
exactly `sigmoid(fragment_logit) * sigmoid(state_logit)` (combined in log
space); there is no third BCE on that product. It never sums one intensity per
explanation. Physical candidates use independent softplus intensities, so
several ion states of one fragment can be non-zero simultaneously.

Fragment and state BCE are class-balanced 50:50 whenever both classes exist;
if only one class exists, that class has weight 1. Positive examples use the
maximum matched, max-normalized peak intensity as their weight. The weight is
raised to `intensity_power`, as are max-normalized targets inside the
full-spectrum and precursor-free intensity losses. Negative classification
examples are unweighted. Fragment and state labels are independent: states are
not dropped merely because their fragment label is negative. Inference
max-normalizes raw intensity outputs without applying `intensity_power`.

The spectrum model loss is
`fragment_loss_weight * L_fragment + state_loss_weight * L_state + intensity_loss_weight * L_intensity`;
the total adds `branch_weight * L_branch` once.

## Prepared-tensor boundary

Training consumes source/fragment graphs, reaction/state transitions, MIL paths,
physical candidates, explanation indices, masses, formulas, and targets already
stored by preparation. Forward, loss, backward, and optimizer steps perform
only tensor and neural-network operations. RDKit parsing, fragmentation,
materialization, ion chemistry, formula calculation, and graph construction are
not permitted in the training hot path.

When the prepared-data contract changes, regenerate it from the original
spectra.

`--resume` requires a checkpoint written by this training command, including
its `training_settings`.

## Metrics and outputs

Reports include branch losses, fragment/state positive and negative losses,
counts, probability means, precision/recall/F1, mean physical probability,
generated-tree search statistics, physical candidate/explanation counts, full
and precursor-free cosine, and intensity MAE. Validation also generates spectra
with the inference path. TensorBoard, JSON/TSV reports, last/best checkpoints,
and detailed spectrum-validation artifacts are written to the run directory.

Run the focused regression suite with:

```bash
pytest -q tests/ml/reaction/test_fragment_tree_model.py
```
