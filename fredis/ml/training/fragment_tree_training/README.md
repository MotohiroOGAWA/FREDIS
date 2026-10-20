# Fragment-tree training

Training jointly optimizes the CE-independent reaction proposal model and a
CE-conditioned competitive ion-transition model. The spectrum model propagates
probability through reaction/ion states; it does not regress independent peak
intensities. See [the architecture](../../specgen/SOURCE_REACTION.md).

```bash
python -m fredis.cli train fragment-tree \
  --train-dir prepared/train_structures \
  --val-dir prepared/validation_structures \
  --output-dir training/run \
  --node-dim 32 --graph-dim 128 \
  --epochs 20 --device cuda \
  --branch-weight 1 --negative-weight 0.2 \
  --branch-mil-temperature 0.1 \
  --intensity-loss-weight 1 --cosine-loss-weight 0.5 \
  --branch-path-threshold 0 --max-fragment-nodes 100
```

This command initializes the Mol Encoder from its architecture options
and trains it jointly with the fragment-tree model.

`max_reaction_count` is inherited from the prepared dataset. A zero branch path
threshold disables score pruning. Search always deduplicates normalized states,
keeps the highest cumulative log probability, and applies the fragment-node
budget once to the shared branch group. The branch path threshold, node budget,
peak intensity threshold form `generation_params`,
which is saved with the model separately from the architecture sections and can
be overridden at prediction. Loss weights are training
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

## Competitive spectrum objective

Every child `(fragment_ion_adduct, hydrogen_shift)` state and self
transition competes in a parent-wise softmax. Fixed-step probability propagation
sums all incoming paths. Final ion probabilities are grouped by output formula.
Two equally weighted objectives are added: one includes the precursor peak,
and one excludes its entire formula group and those of its hydrogen-shift
variants. Each uses
`intensity_loss_weight * (cross_entropy + cosine_loss_weight * (1 - cosine))`.
The excluded term normalizes both predictions and targets by their remaining
sample sums; samples with no positive fragment targets are omitted from its
average. The precursor is Source with the declared precursor adduct.
Total loss adds `branch_weight * branch_loss`. Metrics report both spectrum
terms and their cross entropy and cosine separately.

There are no independent fragment/state classification heads or classification
loss controls. The model uses configurable CE, ion-combination and transition
MLPs from `torch_utils.nn_utils`. Early stopping uses `torch_utils.early_stopping`
and resumes the saved best validation loss and non-improvement count.

## Prepared-tensor boundary

Training consumes source/fragment graphs, reaction/state transitions, MIL paths,
ion states, condition indices, masses, formulas, and targets already
stored by preparation. Forward, loss, backward, and optimizer steps perform
only tensor and neural-network operations. RDKit parsing, fragmentation,
materialization, ion chemistry, formula calculation, and graph construction are
not permitted in the training hot path.

When the prepared-data contract changes, regenerate it from the original
spectra.

`--resume` requires a checkpoint written by this training command, including
its `training_settings`.

## Metrics and outputs

Reports include branch losses, transition cross entropy, cosine similarity,
probability-mass error, ion-state counts and generated-tree search statistics.
Validation runs the inference path. TensorBoard, JSON/TSV reports, last/best
checkpoints and detailed spectrum-validation artifacts are written to the run
directory.

Run the focused regression suite with:

```bash
pytest -q tests/ml/reaction/test_fragment_tree_model.py
```
