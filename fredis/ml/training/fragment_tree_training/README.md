# Fragment-tree training

Training jointly optimizes the CE-independent reaction proposal model and a
CE-conditioned pair of Fragment and State Heads. The spectrum model first
propagates fragment probabilities, then assigns conditional ion/shift probabilities. See [the architecture](../../specgen/SOURCE_REACTION.md).

```bash
python -m fredis.cli train fragment-tree \
  --train-dir prepared/train_structures \
  --val-dir prepared/validation_structures \
  --output-dir training/run \
  --node-dim 32 --graph-dim 128 \
  --epochs 20 --device cuda \
  --branch-weight 1 --negative-weight 0.5 \
  --intensity-loss-weight 1 --cosine-loss-weight 1 \
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

Preparation shares one CompoundReactionGraph across groups. Each BranchGroup
selects edge IDs by CSR and supplies aligned labels: 1 positive, 0 weak negative,
-1 ignored. Positive evidence is unioned across the group's CE observations.
The branch MLP scores parent Compound, child Compound, concrete reaction and main
adduct, without CE. The weighted binary cross entropy is

```text
P_s = mean_positive_at_source_s softplus(-logit)
N_s = mean_weak_negative_at_source_s softplus(logit)
L_s = (P_s + negative_weight * N_s) / (1 + negative_weight)  # with negatives
L_s = P_s                                                       # otherwise
L_branch = sum_s L_s
```

Each source is a (BranchGroup, source Compound) pair with at least one positive
edge. Class counts are averaged separately within each source; source losses are
summed, not averaged across sources or the batch. Thus batch source count affects
branch loss scale relative to spectrum loss. Diagnostic `branch_positive_loss`
and `branch_negative_loss` sum the respective source means before weighting.
An empty objective gives differentiable zero. Labels excluded with -1 and nodes
without positives contribute no loss or gradient. `--reaction-edge-layers`
controls this MLP's hidden layers.
Prediction uses the same tables through `SpectrumPredictionData`, without
observations, targets or edge labels. See [input components](../../input/README.md).

## Competitive spectrum objective

Fragment Head predicts Child, Self and Sink with a parent-fragment softmax,
independently of ion/shift conditions. All fragments include Self and Sink.
Topological DAG traversal adds incoming parent mass, then finalizes Self and Sink
mass and propagates only Child mass. Final fragment mass plus sink mass equals
initial mass per sample. State Head is shared across fragments within a main-adduct
bank and predicts a separately normalized ion/shift distribution after traversal.
Its inputs are fragment, original Source and CE features. Joint probability is
fragment final mass multiplied by conditional state probability. Formula-equivalent
joint probabilities are added only afterward. Spectrum loss differentiates through
both heads; discrete graph construction remains outside autograd.
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
loss controls. The model uses configurable CE, Fragment Head and State Head
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

Head configuration: `--fragment-fc-dims`, `--fragment-dropout`, `--state-fc-dims`,
`--state-dropout` (defaults: `128,64`, `0`, `128,64`, `0`). The same fields appear
in Workbench. Old coupled-head checkpoints require retraining.

Weak negatives are limited to admissible outgoing edges whose source Compound
has at least one positive outgoing edge in the same branch group. Positive
evidence is pooled across CE measurements before selecting these sources.
By default, preparation saves only positive/weak-negative edges, their endpoint
Compounds, and source roots (including isolated roots). Use `--save-all-edges`
(Workbench: **Save all candidate edges and nodes**, config `saveAllEdges`) to
retain the complete candidate graph. Additional edges have label -1 and do not
change branch supervision or its normalization. Spectrum candidate coverage can
change with graph retention; this option guarantees identical branch labels,
not identical spectrum losses.
