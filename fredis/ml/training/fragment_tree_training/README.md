# Fragment-tree training

Training jointly optimizes the CE-independent reaction proposal model and a
CE-conditioned pair of Fragment and State Heads. The spectrum model first
propagates fragment probabilities, then assigns conditional ion/shift probabilities. See [the architecture](../../specgen/SOURCE_REACTION.md).

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir prepared \
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
L_s = P_s                                                       # positive only
L_s = N_s                                                       # negative only, weight > 0
L_branch = sum_s L_s
```

Each source is a (BranchGroup, source Compound) pair with supervised edges.
Negative-only terminal nodes use L_s = N_s when negative_weight > 0; with zero
negative_weight they contribute zero. Class counts are averaged separately
within each source; source losses are
summed, not averaged across sources or the batch. Thus batch source count affects
branch loss scale relative to spectrum loss. Diagnostic `branch_positive_loss`
and `branch_negative_loss` sum the respective source means before weighting.
An empty objective gives differentiable zero. Labels excluded with -1 contribute
no loss or gradient. `--reaction-edge-layers` controls this MLP's hidden layers.
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
is the precursor root or an endpoint of a positive edge in the same branch group
(including positive terminal nodes). Positive
evidence is pooled across CE measurements before selecting these sources.
By default, preparation saves only positive/weak-negative edges, their endpoint
Compounds, and source roots (including isolated roots). Use `--save-all-edges`
(Workbench: **Save all candidate edges and nodes**, config `saveAllEdges`) to
retain the complete candidate graph. Additional edges have label -1 and do not
change branch supervision or its normalization. Spectrum candidate coverage can
change with graph retention; this option guarantees identical branch labels,
not identical spectrum losses.

Spectrum loss uses `--intensity-exponent` (default `0.5`, positive finite values only). Target intensities are raised to this power before normalization into cross-entropy weights; both predicted and target intensities are raised to this power for cosine similarity. Use `1` for the original objective. Prediction output intensities are unchanged. Resuming an older checkpoint without this setting retains exponent `1`.

Reaction-path probabilities are propagated in log space. Transition and state probabilities use grouped log-softmax; incoming paths, execution histories, and formula-equivalent peaks are combined with logsumexp. Cross entropy consumes the merged log probability directly, without a probability floor. Precursor-excluded normalization also uses logsumexp. Cosine inputs are centered in log space before exponentiation; prediction outputs remain ordinary intensities.


## Training report metrics

`spectrum_loss` (also recorded as `intensity_loss`) is the sum of the
with-precursor and without-precursor objectives. `cross_entropy_loss` and
`cosine_loss` record their weighted contributions, so their sum equals
`spectrum_loss`. All three have separate `_with_precursor` and
`_without_precursor` charts. Raw cross-entropy and cosine diagnostics remain
available separately.

`samples` and `num_*` counts report means per batch iteration. `batches`
reports the batch count for an epoch or validation pass, with no train-window
series. `samples_per_compound` divides each batch's sample count by its number
of distinct compound SMILES; epoch and train-window values average these ratios.
`total_samples` and `total_batches` retain interval totals in Dataset;
Throughput contains the per-iteration counts. `loop_seconds` averages batch
preparation, forward evaluation, metric collection, and, during training,
backward and optimizer work. It excludes report publishing and free-running
validation. `epoch_seconds` continues to report the complete epoch duration.
The Display selector enables only modes supported by the selected series:
mean (or a scalar value), median, and distribution with all five quantiles.

Free-running validation evaluates fragment-ion adduct and hydrogen-shift
assignments conditional on an annotated fragment being generated. For each
eligible non-precursor observed peak, select the highest-intensity predicted
state among its annotated fragment identities, then compare its adduct and shift
with the allowed annotations for that identity. Individual accuracies accept any
allowed value; joint accuracy requires an allowed pair. Accuracy is weighted by
the selected fragment SMILES' total retained predicted intensity. Missing fragments reduce
`ion_assignment_coverage` and are excluded from conditional accuracy. Spectra
with no eligible comparisons do not contribute to the accuracy distribution.
These are annotation-consistency metrics, rather than uniquely established
chemical ground truth. Flat, main-adduct, collision-energy, and depth series are
saved in per-chart files under `metrics/distributions/` during subsequent validation runs.


## Per-chart metric storage

`metrics/index.json` contains the available series names and their file paths.
Scalar histories live in `metrics/scalars/`; distribution histories live in
`metrics/distributions/`. A partition contains one chart metric and facet
(e.g. cosine similarity by depth), including its splits, categories, and
statistics. Precursor variants that share a chart share a partition; the
separate with/without-precursor loss charts use separate partitions.

The result viewer initially reads only the catalog and report metadata. It
loads chart files when a chart is selected, reuses them when changing display
controls, and reloads only selected charts on refresh. Training progress and
hyperparameter tuning also support the partitioned format.

Existing TSV runs remain readable. To split one without modifying its original
TSV files, run from the application directory:

```bash
python -m fredis.ml.training.fragment_tree_training.metric_store /path/to/run
```

Migration records the original file sizes and timestamps. If an old training
process subsequently changes those files, the viewer falls back to the live
legacy logs instead of showing a stale converted snapshot. Migration preserves
historical values and reconstructs the weighted loss components when the
training configuration is available.


## Validation spectrum datasets and selection

Each validation writes `spectrum_validation/epoch_N.msds` (or `step_N.msds`
for intermediate validation). The file contains the deduplicated union of
q10/q25/q50/q75/q90 representatives selected by assignment score and cosine,
with and without precursor peaks: at most 20 examples, each stored as a pair
of generated and observed spectra. Metadata includes SMILES, all four scores,
epoch/global step, adduct, collision energy, pair index, and selection reasons.
Small JSON previews support fast browsing; the MSDataset files can also be
opened in the msentity spectrum viewer.

The report plots assignment score on X and cosine on Y. Number and slider
controls select the epoch artifact. Clicking a point shows its mirror spectrum
and SMILES; a selector also makes overlapping points accessible. The precursor
checkbox changes the scores used for coordinates, retaining the entire union
of representatives and all displayed peaks. Missing scores remain selectable
as hollow points at the axis edge.

To convert existing epoch JSON artifacts without replacing the original files:

```bash
node fredis_workbench/scripts/convert-validation-spectra.js /path/to/run
```

The converter streams large legacy files, preserves their representative
selection, and recovers Source SMILES from the matching prepared validation
manifest. It writes `.msds` files and compact previews for subsequent epoch
switching. Older artifacts that saved fewer representatives retain those
examples; missing historical examples are not fabricated.


The selected representative is displayed by the unmodified msentity
`spectrum.js` and `spectrum.css`, loaded from
`fredis/libs/msentity/vscode-extension/media/` in a source checkout. Packaged
Workbench builds bundle those same assets. The iframe keeps the upstream
mirror comparison, zoom, peak tables, metadata, and SVG/PNG/TSV/CSV export
controls isolated from the report styles. The Peaks table stays on the right;
the iframe fits its complete content without an internal scrollbar. Saved validation scores are labelled separately from msentity's
interactive similarity calculation, which uses its own method and Da tolerance.
Updating the msentity assets changes this display after the Workbench is
reloaded (or rebuilt for a packaged installation); protocol changes may require
updating the small comparison adapter.


Time-series chart X coordinates use display-only fractional epochs. Between
completed epoch boundaries `(step_N, N)` and `(step_Nplus1, N+1)`, the position is
`N + (step - step_N) / (step_Nplus1 - step_N)`. The original step values remain
unchanged in storage. Indexed runs read a small scalar partition for boundary
metadata rather than loading all chart files. An unfinished epoch uses the
latest completed interval as an explicitly marked estimate; the next completed
boundary corrects the positions. Before any completed interval exists, charts use a Step axis. Once the first
epoch completes, they switch automatically to fractional Epoch coordinates.

The validation-spectrum number input and slider also include intermediate
`step_N` artifacts at their fractional epoch positions. The Validation selector
can distinguish artifacts that share an epoch coordinate. No fractional epoch
values are written into metrics or spectrum files.

Branch classification charts (`branch_accuracy`, `branch_precision`,
`branch_recall`, `branch_f1`, `branch_balanced_accuracy`) use sigmoid(logit) >= 0.5.
Epoch, intermediate-validation, and train-window summaries sum TP/TN/FP/FN
across batches before computing ratios. Labels -1 are ignored; label 0 is a
weak negative, so these metrics measure agreement with supervision rather than
chemical impossibility. Undefined ratios are omitted. These charts are scalar
summaries, independent of generation path and peak thresholds.

Free-running validation applies `peak_intensity_threshold` after summing
identical physical formulas. Peaks at or below the threshold are removed from
both generated and reference spectra (references use their own maximum intensity). Retained peaks are used
for similarity, assignment scores, representative spectra, and depth metrics.
Top-1 ion-assignment accuracy considers only states contributing to retained
peaks; filtered fragments reduce coverage rather than conditional accuracy.
Hydrogen-shift MAE is no longer recorded or offered as a chart.

Single and batch prediction expose a separate numeric peak intensity threshold,
with default 0, independent of the saved model's validation threshold. Set
"Peak intensity threshold" in Workbench, `--peak-intensity-threshold` on the
prediction CLI, or `peak_intensity_threshold=0.0` in the Python dataset API.
Peaks at or below that value are removed. The value is saved in prediction
configuration and does not change training or validation settings.

Ion-assignment accuracies use predicted-fragment-intensity weighting within each
spectrum: sum(weight * correct) / sum(weight). The weight is the sum of predicted
intensities over retained ion states sharing the selected fragment SMILES; the
highest-intensity state still determines its Top-1 adduct and hydrogen shift.
Teacher-disallowed predictions remain in the denominator and score zero for
whichever correctness conditions they fail. A pair absent from the allowed set
always scores zero for joint accuracy, even if both components occur separately.
Zero total weight gives an undefined accuracy; coverage remains an unweighted
fraction of eligible annotated peaks. Across-spectrum summaries retain their
existing per-spectrum aggregation.
