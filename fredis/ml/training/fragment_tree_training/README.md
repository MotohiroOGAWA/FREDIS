# Fragment-tree training

Joint training of the CE-independent reaction proposal model and the
CE-conditioned spectrum model. See [architecture](../../specgen/SOURCE_REACTION.md)
and [input data](../../input/README.md).

## Usage

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir prepared \
  --output-dir training/run \
  --node-dim 32 --graph-dim 128 \
  --epochs 20 --device cuda \
  --branch-weight 1 --negative-weight 0.5 \
  --intensity-loss-weight 1 --cosine-loss-weight 1
```

The dataset directory must contain `train_structures` and
`validation_structures`. Chemistry settings and reaction limits are inherited
from preparation. Regenerate prepared data when its schema changes.

Resume with `--resume training/run/last.pt`. The checkpoint restores model,
optimizer, epoch, training settings, and early-stopping state.

## Architecture and fine-tuning

Fine-tuning keeps imported local reaction features frozen and trains the molecular
encoder with downstream predictors. Primitive reaction features do not consume
molecular embeddings and remain unchanged when molecular encoder weights update.

Spectrum prediction uses a separate sparse `GAT` stack over unique Compound
structures and local reaction edges. Configure its depth with
`--spectrum-gat-layers` (default 2), or `spectrum_model_params.gat_num_layers`.
Each block includes attention, residual/LayerNorm and FFN/residual/LayerNorm.
Root structure, main adduct and CE condition the graph features. Child, Stay and
Sink logits normalize per parent history. Ion-state scoring additionally uses
a SetTransformer over applied local reaction sets. Same-set histories share a
state distribution and sum retained mass; different sets are scored separately
and their joint ion probabilities are summed after scoring. Configure it with `--state-set-heads` (default 1) and
`--state-set-layers` (default 2). Spectrum losses remain unchanged. Branch selection keeps its Transformer encoder/state decoder, and
spectrum loss does not backpropagate through those branch-specific layers.

## Objectives

```text
loss = branch_weight * branch_loss + spectrum_loss
spectrum_loss = spectrum_loss_with_precursor + spectrum_loss_without_precursor
spectrum_loss_variant = intensity_loss_weight *
    (cross_entropy + cosine_loss_weight * (1 - cosine))
```

`intensity_loss` is an alias of `spectrum_loss`. The precursor-excluded
objective removes precursor formula groups and renormalizes the remaining
distribution. Samples without positive fragment targets are omitted from that
objective.

`--intensity-exponent` defaults to `0.5`. Targets are raised to this power
before normalization for cross entropy; both predictions and targets use it
for cosine. Output intensities are unaffected.

Branch labels are positive (`1`), weak negative (`0`), or ignored (`-1`).
Preparation saves a `BranchTeacher` for each source. Its units are observed
peaks and reaction depths, not deduplicated target Compounds. A peak's alternative
assigned paths (including different structural explanations) form one reaction history graph of
shared execution-state references. Prefix nodes, exact history transitions,
parent-group negatives and terminal negative-only next-depth bags are all saved.
Another peak's positive transition is never relabeled as negative.

```text
s(path) = sum(logsigmoid(branch_logits))
log P(bag) = logsumexp((1/temperature + 1) * s) - logsumexp(s/temperature)
L_pos(peak, depth) = -log P(bag)
L_neg(peak, depth) = mean(softplus(negative_logits))
L(peak, depth) = (I_pos * L_pos + negative_weight * I_neg * L_neg)
                 / (I_pos + negative_weight * I_neg)
L_branch = sum_branch_groups(mean_peaks(mean_active_depths(L)))
```

The parent group includes all parent States of the peak's candidate prefixes at
that depth. A transition appears once within a negative bag. Class balancing,
negative averaging, depth averaging and peak averaging are computed during
training; no numerical weights are baked into the dataset. Zero negative weight
disables negative-only depth bags. Root-only negative units are retained for
branch groups without any assigned fragment paths.

Centered float32 partition functions summarize complete connected prefixes
without enumerating paths. MIL temperature defaults to `0.1` and remains a model
training setting. The spectrum objective and its targets are unchanged.
Regenerate prepared datasets: files without `branch_teacher` are rejected rather
than reconstructed during training.

Training uses prepared tensors. Chemistry and discrete graph construction
remain outside the differentiable training path.

## Periodic checkpoints

`--checkpoint-interval-epochs N` saves a full resumable checkpoint every N completed
epochs. The default is `0`, which disables periodic saves and does not create the
`checkpoints/` directory. `best.pt` and `last.pt` continue to be saved as usual.
For example, `--checkpoint-interval-epochs 5` writes
`checkpoints/epoch_000005.pt`, `checkpoints/epoch_000010.pt`, and so on. Snapshots
include optimizer, scheduler, random generator states, and training history.
Epoch numbering continues across resumed runs. Metadata paths are relative to
each checkpoint file, including snapshots in the subdirectory.

Resume with `--resume ./checkpoints/epoch_000005.pt` from the run directory, or
select that file in Workbench. Workbench's Resume checkpoint field resolves
`./checkpoints/epoch_000005.pt` relative to Output directory. The periodic save
interval remains editable when resuming.

## Validation and outputs

Validation evaluates prepared-data losses and freely generated spectra.
Generation uses `branch_path_threshold` and `max_fragment_nodes`. The
`peak_intensity_threshold` removes peaks at or below the cutoff from both
generated and reference spectra before scoring. These generation settings do
not filter the training loss.

The run directory contains:

- `last.pt`, `best.pt`: resumable checkpoints.
- `fragment-tree.training-config.pft`: training configuration.
- `result.training.pft`: report entry point.
- `metrics/index.json`, `metrics/scalars/`, `metrics/distributions/`: chart
  catalog and per-chart histories.
- `spectrum_validation/epoch_N.msds`, `step_N.msds`: paired generated/reference
  representatives, their scores, SMILES, and selection metadata.
- Validation JSON files for inspection and legacy compatibility.

Representative spectra are the deduplicated union of q10/q25/q50/q75/q90
examples selected by assignment score and cosine, with and without precursor
peaks. The viewer reads MSDS directly. Metric formulas, aggregation rules, and
interpretation are documented in each chart's **About this metric** section.

Legacy metric TSV files remain readable. To partition an existing run:

```bash
python -m fredis.ml.training.fragment_tree_training.metric_store /path/to/run
```
