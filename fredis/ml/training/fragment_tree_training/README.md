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

Branch labels are positive (`1`), weak negative (`0`), or ignored (`-1`). For
each (branch group, source compound), compute the mean positive BCE and mean
weak-negative BCE separately, combine them using `negative_weight`, and divide
by the sum of weights for classes present. Sum these contributions across
sources. Negative-only sources contribute zero when `negative_weight=0`.

Training uses prepared tensors. Chemistry and discrete graph construction
remain outside the differentiable training path.

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
