# Train and fine-tune

```bash
python -m fredis.cli train fragment-tree \
  --train-dir data/prepared/train_structures \
  --val-dir data/prepared/validation_structures \
  --output-dir models/my-model \
  --node-dim 32 --graph-dim 128 \
  --epochs 100 --batch-size 4 --lr 0.0001
```

The command jointly trains a newly initialized Mol Encoder and the
fragment-tree model.

The Mol Encoder concatenates its graph embedding with an embedding of the
selected molecular descriptors (from `fredis.domain.molecule.descriptors`) and
projects the result back to the graph dimension. Choose them with
`--descriptors` (space-separated, e.g. `--descriptors FormalCharge TPSA`; give
no names to use none). The default is `FormalCharge`. In Workbench they are the
**Molecular Descriptors** checkboxes of the Molecular Architecture settings.
Descriptor values are stored with every prepared graph, so changing the
selection needs no new data preparation.
The model learns CE-independent fragment branches with normalized positive MIL and depth-diverse weak negatives, then predicts CE-dependent physical-ion intensities. Their weights are configurable with `--branch-weight`, `--negative-weight`, `--branch-mil-temperature`, `--fragment-loss-weight`, `--state-loss-weight`, `--intensity-loss-weight`, `--cosine-loss-weight`, `--precursor-free-loss-weight`, and `--intensity-power`. These are training settings and are not part of `model_config`. AdamW uses `--weight-decay`; `--gradient-clip 0` disables gradient clipping.

In **Workbench → Fragment Tree Training**, **Load Configuration** and **Save Configuration** are in the page header; the load control also accepts a dropped `*.pfttrain.json` file. Saved training configurations contain run settings and the trainable Mol Encoder / Branch / Fragment Transformer / Ion-State / Intensity parameters. Fragmenter settings, Reaction Patterns, ion-adduct rules, Symbols, and adduct ordering are inherited from the prepared train/validation datasets.

After selecting both prepared datasets, Workbench checks that their Fragmenter settings, adduct ordering, and Symbols agree. Reaction Pattern Sets and Ion Adduct Rule Sets are shown read-only, while Mol Encoder architecture parameters are editable for joint training.

## Initialize from or resume a checkpoint

Use `--initialize-from models/base/last.pt` to fine-tune compatible model weights with a fresh optimizer and epoch counter. Configuration and prepared structures must match the checkpoint architecture. Use `--resume` to continue an interrupted training run with its optimizer state instead.

## Expand a frozen base

Advanced fine-tuning uses `--fine-tune-checkpoint` and `--adapter-width`. Select a complete expanded reaction pattern set and regenerate both prepared splits with its model configuration before training. The expanded Fragmenter is inherited from those datasets. Existing base parameters remain frozen while the newly added parameters are trained.

## Monitoring

Training writes `last.pt`, `metrics.json` and `metrics.tsv` and emits `epoch_end` JSON Lines. Workbench shows epoch progress, train/validation losses, reaction metrics and bounded searchable stdout/stderr logs. Training processes continue after their panel closes. Restored job metadata distinguishes a missing process from a known successful exit.

### Intermediate validation

For large training datasets, evaluate a validation subset between epoch boundaries:

```bash
python -m fredis.cli train fragment-tree \
  --train-dir data/prepared/train_structures \
  --val-dir data/prepared/validation_structures \
  --output-dir models/my-model \
  --epochs 100 \
  --validation-interval-steps 1000 \
  --validation-fraction 0.1
```

`--validation-interval-steps` counts optimizer updates across epochs and Resume. Its default is `0`, which disables intermediate checks. `--validation-fraction` defaults to `0.1` and must be greater than 0 and at most 1. It selects individual validation samples, rather than a fraction of structure files or batches. The sample count is rounded up, with a minimum of one sample.

A uniformly sampled subset is fixed using the training seed so successive checks are comparable. Resume retains the subset seed and previous intermediate results. The interval and fraction remain editable when resuming. CUDA evaluation runs without gradients, then restores training mode. `--max-samples` continues to limit each validation batch.

In **Workbench → New Training → Optimization and Loss**, set **Validation interval (optimizer steps, 0 disables)** to `1000` and **Intermediate validation fraction** to `0.1`. These values are included in Copy Command, Start Training, and saved training settings. New-form defaults can also be set through `fredis.workbench.trainingDefaults` using `validationIntervalSteps` and `validationFraction`.

Each check emits an `intermediate_validation_end` JSON event with step, sample count, loss components, and reaction/spectrum metrics. Workbench Training Jobs shows the latest result and a loss chart against optimizer steps. Results are saved in `intermediate_validation.json` and `intermediate_validation.tsv`; `intermediate_validation_subset.json` records the selected sample indices. TensorBoard step metrics are stored under `tensorboard/iterations` and are included when opening the parent `tensorboard` directory.

Full validation still runs at each epoch end. Learning-rate scheduling, early stopping, and best-checkpoint selection use full-validation results; intermediate checks are for monitoring.
