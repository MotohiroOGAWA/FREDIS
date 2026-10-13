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

## Fine-tune immutable pattern features

Use `--fine-tune-checkpoint models/base/last.pt` with train/validation structures
prepared using the complete expanded pattern set. The checkpoint can be a fresh,
already fine-tuned, or integrated model. Architecture settings are inherited
from it. No adapter layers or low-rank residuals are attached.

| Module | During pattern fine-tuning |
| --- | --- |
| Molecular encoder and descriptor fusion | Frozen, evaluation mode |
| Shared source atom projection, SMARTS roles, status and hop projections | Frozen |
| Existing pattern Graphormers | Frozen, evaluation mode |
| Existing pattern rule/product embeddings and final reaction-feature readout | Frozen |
| New pattern Graphormers and complete feature readouts | Trained |
| Reaction Sequence/SetTransformer, branch scorer, condition projections | Trained |
| Fragment/tree processing, ion classification and intensity predictors | Trained |

Thus the **final reaction feature**, not only a Graphormer intermediate, stays
unchanged for inherited patterns on identical source inputs. Predictions may
change because downstream predictors are trained and additional patterns are
available. Each pattern owns its own rule/product embeddings and final readout,
so growing another pattern's vocabulary does not modify its features.
`trainable_parameters.json` lists the actual trainable tensors for each run.

```bash
python -m fredis.cli train fragment-tree \
  --train-dir data/expanded/train_structures \
  --val-dir data/expanded/validation_structures \
  --output-dir models/expanded --device cuda \
  --fine-tune-checkpoint models/base/last.pt
```

Repeat with `models/expanded/last.pt` and another superset to add a further stage.
Use `--resume models/expanded/last.pt` with the same pattern universe to restore
optimizer/epoch state; this continues the existing stage rather than starting
another expansion. Old GNN/shared-readout or adapter checkpoints require
retraining against this architecture.

## Integrate independent branches

```bash
python -m fredis.cli train integrate-patterns \
  --checkpoints models/pc/last.pt models/pe/last.pt \
  --head-checkpoint models/pc/last.pt \
  --output models/lipids.pt
```

Integration matches patterns by chemical definition, remaps their numeric IDs,
and copies their complete feature encoders. It never averages weights. Shared
feature configurations/weights must match, and duplicate pattern definitions
must have identical complete encoder weights. Conflicts fail with the offending
pattern or tensor; independently pretrained molecular encoders cannot be merged
this way. Branches should descend from the same base.

The downstream predictor is taken from `--head-checkpoint`, which must be an
input checkpoint. It is an initialization for union-pattern training, not a
claim that either specialist's accuracy is preserved by integration alone.
Integration also exports `models/lipids.fragmenter.pft` for preparing union data.
Both output files must be new paths. The integrated checkpoint is self-contained
for prediction and does not load its source checkpoints.

Prepare data with that exported configuration, then train the downstream model
without changing any pattern features:

```bash
python -m fredis.cli train fragment-tree \
  --train-dir data/lipids/train_structures \
  --val-dir data/lipids/validation_structures \
  --output-dir models/lipids_heads --device cuda \
  --fine-tune-checkpoint models/lipids.pt --head-only
```

Workbench **Fine-tune patterns** accepts checkpoints from any stage and includes
**Train prediction heads only**. **Integrate Pattern Models** accepts one input
checkpoint path per line, a downstream-predictor checkpoint and a new output
path, and reports compatibility errors. The adapter-width field is removed.

To reproduce the complete small experimental-data workflow, run:

```bash
PYTHONPATH=. python tests/workflows/verify_pattern_workflow.py \
  --output data/workflow_verification_new
```

It requires CUDA and the local MoNA flavonoid dataset (override `--source` for
its location), and uses four experimental records. The added carbonyl/ether
patterns are workflow probes, not validated fragmentation rules. This checks
preparation, training, parallel specialist branches, sequential expansion,
resume, integration, downstream-only training, exact feature preservation and
strict checkpoint-based prediction. It does not measure predictive accuracy.

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
