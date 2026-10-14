# Train and assemble reaction models

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

## Reaction models belong to Training Data preparation

In **Training Data → Fragmentation**, add pretrained reaction models (`.reaction-model.pt` or a complete prediction checkpoint) and new definitions (`.reaction.pft` / `.reactionset.pft`). Imported model definitions are included in the Fragmenter before any reaction enumeration or peak assignment. Lists support per-pattern removal and the same detail component as **Models**.

Preparation specifies only reaction definitions and pretrained source references. In **Training → New model**, new patterns have independent `hidden_dim`, `graph_dim`, `num_layers`, `num_heads`, `dropout`, `neighborhood_hops`, `neighborhood_hidden_dim`, and `descriptors`. Imported architectures remain read-only, and Resume locks every architecture setting. `hidden_dim` controls Graphormer attention; raw atom feature width comes from the chemical feature schema and is not configurable per pattern. Atom capacity is derived from SMARTS.

CLI can add every pattern from a checkpoint during preparation:

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input spectra.msds --output-dir data/prepared \
  --params chemistry.fragmenter.pft \
  --pretrained-pattern-model models/base/last.pt
```

The repeatable `--pretrained-pattern-model` also accepts a single `.reaction-model.pt`. For selected imports and new chemical definitions, `--reaction-models-json` accepts a JSON array:

```json
[
  {
    "pattern": {"name": "carbon", "reactant_smarts": "[#6:1]-[#6:2]", "products": [{"name": "left", "smarts": "[*:1]"}]},
    "source": "models/carbon.reaction-model.pt",
    "source_pattern_id": 0
  },
  {
    "pattern": {"name": "ether", "reactant_smarts": "[#6:1]-[#8:2]-[#6:3]", "products": [{"name": "oxygen", "smarts": "[*:2]"}]}
  }
]
```

Training accepts `--model-architecture architecture.json` for per-pattern model settings. IDs below refer to the prepared dataset, not to imported source checkpoints:

```json
{
  "model_config": {
    "mol_encoder_params": {"node_dim": 16, "graph_dim": 64, "num_layers": 2, "num_heads": 4},
    "reaction_model_params": {
      "pattern_input_version": 2,
      "pattern_model_params": {
        "2": {"hidden_dim": 64, "graph_dim": 128, "num_layers": 3, "num_heads": 4, "dropout": 0, "neighborhood_hops": 3, "neighborhood_hidden_dim": 32, "descriptors": []}
      }
    }
  }
}
```

A SMARTS atom input is `[raw chemical features | external-neighborhood feature | fixed SMARTS-position one-hot]`, projected to `hidden_dim` before Graphormer attention. Neighborhood features use their own learnable `neighborhood_hidden_dim`. Propagation moves through external bonds from radius k toward the connected SMARTS atom and includes shells 1..k. Other SMARTS atoms are blocked while collecting that role's external structure, so a substituent on one role does not populate another role through the SMARTS backbone. Roles with no external neighbors receive exact zeros. `neighborhood_hops=0` disables this channel; the prepared full source graph permits radii beyond three without changing reaction chemistry or re-preparing data.

Existing pretrained checkpoints retain their legacy input/position encoding. Fresh training and new pattern settings use `pattern_input_version=2`, and checkpoints record each pattern's input mode. Imported modules are not silently converted. Pattern-owned neighborhood weights are exported, initialized, frozen and integrated with their Graphormer/readout.

The source ID belongs to the imported checkpoint. IDs in the prepared dataset are reassigned from chemical definitions. Imported architectures are restored from their source rather than overridden. The usual `train fragment-tree --train-dir ... --val-dir ...` then automatically freezes imported features and trains new pattern modules and downstream predictors.

Dataset preparation copies every selected pretrained checkpoint or `.reaction-model.pt` into the output's `pretrained_models/` directory. SHA-256 names deduplicate repeated references to the same file contents, and `manifest.json` records each copy's size and checksum. Configuration files store paths relative to their own location; train and validation splits share the copies in their dataset root. Copy or move the entire output directory, including `pretrained_models/`, to use it elsewhere. Training resolves the local copies automatically and never needs the original pretrained files. Workbench also resolves these relative paths when restoring preparation settings. Direct Python preparation calls bundle their dependencies in the same way.

A reaction file includes complete encoder/readout weights and shared source-feature dependencies. Imports require compatible shared source weights and settings. Prediction checkpoints embed all weights; prediction and Resume do not require the original import files. A prediction-only integrated checkpoint is selected as a pretrained source while preparing another dataset; it cannot Resume an optimizer it does not contain.

Models exports individual patterns with:

```bash
python -m fredis.cli train export-pattern \
  --checkpoint models/base/last.pt --pattern-id 0 \
  --output models/bond.reaction-model.pt
```

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

Workbench uses **New model** and **Resume training**. Add checkpoints or single reaction models under **Training Data → Fragmentation**, then select those prepared datasets in New model. Imported patterns are frozen; patterns without imported models are trained from scratch. Import an integrated checkpoint to train its prediction modules with every reaction feature frozen. **Integrate Pattern Models** combines compatible branches and selects their downstream predictor.

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

Each check emits an `intermediate_validation_end` JSON event with step, sample count, loss components, and reaction/spectrum metrics. The training progress section shows the latest result and a loss chart against optimizer steps. Results are saved in `intermediate_validation.json` and `intermediate_validation.tsv`; `intermediate_validation_subset.json` records the selected sample indices. TensorBoard step metrics are stored under `tensorboard/iterations` and are included when opening the parent `tensorboard` directory.

Full validation still runs at each epoch end. Learning-rate scheduling, early stopping, and best-checkpoint selection use full-validation results; intermediate checks are for monitoring.

### Global structure for an individual reaction pattern

New reaction architectures may enable a pattern-owned molecular GNN and separate
cross-attention through `reaction_model_params.pattern_model_params[pattern_id]`:

```json
{
  "hidden_dim": 64,
  "graph_dim": 128,
  "num_layers": 2,
  "num_heads": 4,
  "neighborhood_hops": 3,
  "neighborhood_hidden_dim": 32,
  "global_num_layers": 3,
  "global_hidden_dim": 64,
  "global_num_heads": 4
}
```

`global_num_layers: 0` (the default) disables this mechanism and allocates no
additional pattern parameters. The GNN consumes chemical atom features and bond
features from the complete original source molecule. Ordered SMARTS atoms supply
queries; all atoms of that same molecule supply keys and values. Learned shortest
bond-distance embeddings relative to each queried atom distinguish distant bond
positions. Distances 0–64 have separate embeddings; larger distances and disconnected
atoms share the overflow embedding. Attention returns one vector of
`global_hidden_dim` per SMARTS atom, concatenated with its chemistry and optional
external-neighborhood features before the fixed-capacity reactant Graphormer.
Context atoms do not join its self-attention token sequence.

Workbench exposes these settings per new pattern in Training. Imported architectures
and Resume remain fixed. Exports and integrations include the GNN, cross-attention,
and distance embeddings as part of that complete pattern model.

Independent branches share a frozen pretrained feature model. Define different
SMARTS for newly trained specialists, import the same base into each dataset,
train each branch, and use `train integrate-patterns` to collect their pattern
models. The integrated fragmenter generates matching reactions and the encoder
dispatches each reaction to its pattern ID. Matching general and specialist patterns
may both produce reactions; this does not replace a general pattern or arbitrate
between overlapping definitions. Duplicate chemical definitions with conflicting
weights are rejected. After integration, prepare a union dataset from the integrated
checkpoint and train the downstream predictors while all imported pattern features
remain frozen.
