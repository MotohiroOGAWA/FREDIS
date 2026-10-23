# Train and assemble reaction models

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir data/prepared \
  --output-dir models/my-model \
  --node-dim 32 --graph-dim 128 \
  --epochs 100 --batch-size 4 --lr 0.0001
```

The command jointly trains a newly initialized Mol Encoder and the
fragment-tree model.

Each run writes `result.training.pft` for metrics/results and `fragment-tree.training-config.pft` for the actual training settings. Separate `training_args.json` and `training_config.json` files are not written. Open the result file to inspect metrics; open the configuration file to restore the Training Workbench page. Workbench **Save Configuration** also uses `.training-config.pft`. Relative dataset, output and checkpoint paths in a configuration are resolved from that file's directory.

Prepared compound batches retain canonical SMILES metadata across serialization and device transfers. Previously, the training input's display adapter rebuilt every compound's SMILES on each node lookup. The cached identities and one-list-per-display lookup avoid that quadratic RDKit work; existing prepared files compute the missing identity cache once on loading and need no regeneration.

Descriptors are optional inputs to post-branch intensity prediction. Set
`--intensity-descriptors TPSA MolLogP` (or its alias `--descriptors`). The
selected descriptors are calculated for each materialized fragment and appended
only to the intensity-head input. They do not enter reaction features, branch
selection, fragment classification or ion-state classification. The default is
an empty list; Workbench exposes the selection under Intensity Head. Descriptor
values are already stored in prepared graphs, so selecting them does not require
new data preparation.

In **Workbench → Fragment Tree Training**, **Load Configuration** and **Save Configuration** are in the page header; the load control also accepts a dropped `*.pfttrain.json` file. Saved training configurations contain run settings and the trainable Mol Encoder / Branch / Fragment Transformer / Ion-State / Intensity parameters. Fragmenter settings, Reaction Patterns, ion-adduct rules, Symbols, and adduct ordering are inherited from the prepared train/validation datasets.

After selecting the preparation output directory, Workbench checks that their Fragmenter settings, adduct ordering, and Symbols agree. Reaction Pattern Sets and Ion Adduct Rule Sets are shown read-only, while Mol Encoder architecture parameters are editable for joint training.

## Resume a checkpoint

Use `--resume models/base/last.pt` together with `--dataset-dir` to continue an interrupted run with its optimizer state. Reaction definitions and pretrained sources are selected only during dataset preparation.

## Reaction models belong to Training Data preparation

In **Training Data → Fragmentation**, add pretrained reaction models (`.reaction-model.pt` or a complete prediction checkpoint) and new definitions (`.reaction.pft` / `.reactionset.pft`). Imported model definitions are included in the Fragmenter before any reaction enumeration or peak assignment. Lists support per-pattern removal and the same detail component as **Models**.

Preparation specifies only reaction definitions and pretrained source references. In **Training → New model**, the default **Shared** reaction settings apply `hidden_dim`, `graph_dim`, `num_layers`, `num_heads`, `neighborhood_hops`, and `neighborhood_hidden_dim` to every new reaction. Switch **Reaction parameter settings → Individual** to edit each reaction separately. Pretrained reactions are excluded from both modes. Mol Encoder and prediction heads keep their own architecture controls. Mol Encoder external feature width is input metadata, not an editable or tunable architecture parameter. Workbench currently supplies no external Mol Encoder features (width 0); its selected descriptors are consumed by the post-branch prediction heads. In tuning mode, enable **Tune** beside a shared reaction parameter to vary every new reaction together in one search dimension; Individual mode tunes each selected reaction independently. The selected mode, fixed values, and candidate lists are saved in the training configuration. The single **dropout** in Training Configuration (`--dropout`, default 0.5) applies to all newly trained neural networks: Mol Encoder, reaction Graphormers, external neighborhood encoders, reaction readout, branch scoring, Fragment Head, and State Head. CE layers, Adduct embeddings, and Adduct feature projections have no dropout. Network constructor defaults are 0.0; CLI and Workbench supply the shared training default of 0.5. This value is passed explicitly to each network constructor; there is no mutation of dropout rates after construction. Imported pretrained Mol Encoder and reaction features remain frozen with their saved settings. Dropout is disabled during evaluation and prediction. Imported architectures remain read-only, and Resume locks every architecture setting. `hidden_dim` controls Graphormer attention; raw atom feature width comes from the chemical feature schema and is not configurable per pattern. Atom capacity is derived from SMARTS.

CLI training uses explicit reaction IDs for every reaction architecture parameter. Repeat each option for additional new reactions; pretrained IDs, unknown IDs, duplicate assignments, and architecture changes on Resume are rejected. For example, configure reaction ID 2:

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir prepared --output-dir trained \
  --pattern-hidden-dim 2=64 \
  --pattern-graph-dim 2=128 \
  --pattern-num-layers 2=3 \
  --pattern-num-heads 2=4 \
  --pattern-neighborhood-hops 2=3 \
  --pattern-neighborhood-hidden-dim 2=32
```

Workbench expands its Shared settings into these individual CLI options before starting training or a tuning trial.

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
        "2": {"hidden_dim": 64, "graph_dim": 128, "num_layers": 3, "num_heads": 4, "neighborhood_hops": 3, "neighborhood_hidden_dim": 32}
      }
    }
  }
}
```

A SMARTS atom input is `[raw chemical features | external-neighborhood feature | fixed SMARTS-position one-hot]`, projected to `hidden_dim` before Graphormer attention. Neighborhood features use their own learnable `neighborhood_hidden_dim`. Messages consume bond-type/ring features as well as atom hidden states. Propagation moves through external bonds from radius k toward the connected SMARTS atom and includes shells 1..k. Other SMARTS atoms are blocked while collecting that role's external structure, so a substituent on one role does not populate another role through the SMARTS backbone. Roles with no external neighbors receive exact zeros. `neighborhood_hops=0` disables this channel; the prepared full source graph permits radii beyond three without changing reaction chemistry or re-preparing data.

Existing pretrained checkpoints retain their legacy input/position encoding. Fresh training and new pattern settings use `pattern_input_version=3`, and checkpoints record each pattern's input mode. Imported modules are not silently converted. Pattern-owned neighborhood weights are exported, initialized, frozen and integrated with their Graphormer/readout.

The source ID belongs to the imported checkpoint. IDs in the prepared dataset are reassigned from chemical definitions. Imported architectures are restored from their source rather than overridden. The usual `train fragment-tree --dataset-dir ...` then automatically freezes imported features and trains new pattern modules and downstream predictors.

Dataset preparation copies every selected pretrained checkpoint or `.reaction-model.pt` into the output's `pretrained_models/` directory. SHA-256 names deduplicate repeated references to the same file contents, and `manifest.json` records each copy's size and checksum. Configuration files store paths relative to their own location; train and validation splits share the copies in their dataset root. Copy or move the entire output directory, including `pretrained_models/`, to use it elsewhere. Training resolves the local copies automatically and never needs the original pretrained files. Workbench also resolves these relative paths when restoring preparation settings. Direct Python preparation calls bundle their dependencies in the same way.

A reaction file includes complete encoder/readout weights and shared source-feature dependencies. Imports require compatible shared source weights and settings. Prediction checkpoints embed all weights; prediction and Resume do not require the original import files. A prediction-only integrated checkpoint is selected as a pretrained source while preparing another dataset; it cannot Resume an optimizer it does not contain.

Models exports individual patterns with:

```bash
python -m fredis.cli train export-pattern \
  --checkpoint models/base/last.pt --pattern-id 0 \
  --output models/bond.reaction-model.pt
```

## Use pretrained reactions when learning new reactions

Select all pretrained checkpoints and new reaction definitions during data preparation. Preparation checks compatibility and archives the checkpoints in `pretrained_models/`. Start training with `--dataset-dir` pointing to that preparation output. Imported reaction features and their shared source encoders stay frozen; new reaction features and downstream prediction heads are trained. Training cannot add reactions or select another pretrained source. Workbench provides the same workflow through one **Prepared dataset directory** field.

The root contains `fragment-tree.training-data-config.pft`, the standalone `fragment-tree.fragmenter.pft`, `split_manifest.json`, input-validation `invalid_records.json`, and optional `pretrained_models/`. It does not duplicate the shared configuration in `preparation_config.json`, or create a root `fragment-tree.training-data.pft`. Each split retains its own `fragment-tree.training-data.pft`, `invalid_records.json`, manifests, scores, statistics, skipped-source report and `data/` tensors. The root invalid-record report preserves failures detected before splitting; it is distinct from split-local validation results.

## Monitoring

Training writes `last.pt`, `metrics.json` and `metrics.tsv` and emits `epoch_end` JSON Lines. Workbench shows epoch progress, train/validation losses, reaction metrics and bounded searchable stdout/stderr logs. Training processes continue after their panel closes. Restored job metadata distinguishes a missing process from a known successful exit.

### Intermediate validation

For large training datasets, evaluate a validation subset between epoch boundaries:

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir data/prepared \
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

To combine reactions learned in compatible branches, select their checkpoints together during preparation. Shared encoders and overlapping reaction features must have identical weights. Then train from the resulting preparation output directory.

New version 3 pattern models enable `neighborhood_edge_features=true`. Version 1/2
imports retain their original connectivity-only neighborhood weights and any
saved descriptor-dependent source/readout modules; they are not silently
converted. New patterns reject descriptor inputs. New source MolEncoders have no
selected descriptors. `spectrum_model_params.descriptors` is the only descriptor
selection for newly trained intensity models, and remains fixed on Resume.
