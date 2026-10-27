# Train and resume a model

Training consumes prepared `.preft.pt` structures and their chemical configuration.
A fresh run jointly learns MolEncoder, pattern-specific reaction features, branch
selection, and spectrum prediction. Use [fine-tuning](fine-tuning.md) to import
learned reaction patterns into a new dataset.

## Prerequisites

Complete [data preparation](data-preparation.md). Confirm that the training and
validation manifests contain usable sources and inspect excluded/skipped records.
The preparation root must contain its configuration and both split directories:

```text
data/prepared/
  fragment-tree.training-data-config.pft
  train_structures/
  validation_structures/
```

Training inherits reaction definitions, symbols, adduct vocabulary, and selected
pretrained reaction sources from this root. Neural dimensions belong to training;
chemical definitions belong to preparation.

## Start a fresh run

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir data/prepared \
  --output-dir models/first \
  --device cuda --epochs 100 --batch-size 4 \
  --hidden-dim 256 --num-heads 8 --num-layers 3 --ffn-mult 2 \
  --reaction-encoder-layers 2 --reaction-decoder-layers 2 \
  --reaction-instance-max-distance 4 --fc-layers 3 --fc-mult 2 \
  --lr 0.0001 --dropout 0.5
```

Use `--device cpu` for CPU execution. `--batch-size` controls prepared source
batching; `--max-samples` bounds simultaneous spectrum conditions. Several CE
measurements of the same source retain shared chemical and molecular inputs.

The standalone [quickstart](../quickstart.md) supplies a complete small example
using these shared settings. The [configuration reference](../reference/configuration.md)
explains dimensions, defaults, and module ownership.

## Shared architecture

`fragment-tree` sets one hidden width, head count, base depth and FFN multiplier
across the model. It expands these into ordinary detailed CLI arguments and uses
the same training implementation as `fragment-tree-detailed`.

| Option | Applies to |
|---|---|
| `--hidden-dim H` | Molecular atom/readout, pattern atom/reaction output, adduct and CE widths |
| `--num-heads N` | Molecular/pattern attention, reaction encoder/decoder and state SetTransformer |
| `--num-layers L` | Molecular Graphormer, pattern cross attention, spectrum GAT and state SetTransformer |
| `--ffn-mult M` | All these FFNs, including the reaction encoder/decoder |
| `--max-degree`, `--max-spatial-dist`, `--max-edge-dist` | Molecular and new pattern atom encodings |
| `--reaction-encoder-layers`, `--reaction-decoder-layers` | Independent reaction Transformer depths |
| `--reaction-instance-max-distance` | Minimum atom-graph distance between matched reaction substructures |
| `--fc-layers`, `--fc-mult` | CE, Fragment Head and State Head hidden widths |

The one-hop reaction memory GNN remains one layer. The spectrum GAT remains a
single-head sparse attention network. `--num-heads` applies to multihead modules.
The common hidden width must be divisible by the head count. Pattern weights
remain independent. There is no `role_dim` parameter.

FC hidden width at zero-based position `i` is
`H * mult ** min(i, layers - 1 - i)`. For H=256 and mult=2, three widths are
`[256,512,256]`, four are `[256,512,512,256]`, and five are
`[256,512,1024,512,256]`. `--fc-layers 0` selects a direct Linear readout.
The count describes hidden layers; each head also has an output Linear layer.

Reaction-instance distance is the minimum bond distance between the two matched
atom sets in the source molecule. Overlap gives distance 0. Larger or disconnected
distances share an overflow bucket. This differs from `--max-spatial-dist`, which
describes atom pairs. See the [mapping](../reference/configuration.md#shared-training-arguments).

## Configure components independently

Use `fragment-tree-detailed` for independent sizes and architecture documents.
Existing options such as `--node-dim`, `--graph-dim`, `--pattern-*`, and
`--model-architecture` belong to this command. Optimizer, data, validation and
resume controls are available on both commands.


Each dataset pattern has its own internal width, layer count, heads, FFN, and
structural encoding limits. Read the dataset's reaction definitions to determine
its IDs. For a new, trainable pattern with ID 2:

```bash
python -m fredis.cli train fragment-tree-detailed \
  --dataset-dir data/prepared --output-dir models/individual \
  --device cuda --epochs 100 \
  --pattern-hidden-dim 2=64 --pattern-num-heads 2=4 \
  --pattern-num-layers 2=3 --pattern-ffn-mult 2=4 \
  --pattern-max-degree 2=4 \
  --pattern-max-spatial-dist 2=4 --pattern-max-edge-dist 2=4
```

`hidden_dim` must be divisible by `num_heads`. Pattern outputs all project to the
shared reaction width even when their internal widths differ. A pattern's
`max_degree` defaults to **4**; molecular distance limits and reaction-atom
distance limits are separate settings.

Workbench defaults to **Shared settings** for the whole architecture. Select
**Detailed settings** to edit components independently; its reaction-only
**Shared** choice applies one pattern architecture to every new pattern, while
**Individual** edits each separately. Both modes maintain independent weights.
Imported, frozen patterns retain their saved architecture.

## Use an architecture document

```json
{
  "model_config": {
    "mol_encoder_params": {
      "node_dim": 16,
      "graph_dim": 64,
      "num_layers": 2,
      "num_heads": 4
    },
    "reaction_model_params": {
      "hidden_dim": 128,
      "pattern_model_params": {
        "2": {
          "hidden_dim": 64,
          "num_layers": 3,
          "num_heads": 4,
          "ffn_mult": 4,
          "max_degree": 4,
          "max_spatial_dist": 4,
          "max_edge_dist": 4
        }
      }
    }
  }
}
```

Pass it to `fragment-tree-detailed` with `--model-architecture architecture.pft`. Define only pattern IDs
that actually belong to the prepared dataset. Preparation decides imported
pattern ownership; the architecture file does not select pretrained sources.
Resolved settings are written into each checkpoint and training configuration.

The detailed CLI also accepts `--ce-fc-layers/--ce-fc-mult`,
`--fragment-fc-layers/--fragment-fc-mult`, and `--state-fc-layers/--state-fc-mult`.
These generate symmetric widths relative to `--reaction-hidden-dim` and take
precedence over the corresponding `--*-fc-dims`. Raw width lists remain available
for arbitrary architectures.

The shared `--dropout` applies to newly trained networks. Frozen imports keep
their saved values. Optional `--intensity-descriptors TPSA MolLogP` extends the
spectrum model's molecular inputs; preparation does not persist descriptor vectors.

## Objectives and validation

The branch objective uses saved peak/depth alternative paths from BranchTeacher.
It evaluates cumulative log probabilities with softmax-weighted path MIL and a
weak-negative term. The spectrum objective compares formula-aggregated physical
ion probabilities with assigned intensities using cross entropy and cosine loss,
both with and without precursor groups. See
[objectives and gradient flow](reaction-prediction-training.md).

Admission thresholds `--minimum-assignment-score` and
`--minimum-assignment-score-without-precursor` determine which measured spectra
are eligible. Inspect `dataset_summary.json` when fewer records train than were
prepared. Lowering these thresholds changes supervision selection, rather than
chemical preparation.

| Option | When to use it |
|---|---|
| `--validate-before-training` | Inspect an initial epoch-0 baseline before optimization |
| `--validation-interval-steps N` | Run intermediate validation subset checks |
| `--validation-fraction F` | Choose the fraction of validation source molecules for intermediate checks |
| `--train-log-interval-steps N` | Control training metric publication frequency |
| `--checkpoint-interval-epochs N` | Save periodic resumable checkpoints; 0 disables periodic saves |
| `--cache-dataset-in-memory` | Retain loaded prepared tensors in RAM; the default streams them |

Full epoch validation and intermediate validation are distinct. Subsets select
source molecules and keep their eligible spectrum conditions together.

## Resume optimization

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir data/prepared \
  --output-dir models/first \
  --resume models/first/last.pt \
  --epochs 10 --device cuda
```

On resume, `--epochs 10` requests **ten additional epochs**. The optimizer,
scheduler, completed epoch, random states, and saved architecture are restored.
Resume reuses its output directory and does not change reaction ownership or
allow architecture overrides. Creating a new training run with imported
reactions is [fine-tuning](fine-tuning.md), which has a new optimizer and new
trainable downstream modules.

## Inspect the outputs

| Artifact | Use |
|---|---|
| `last.pt` | Resume from the final saved training state or predict with it |
| `best.pt` | Predict with the selected validation checkpoint |
| `fragment-tree.training-config.pft` | Restore the resolved run settings in Workbench |
| `result.training.pft` | Open the training report |
| `metrics/` | Read scalar series and validation distributions |
| `dataset_summary.json` | Inspect accepted and excluded training spectra |
| `trainable_parameters.json` | Verify which networks are trainable or frozen |
| `spectrum_validation/` | Inspect representative measured/generated spectra |

The [file-format reference](../reference/file-formats.md) explains which files
are models, configurations, reports, and prepared tensors. For a fresh run,
existing output requires overwrite authorization. Resume preserves the existing
run. Input paths and protected application paths cannot be overwrite destinations.

Continue with [prediction](prediction.md), [fine-tuning](fine-tuning.md), or
[parameter tuning](tuning.md).
