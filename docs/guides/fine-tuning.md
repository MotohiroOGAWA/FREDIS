# Reuse reaction models and add new patterns

Fine-tuning starts a new run using learned reaction features selected during
**dataset preparation**. Imported local reaction encoders are frozen; newly added
patterns and downstream prediction networks train. MolEncoder remains trainable
under the current fine-tuning policy.

## What is preserved

| Component | Behavior in the new run |
|---|---|
| Imported reaction atom encoder | Architecture and weights inherited; frozen |
| Imported rule/product embeddings and endpoint projections | Inherited and frozen with the complete pattern |
| New reaction pattern | Initialized and trained with its chosen architecture |
| MolEncoder | Initialized from the imported shared encoder; continues training |
| Branch memory encoder, state decoder, and score projections | Newly trainable downstream networks |
| Spectrum GAT, Fragment/State Heads, and set encoder | Newly trainable downstream networks |

The imported reaction encoder consumes source atom/bond features independently
of MolEncoder outputs. Updating MolEncoder therefore does not modify the frozen
primitive reaction representation. New internal pattern widths can differ;
shared input vocabularies and the common output width must remain compatible.

## 1. Select a source checkpoint

Use a trained prediction checkpoint such as `models/base/last.pt`, or export one
complete pattern:

```bash
python -m fredis.cli train export-pattern \
  --checkpoint runs/demo-model/last.pt --pattern-id 0 \
  --output runs/demo-export.reaction-model.pt
```

The exact export arguments appear in the [CLI reference](../reference/cli.md#export-a-reaction-pattern).
To inspect available patterns from Python:

```python
from fredis.ml.training.fragment_tree_training.pattern_models import inspect_model

report = inspect_model("runs/demo-model/last.pt")
for item in report["patterns"]:
    print(item["pattern"]["pattern_id"], item["pattern"]["name"], item["architecture"])
```

A pattern ID is local to the model that stores it. Adding patterns can assign
new IDs in the destination dataset; import matching and weight remapping use the
reaction definitions.

## 2. Prepare a dataset with the imports

For the quickstart dataset, the bundled
{download}`expanded chemistry <../examples/expanded.fragmenter.pft>` adds a
three-atom oxygen-context pattern to the original single-bond definition:

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input runs/demo-input/train.msds \
  --validation-input runs/demo-input/validation.msds \
  --output-dir runs/demo-expanded \
  --params docs/examples/expanded.fragmenter.pft \
  --pretrained-pattern-model runs/demo-model/last.pt \
  --validation-ratio 0.5 \
  --minimum-relative-intensity 0 \
  --max-unique-fragment-smiles 100 \
  --max-reaction-combinations 1000
```

`--pretrained-pattern-model` imports every pattern from the supplied checkpoint
and can be repeated for compatible sources. `--reaction-models-json` allows
individual selections, using records with `pattern`, `source`, and
`source_pattern_id`; new patterns need only `pattern`.

Preparation copies imported dependencies into `pretrained_models/` and records
relative paths. The prepared dataset can then move without depending on the
original checkpoint location. Conflicting shared encoders or duplicate imported
pattern definitions are rejected.

## 3. Start the new training run

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir runs/demo-expanded \
  --output-dir runs/demo-finetuned \
  --hidden-dim 8 --num-heads 2 --num-layers 1 --ffn-mult 2 \
  --reaction-encoder-layers 1 --reaction-decoder-layers 1 \
  --fc-layers 3 --fc-mult 2 \
  --device cpu --epochs 1 --batch-size 1 --max-samples 2 \
  --dropout 0 --warmup-steps 0 \
  --minimum-assignment-score 0 \
  --minimum-assignment-score-without-precursor 0
```

The dataset supplies imported architectures. Shared settings do not override
imported molecular or reaction feature architectures or frozen pattern weights.
New patterns receive the shared architecture and retain independent weights;
trainable Transformer encoder/decoder depths, attention heads and FFNs can differ
from the source checkpoint. The common reaction output width remains inherited
for compatibility with frozen endpoint projections.

For different internal widths per new pattern, use `fragment-tree-detailed`
with `--pattern-* ID=VALUE` or `pattern_model_params` after inspecting destination
IDs. Imported pattern architectures and the shared molecular architecture are
inherited from the dataset.

The added SMARTS pattern is an execution example, rather than a validated
fragmentation mechanism. Use chemically appropriate definitions and held-out
experimental spectra for a scientific model.

## 4. Verify the import

Read `trainable_parameters.json` and the checkpoint's `model_config.fine_tuning`.
The latter records frozen destination IDs and source/destination pattern bindings.
When comparing weights, map IDs through those bindings before comparing Tensor
names. The [workflow verification](pattern-workflow-verification.md) checks exact
preservation of imported pattern weights while MolEncoder updates.

To continue the same optimizer and model, use [resume](training.md#resume-optimization)
with the fine-tuned checkpoint. Resume does not import another pattern set.

## Selecting spectra for trainable patterns

Dataset preparation matches each valid Source against reactant SMARTS of patterns
whose encoders are not imported from pretrained reaction models. At least one
such match is required. Frozen-pattern-only compounds are excluded before any
fragment trees or training tensors are built. When no pretrained models are
selected, all reaction patterns participate in selection.

All metadata-valid conditions of qualifying compounds remain eligible for
preparation. A match does not require a newly assigned peak. Preparation writes
`pattern_filter_summary.json` per split and selection counts in
`reaction_statistics.json`. Source record indices are preserved after filtering.
Prepared files retain the complete reaction set, including imported patterns, so
training can use their fixed features. Training streams the resulting files in
the usual way and does not repeat pattern matching. Recreate older datasets to
apply this selection rule.

Pattern selection precedes molecule-disjoint splitting and validation/test caps.
With a validation/train cap, its denominator is the number of qualifying training
molecules, and validation is sampled only from qualifying held-out molecules.

Dataset creation offers `--pattern-selection trainable` (default) or `all`.
The Workbench calls this **Compound selection** under Fragmentation. `trainable`
requires a match to a new/non-frozen pattern; `all` also admits compounds matching
only imported patterns, for integrated-model training. Both modes discard
compounds matching none of the selected patterns, before splitting and caps.

## Preventing long gaps between reaction patterns

`--rare-pattern-max-gap` (Workbench: **Rare pattern maximum batch gap**, default
`100`) controls online insertion of training batches. After 100 training batches
without a pattern, a batch containing it is inserted before the next ordinary
batch. Ordinary data is still processed in full; this does not wait for the end
of an epoch. Zero disables insertion. If several patterns are overdue, the
longest-absent available pattern is served first, with at most one forced batch
before each ordinary batch. A pattern never present in the training data cannot
be inserted and is marked unavailable.

Occurrences are detected from prepared CPU reaction registries. When a missing
pattern needs a batch, registries are searched lazily, without chemistry or an
upfront full scan. Validation is unchanged. The gap counter and last-seen steps
continue across epochs and are saved in checkpoints. Epoch metrics record
`train/rare_pattern_replay_batches` and `train/replay_pattern_<id>`.
