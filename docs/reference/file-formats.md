# Files, checkpoints, and reports

A `.pft` suffix identifies a FREDIS editor/document convention, rather than one
binary encoding. Most configuration/report `.pft` documents are JSON; reaction
pattern analysis uses an HDF5 result with the same suffix. `.preft.pt` files are
PyTorch-serialized prepared structures.

## Artifact map

| Artifact | Storage | Purpose / owner |
|---|---|---|
| `*.msds` | msentity native dataset | Spectrum metadata and peak arrays |
| `*.mssim` | msentity similarity result | Similarity/comparison output |
| `*.fragmenter.pft` | JSON | Reaction patterns, ion rules, chemical search parameters |
| `*.adductset.pft`, `*.adduct.pft` | JSON | Fragment ion rule set or one rule |
| `*.architecture.pft` | JSON | `model_config` neural architecture |
| `fragment-tree.training-data-config.pft` | JSON | Resolved preparation inputs, mapping, and chemistry |
| `fragment-tree.training-data.pft` | JSON | One split's prepared dataset result/manifest document |
| `*.preft.pt` | PyTorch | A validated SourceReactionStructure and its supervision |
| `last.pt`, `best.pt` | PyTorch checkpoint | Model weights and saved model configuration |
| `*.reaction-model.pt` | PyTorch | Complete portable reaction pattern feature asset |
| `fragment-tree.training-config.pft` | JSON | Resolved training settings for Workbench |
| `result.training.pft` | JSON | Training result report and metric locations |
| `result.tuning.pft` | JSON | Trial settings, observations and statuses |
| `prediction.pft` | JSON | Prediction settings and provenance |
| `result.reactionpatternanalysis.pft` | HDF5 | Analysis-wide summaries and indexes |
| `compounds/*.peaks.parquet` | Parquet | Reaction-analysis peak summaries |

Use the workflow's loader/editor for the artifact's contents. Renaming a report
or configuration does not turn it into a checkpoint.

## Prepared structures

A `.preft.pt` payload records schema `fredis.annotated-spectrum-training` and
its `structure`. It contains shared chemistry, observed spectra, assignments,
BranchTeacher memberships/history graph, and spectrum formula targets. Files
are saved on CPU. Loading validates the stored data without repeating RDKit
preparation or reconstructing the teacher.

```python
from fredis.ml.input.source_reaction_structure import SourceReactionStructure

structure = SourceReactionStructure.load("data/prepared/train_structures/data/source.preft.pt")
structure.validate()
print(structure.source_smiles, structure.num_samples)
```

Geometry and `spectrum_batch` are runtime caches. They do not become extra stored
chemical tables. See the [input schema](input-schema.md) for field ownership and
batch-local index spaces.

## Checkpoint contents

Training checkpoints carry `architecture`, `model_config`, `model_state_dict`,
completed epoch/global step, optimizer and scheduler states, random states,
and training histories. `last.pt` is the final resumable state; `best.pt` is the
selected validation state. A portable exported reaction asset is not necessarily
a resumable optimizer checkpoint.

Use `load_generator(..., strict=True)` to load a checkpoint for prediction and
`--resume` to continue a resumable training run. Raw state dictionaries require
an explicit compatible parameter file. The recorded `architecture` is
`fragment-tree-competitive-ion` for the current generator.

## Dataset imports and relocation

Preparation copies selected pretrained dependencies under `pretrained_models/`.
Configuration paths are resolved relative to the saved document. Keep this
directory with the preparation root when moving a dataset. Pattern IDs can
change as definitions are combined; import bindings record old and new IDs.

## Results and diagnostic tables

Preparation's `manifest.tsv`, `assignment_scores.tsv`, `invalid_records.json`,
and `skipped_sources.json` distinguish prepared sources, weak assignment, invalid
records, and per-source failures. `reaction_statistics.json` summarizes search
cost and retained depth.

Training's `metrics/` catalog stores scalar time series and validation
distributions. `spectrum_validation/` contains representative measured/generated
spectra. `trainable_parameters.json` exposes parameter counts and freezing.
These diagnostics do not replace the model checkpoint.

Prediction's `run/failures.tsv` identifies records or molecules that failed.
`predicted.msds` can contain valid records from a partially successful run, so
inspect the failure table together with record counts.

Reaction-analysis results follow a separate three-level layout, described in
[Reaction Pattern Analysis](../guides/reaction-pattern-analysis.md).
