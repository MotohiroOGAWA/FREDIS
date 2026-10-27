# Predict and inspect spectra

Prediction requires a trained checkpoint, molecular SMILES, a supported precursor
adduct, and collision energy. Single prediction and batch dataset prediction use
the same generator. Model vocabulary and chemical rules come from the checkpoint.

## Predict one condition

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --model models/first/best.pt \
  --smiles CCO --ce 20 --adduct-type '[M+H]+' \
  --output-dir predictions/ethanol \
  --device cuda --strict
```

Without `--input`, the CLI creates input records from `--smiles`, `--ce`, and
`--adduct-type`. Precursor m/z is derived from the requested chemistry. The
installed `fredis-predict-spectrum` and `msentity-predict-fredis` commands are
equivalent to the module command.

## Predict an MSDataset

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --model models/first/best.pt \
  --input spectra.msds \
  --output-dir predictions/dataset \
  --device cuda --strict \
  --max-samples 32 --num-workers 4 --chunk-size 1
```

The required default metadata columns are:

| Column | Meaning |
|---|---|
| `SpecID` | Unique, nonempty input record identifier |
| `SMILES` | Molecular structure |
| `PrecursorMZ` | Recorded positive precursor m/z |
| `AdductType` | Requested precursor adduct |
| `CollisionEnergy` | Parseable collision energy |

Use `--smiles-column`, `--spec-id-column`, `--adduct-type-column`,
`--collision-energy-column`, and `--precursor-mz-column` for different names.
`--instrument-column` supplies instrument metadata when needed for supported
energy parsing. A numeric CE value is interpreted by the input adapter; the
model receives eV features. Model normalization does not itself convert units.

Prediction can consume a metadata-only input dataset; measured peaks are not
needed to generate a spectrum. Training preparation instead requires measured
peak lists. Input validation and failed records are reported in the run outputs.

## Batching and chemical preparation

The checkpoint stays loaded while independent molecule groups are processed in
bounded chunks. Repeated SMILES can share chemical preparation and static reaction
encodings. CE conditions remain separate predicted spectra. `--max-samples`
bounds simultaneous conditions; `--num-workers` and `--chunk-size` control
CPU subprocess preparation, rather than neural attention layers.

A failed multi-molecule chunk is retried per molecule so valid records can still
be produced. Inspect the failure report to distinguish a complete prediction
from a partial result.

## Selection and output thresholds

| Option | Meaning |
|---|---|
| `--branch-path-threshold` | Prune histories by cumulative branch probability |
| `--max-fragment-nodes` | Bound unique retained structures per source/main-adduct group |
| `--peak-intensity-threshold` | Remove output peaks at or below the requested relative intensity |
| `--no-formula-annotation` | Omit formula strings in peak metadata |
| `--strict` | Require checkpoint parameter names/shapes to match |
| `--overwrite` | Allow replacing the selected prediction output |

The fragment budget constrains structural search. It is independent of the
number of ion/hydrogen-shift candidates. See the
[architecture guide](../concepts/architecture.md) for their probability flow.

## Output files and provenance

```text
predictions/dataset/
  predicted.msds
  predicted.mssim
  prediction.pft
  run/
    run.json
    failures.tsv
```

`predicted.msds` contains generated peaks and retained input metadata.
`predicted.mssim` contains associated comparison information where available.
`prediction.pft` and the run directory record execution settings and outcomes.

| Metadata | Meaning |
|---|---|
| `SpecID` | Generated prediction identifier |
| `SourceSpecID` | Immediate input record identifier |
| `PredictionTool`, `PredictionToolVersion` | Tool and package version |
| `PredictionModel`, `PredictionTimestamp` | Model and execution provenance |
| `PredictedPrecursorMZ` | Precursor m/z used by prediction |
| `PredictionCollisionEnergy`, `PredictionCollisionEnergyUnit` | Resolved energy condition |
| `PredictionAdductType` | Resolved prediction adduct |

Input `PrecursorMZ`, `ExactMass`, and collision-energy metadata are retained.
Use the `Prediction*` fields to inspect the actual resolved prediction condition.
Existing outputs require `--overwrite`; publication uses an atomic save.

## Inspect from Python

```python
from fredis.data.core.MSDataset import MSDataset

predicted = MSDataset.load("predictions/dataset/predicted.msds")
print(len(predicted), predicted.n_peaks_total)
print(predicted.metadata[["SpecID", "SourceSpecID", "PredictionAdductType"]].head())
print(predicted.peaks[0].data)
```

The [Python API](../api/prediction.md) exposes generator loading, chunked outputs,
input-order mapping, and dataset prediction. Workbench's **Prediction** pages
provide single and batch forms; spectrum and reaction viewers expose the
resulting peaks and structures.
