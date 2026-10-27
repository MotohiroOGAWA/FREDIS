# Quickstart: prepare, train, and predict

This example runs the complete pipeline on a small CPU model. It creates four
spectra for two molecules, prepares training supervision, trains one epoch, and
predicts spectra from the saved checkpoint. Run the commands from the repository
root after completing [installation](guides/installation.md).

The demonstration peaks are **synthetic**. They exercise the workflow and should
not be used to assess prediction accuracy or train a scientific model.

## 1. Create the example spectra

```bash
python docs/examples/create_demo_dataset.py --output runs/demo-input
```

Expected summary:

```text
train.msds: 2 spectra, 6 peaks
validation.msds: 2 spectra, 6 peaks
```

Training contains ethanol (`CCO`) at 20 and 40 eV; validation contains `CC(N)O`
at 10 and 30 eV. Every record has a unique `SpecID`, `SMILES`, `AdductType`,
`CollisionEnergy`, `PrecursorMZ`, and a peak list. The two splits have different
source molecules.

The input creator is included as
{download}`create_demo_dataset.py <examples/create_demo_dataset.py>`. To inspect
one file:

```python
from fredis.data.core.MSDataset import MSDataset

dataset = MSDataset.load("runs/demo-input/train.msds")
print(dataset.metadata[["SpecID", "SMILES", "CollisionEnergy"]].to_string(index=False))
print(dataset.peaks[0].data)
```

## 2. Prepare chemical structures and supervision

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input runs/demo-input/train.msds \
  --validation-input runs/demo-input/validation.msds \
  --output-dir runs/demo-prepared \
  --params docs/examples/chemistry.fragmenter.pft \
  --validation-ratio 0.5 \
  --minimum-relative-intensity 0 \
  --max-unique-fragment-smiles 100 \
  --max-reaction-combinations 1000
```

The bundled {download}`chemistry configuration <examples/chemistry.fragmenter.pft>`
uses a two-atom single-bond reaction and a maximum reaction depth of two.
Preparation enumerates compatible source reactions, materializes structures,
assigns measured peaks to formula explanations, and saves branch/spectrum teachers.
It does not initialize neural-network weights.

The output contains:

```text
runs/demo-prepared/
  fragment-tree.training-data-config.pft
  fragment-tree.fragmenter.pft
  split_manifest.json
  invalid_records.json
  train_structures/
    data/*.preft.pt
    manifest.tsv
  validation_structures/
    data/*.preft.pt
    manifest.tsv
```

Inspect `manifest.tsv` for prepared/skipped sources before training. The complete
artifact descriptions are in [file formats](reference/file-formats.md).

## 3. Train one epoch

```bash
python -m fredis.cli train fragment-tree \
  --dataset-dir runs/demo-prepared \
  --output-dir runs/demo-model \
  --hidden-dim 8 --num-heads 2 --num-layers 1 --ffn-mult 2 \
  --reaction-encoder-layers 1 --reaction-decoder-layers 1 \
  --fc-layers 3 --fc-mult 2 \
  --device cpu --epochs 1 --batch-size 1 --max-samples 2 \
  --dropout 0 --warmup-steps 0 \
  --minimum-assignment-score 0 \
  --minimum-assignment-score-without-precursor 0
```

This command uses width 8 throughout, two attention heads and FC hidden widths
`[8,16,8]`. The same architecture is available as a
{download}`detailed architecture document <examples/compact.architecture.pft>`
for `fragment-tree-detailed --model-architecture`. The prepared dataset supplies the chemical definitions.
For a CUDA environment, use `--device cuda` with the same commands.

A successful run writes `last.pt`, `best.pt`, a training configuration, metrics,
and `result.training.pft`. One epoch verifies execution; its predictions are not
a validated scientific result.

## 4. Predict a single spectrum

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --model runs/demo-model/last.pt \
  --smiles CCO --ce 20 --adduct-type '[M+H]+' \
  --output-dir runs/demo-single \
  --device cpu --strict
```

The output is `runs/demo-single/predicted.msds`, with associated run information
and similarity output. Direct prediction creates one input condition from the
arguments; no measured spectrum is required.

## 5. Predict the input dataset in batches

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --model runs/demo-model/last.pt \
  --input runs/demo-input/train.msds \
  --output-dir runs/demo-batch \
  --device cpu --strict --max-samples 2
```

```python
import numpy as np
from fredis.data.core.MSDataset import MSDataset

single = MSDataset.load("runs/demo-single/predicted.msds")
batch = MSDataset.load("runs/demo-batch/predicted.msds")
assert len(single) == 1
assert len(batch) == 2
assert batch.n_peaks_total > 0
np.testing.assert_allclose(single.peaks[0].data, batch.peaks[0].data, rtol=1e-4, atol=1e-5)
print("Single and batch predictions agree.")
```

Open `result.training.pft` or `predicted.msds` in Workbench to inspect the result.
Continue with [measured-data preparation](guides/data-preparation.md),
[importing reaction models](guides/fine-tuning.md), or the
[Python prediction API](api/prediction.md).
