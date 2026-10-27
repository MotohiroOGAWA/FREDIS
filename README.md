# FREDIS(Fragmentation with Reactions that are Externally Definable for In silico Spectra)

**Fragmentation with Reactions that are Externally Definable for In Silico Spectra**

FREDIS models mass-spectrometry fragmentation as externally defined reaction
sequences. The Python package and command-line entry point are both named
`fredis`.

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)
![RDKit](https://img.shields.io/badge/RDKit-2025.03.6-green)

---

## Installation and separated msentity packages

The snapshots in `upstream/` are pinned in `upstream/versions.json` and contain
no Git metadata. The viewer is maintained in the separate Workbench repository.

Install CMake, a C++20 compiler, HDF5 development files, `nlohmann-json3-dev`,
and `pyarrow>=20`, then run:

```bash
python scripts/build-native.py                    # dataset library only
python -m pip install -e .
```

Spectrum prediction also writes an embedded `.mssim` comparison with the source
spectra, so that operation needs the separate similarity library:

```bash
python scripts/build-native.py --similarity
python -m pip install -e '.[similarity]'
```

The native helper uses Arrow/Parquet from pyarrow. Extra CMake options may be
passed after `--`. The `fredis.data` compatibility model retains the existing
mutable pandas/NumPy training interfaces; native msentity handles file I/O and
msentity-similarity handles prediction scoring. Ordinary data preparation and
fragmentation do not import similarity.

## Reaction Pattern Analysis

`python -m fredis.cli analysis reaction-pattern` (Workbench: **Data → Reaction Pattern
Analysis**) measures peak assignment along a user-defined tree of reaction
pattern sets. Each node analyses the union of the sets on its path
(`A → B → C` gives `A`, `A + B`, `A + B + C`); each edge compares the parent
condition with the parent condition + the child set, and roots report absolute
coverage. Spectra are grouped into unique compounds across all datasets, Source
reactions are generated once per compound and pattern, child conditions are
built incrementally from their parent, and an edge is calculated only when the
**original precursor SMILES** matches a pattern the child set adds. Nodes can be
added to an existing result (`--extend 1`) without recalculating it.

The result is read A → B → C:

```text
A: result.reactionpatternanalysis.pft
   standard HDF5
   analysis-wide summaries / distributions / indexes

B: compounds/*.peaks.parquet
   standard Parquet
   one compound / all spectra / peak-level binary summary

C: details/**/*.preft.pt
   detailed fragment/reaction/assignment data
```

Workbench opens `result.reactionpatternanalysis.pft` directly (click, Open With, or
drag-and-drop), reads only A at first, then the B file of a selected compound
and the C files of a selected peak. See
[`docs/guides/reaction-pattern-analysis.md`](docs/guides/reaction-pattern-analysis.md).

---

## Branch selection and spectrum intensity

Branch selection uses the directed-subgraph Transformer encoder/state decoder
and the existing path MIL objective. Spectrum intensity uses a separate sparse
GAT over the known fragment structures: Compound graph features are nodes and
local reaction features are edges. Root structure, main adduct and collision
energy condition the GAT. Message passing runs in both directions with explicit
direction features; predicted probability flows only from parent to child.

For every reaction history, Child, Stay and Sink logits are normalized together
with a log softmax over its outgoing events. The ion/shift State Head additionally receives a SetTransformer representation
of each applied local reaction set. Reaction orders with the same set share
one state distribution and sum their retained probabilities. Distinct sets
are scored separately, then their joint ion probabilities are summed. Log-space probability propagation, formula
aggregation and spectrum losses are unchanged.
Spectrum scoring does not consume branch Transformer memory or decoder outputs.

Canonical-SMILES-identical Compounds share one structure entry and molecular
encoding. GAT nodes and physical reaction edges are deduplicated within each
spectrum, while reaction histories retain their distinct valid next reactions.
Contributions from those histories merge into the same physical fragment.
Different measurement conditions remain separate GAT graphs.

Configure `spectrum_model_params.gat_num_layers` (default `2`) or training CLI
`--spectrum-gat-layers`. Workbench exposes this under **Fragment Graph Attention**.
See [branch and spectrum training](docs/guides/reaction-prediction-training.md).

## Batch MS/MS spectrum prediction

Predict every compatible record in an MSDataset with one model load and bounded
unique-SMILES chunks:

```bash
msentity-predict-fredis \
  --input source.msds \
  --output-dir source_fredis \
  --model model.pt \
  --db MoNA \
  --max-samples 32 \
  --device cuda
```

The input must provide unique, non-empty `SpecID`, `SMILES`, `PrecursorMZ`,
`AdductType`, and `CollisionEnergy` values (column names are configurable).
Output records retain the input metadata, move the immediate input identifier to
`SourceSpecID`, and receive `fredis-<DB>-<sequence>` identifiers. Provenance and
prediction conditions are recorded in `PredictionTool`, `PredictionToolVersion`,
`PredictionModel`, `PredictionTimestamp`, `PredictedPrecursorMZ`,
`PredictionCollisionEnergy`, `PredictionCollisionEnergyUnit`, and
`PredictionAdductType`. Input `PrecursorMZ`, `ExactMass`, and collision-energy
metadata are not overwritten. Dataset-level attributes and tags also identify
the generated spectra.

Prediction structures are processed in chunks while the checkpoint remains
loaded. A failed multi-SMILES chunk is retried per molecule so valid records can
still be saved. The output directory contains `predicted.msds`,
`predicted.mssim`, and `prediction.pft`. `run/run.json` records the execution and
`run/failures.tsv` contains record-level failures. Existing output is
replaced only with `--overwrite`; output publication uses an atomic rename.

The same operation is available in **Workbench → Predict Spectrum → Batch
MSDataset prediction**. `fredis-predict-spectrum` is an equivalent command name.



---

## Documentation

Start with the [documentation home](docs/index.md) or the runnable
[quickstart](docs/quickstart.md). The guide covers preparation, training/resume,
reaction-model fine-tuning, single/batch prediction, Workbench, architecture,
input schemas, CLI options, and Python APIs.

Build with `python -m pip install -e '.[docs]'` and
`python -m sphinx -W --keep-going -b html docs docs/_build/html`.
Run the published examples with `python docs/examples/verify_examples.py`.
See [documentation development](docs/development.md) for checks. The optional
FREDIS Workbench GUI is developed, tested and packaged in a separate repository;
this repository contains the Python backend and CLI.

## Python code style

Install the development tools with `pip install -e ".[dev]"`. Format the FREDIS
package with `python -m black fredis`, or verify its formatting with
`python -m black --check fredis`. The shared configuration in `pyproject.toml`
excludes the third-party repositories under `fredis/libs`.

Keep one operation per statement, name tensor indices after the objects they
reference, and split long workflows at domain boundaries such as feature
encoding, transition enumeration, probability propagation and loss calculation.
Document tensor ownership and indexing where these are not apparent from types.
