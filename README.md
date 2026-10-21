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

## Batch MS/MS spectrum prediction

Predict every compatible record in an MSDataset with one model load and bounded
unique-SMILES chunks:

```bash
msentity-predict-fredis \
  --input source.msds \
  --output source_fredis.msds \
  --model model.pt \
  --db MoNA \
  --batch-size 32 \
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
still be saved. `<output>.run/run.json` records the execution and
`<output>.run/failures.tsv` contains record-level failures. Existing output is
replaced only with `--overwrite`; output publication uses an atomic rename.

The same operation is available in **Workbench → Predict Spectrum → Batch
MSDataset prediction**. `fredis-predict-spectrum` is an equivalent command name.



---

## Documentation

See the [FREDIS documentation](docs/index.md) and the [shared product overview](fredis_workbench/content/overview.md). Build the Sphinx + Furo guide with `pip install -e ".[docs]"` and `python -m sphinx -W --keep-going -b html docs docs/_build/html`.

## Python code style

Install the development tools with `pip install -e ".[dev]"`. Format the FREDIS
package with `python -m black fredis`, or verify its formatting with
`python -m black --check fredis`. The shared configuration in `pyproject.toml`
excludes the third-party repositories under `fredis/libs`.

Keep one operation per statement, name tensor indices after the objects they
reference, and split long workflows at domain boundaries such as feature
encoding, transition enumeration, probability propagation and loss calculation.
Document tensor ownership and indexing where these are not apparent from types.
