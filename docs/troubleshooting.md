# Troubleshooting

Start with the stage that failed. A prepared dataset, architecture file,
checkpoint, and result report have different roles even when their extensions
look similar. Preserve the failing command and log with the resolved configuration.

## Environment and imports

| Symptom | Check / action |
|---|---|
| `torch`, `torch_geometric`, or `rdkit` cannot import | Use the intended Python environment and install scientific dependencies before the editable package |
| A bundled `mmkit`/`msentity` import fails | Run `python scripts/install.py` in the checkout to build and install both native bindings |
| CUDA is unavailable | Check `torch.cuda.is_available()` and the installed driver/build; use `--device cpu` for a CPU check |
| Workbench succeeds in a terminal but fails in its job | Compare `fredis.pythonPath`, `fredis.applicationRoot`, and **Settings → Environment** with that terminal |

## Preparation and data selection

| Symptom | Check / action |
|---|---|
| Required column missing | Inspect actual metadata and set the corresponding `--*-column` mapping |
| No valid records | Read `invalid_records.json`; correct structures, adducts, CE values, and positive precursor m/z |
| Validation is empty | Inspect canonical-molecule overlap and `split_manifest.json`; choose disjoint molecules |
| Source skipped after a limit | Read retained-depth/search counters in the manifest and `skipped_sources.json` |
| Many peaks remain unassigned | Review chemistry, ion/shift coverage, precursor resolution, mass tolerance, and intensity filtering |
| Search cost is too high | Inspect maxima in `reaction_statistics.json`; use finite combination/structure limits and appropriate reaction depth |

Changing a tolerance or reaction set changes candidate coverage. Inspect and
regenerate prepared data when those chemical definitions change.

## Training and imported models

| Symptom | Check / action |
|---|---|
| Dataset preparation configuration not found | Pass the preparation root to `--dataset-dir`, keeping its root config and split directories together |
| Model vocabulary mismatch | Compare dataset symbols/adduct rules with the checkpoint's resolved `model_config` |
| Imported pattern ID appears different | Inspect destination definitions and `fine_tuning.pattern_imports`; IDs are remapped |
| Architecture override rejected on resume | Resume restores saved architecture; use a new fine-tuning dataset/run to add patterns |
| No spectra train despite prepared files | Read `dataset_summary.json` and assignment-score admission thresholds |
| CUDA out of memory | Reduce source `--batch-size` and simultaneous `--max-samples`; inspect molecular sizes, reaction counts, and history depth |
| Training output already exists | Choose a new run, explicitly authorize overwrite for fresh training, or use `--resume` for continuation |

## Prediction

| Symptom | Check / action |
|---|---|
| Unsupported precursor | Verify the checkpoint's main-adduct/precursor chemistry before selecting a different model |
| Strict state loading fails | Use the checkpoint's own compatible configuration; a report or reaction asset may not be a full prediction checkpoint |
| Fewer output records than input | Read `run/failures.tsv` and `run/run.json` alongside output counts |
| Direct `predict()` rejects several packs | Use `predict_batches()` and map `sample_input_index` back to request order |
| Single/batch rows look reordered | Use input identifiers/`SourceSpecID` and chunk `sample_input_index` |
| Output m/z differs from recorded precursor metadata | Compare `PredictedPrecursorMZ` with the retained original `PrecursorMZ` |

## Documentation

Build with `python -m sphinx -W --keep-going -b html docs docs/_build/html`.
An unknown Mermaid/MyST directive usually means the docs extra is missing or a
different Python is invoking Sphinx. Build from a complete checkout: API and CLI
reference pages import the current implementation. See [development](development.md).
