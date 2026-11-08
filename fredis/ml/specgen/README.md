# Spectrum inference

The model is described in [SOURCE_REACTION.md](SOURCE_REACTION.md) and its
training objectives in
[the training README](../training/fragment_tree_training/README.md).

`predict_spectrum.py` loads a checkpoint produced by
`fragment_tree_training` and predicts an MS/MS spectrum. A direct prediction
needs four values: the trained model, SMILES, collision energy (CE), and
`AdductType`.

Run commands from the application root (the directory containing
`pyproject.toml`).

## Model and configuration

Checkpoints written by `fragment_tree_training` (`last.pt` and `best.pt` in the
training output directory) include both `model_config` and `model_state_dict`,
so only `--model` is required.

For a raw `state_dict`, also supply the matching generator configuration:

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --model /path/to/state_dict.pt \
  --params /path/to/model_config.json \
  --smiles 'CCO' \
  --ce 20 \
  --adduct-type '[M+H]+'
```

The configuration must describe `SourceAnchoredFragmentSpectrumGenerator` and must match
the preprocessing definitions and model dimensions used for training.


## Direct prediction

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --model /path/to/training-output/best.pt \
  --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' \
  --ce 20 \
  --adduct-type '[M+H]+' \
  --output-dir prediction-output
```

The precursor m/z is calculated from the neutral exact
mass represented by the SMILES and the specified adduct.

CE should normally be an eV number such as `20`, or a string such as `20eV`.
Percentage CE is also accepted, although a numeric eV value is preferable
when no instrument is supplied.

Multiple SMILES can be passed after `--smiles`; the same CE and adduct are
used for each one:

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --model /path/to/model.pt \
  --smiles 'CCO' 'CCN' \
  --ce 20 \
  --adduct-type '[M+H]+' \
  --output-dir prediction-output \
  --output-name predictions.msds
```

The default device is CUDA when available, otherwise CPU. Override it with
`--device cpu` or `--device cuda`. The output is an `MSDataset` (`.msds`)
whose peaks contain m/z, normalized intensity, and formula metadata.


## Predict records from an existing MSDataset

Predict every record of an existing MSDataset:

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --input input.msds \
  --model /path/to/model.pt \
  --output-dir prediction-output
```

By default it uses the `SMILES`, `PrecursorMZ`, `AdductType`, and
`CollisionEnergy` columns and predicts all unique SMILES. Use the corresponding
`--*-column` options when a dataset uses different column names. Passing
`--smiles` in this mode restricts prediction to matching records.

Before inference, the predictor fragments every unique SMILES with RDKit.
`--num-workers` runs this precompute in parallel worker processes (1 runs
serially) and `--chunk-size` sets how many compounds each worker chunk receives.
Temporary task/result files are deleted after every chunk completes unless
`--keep-temp` is supplied. `--max-samples` bounds the number of simultaneous
prediction samples; larger inputs are split.

Records containing atoms outside the checkpoint's learned atom vocabulary are
skipped and written to `run/failures.tsv`. Unexpected record-specific errors
during inference are isolated by retrying the affected batch in progressively
smaller groups; successful records continue to `predicted.msds`.

Use `--strict` when you want checkpoint loading to fail on any missing or
unexpected parameter. Without it, counts are printed for incompatible keys.

### Optional full theoretical fragment coverage

Batch prediction accepts --calculate-theoretical-fragments (default OFF).
After predicting and comparing spectra it generates full chemistry with the
checkpoint's Fragmenter and the chosen similarity m/z tolerance, then assigns
the matched observed spectra's peaks. It uses --num-workers subprocesses and
shares each SMILES tree across its spectra. The tree depth comes from the
Fragmenter; the combination limit is 10000 and the unique-fragment limit is
unlimited. A limit or chemistry failure is recorded explicitly, never replaced
with a truncated tree or a zero score.

predicted.mssim adds theoretical_fragment_peak_intensity_coverage and
theoretical_fragment_error, preserving spectra, scores and source indices.
Metadata records the effective Fragmenter parameters, alpha and search limits.
theoretical-coverage.tsv carries the same result table; run/run.json records
the failure count. Such failures mark partial success (exit 3), while predictions
and successful assignments remain available. Overwriting with the flag OFF
removes the stale theoretical TSV.

Both predicted and theoretical peak coverage refer to the observed spectrum:
Σ assigned observed I^(2α) / Σ all observed I^(2α), using --intensity-exponent
(default 0.5). The former uses similarity's peak pairing; the latter uses
Fragmenter's formula/adduct/hydrogen-shift explanations, independently of
learned branch probabilities and predicted intensity thresholds. The enhanced
msentity viewer plots predicted coverage vs similarity and theoretical coverage
vs predicted coverage, with alpha labels and existing scatter/density and
image export controls.
