# Exporting trainable SpecIDs

Export FREDIS-eligible SpecIDs for comparison with independently verified FIORA, ICEBERG, MassFormer, or other tool outputs. FREDIS runs native training data preparation, forward propagation, loss calculation, and backward propagation. Accepted spectra must produce finite losses and gradients. No optimizer updates are performed.

## CLI

```bash
fredis train trainable-ids \
  --input source.msds \
  --params patterns.fragmenter.pft \
  --output-dir /path/to/fredis-eligible \
  --workers 2 --batch-size 8
```

The input requires unique, nonempty `SpecID` values. Use `--spec-id-column` and the other column options for custom metadata mappings. Reaction patterns, adducts, peak filtering, and search limits affect eligibility; use the same settings as production preparation. Without `--params`, the command uses the nonring single bond preset.

Use `--model-architecture architecture.json` to supply production model settings, either directly or inside a `model_config` object. Otherwise, the default training architecture is used. Eligibility depends on the architecture and batch size; `stats.json` records the effective configuration. This check does not guarantee that every future model or training epoch will succeed. Additional assignment-score thresholds are not applied.

Choose a new output directory.

| File | Contents |
|---|---|
| `trainable_spec_ids.txt` | Accepted SpecIDs, one per line, in input order |
| `records.tsv` | Per-record status, processing stage, exclusion reason, loss, and gradient check |
| `stats.json` / `stats.txt` | Counts, exclusion reasons, SMILES/adduct/instrument summaries, and input SHA-256 |
| `run/` | Resume manifest, completed group results, and worker logs |

A successful batch records its aggregate loss for every accepted member. Failed batches are retried individually. Worker crashes are recorded as exclusions. Use `--keep-export` to retain intermediate data.

Repeat the same command with `--resume` to reuse saved group results. Resume rejects changes to the input, implementation, architecture, imported weights, environment, or processing settings. Completed results also verify the SpecID file's SHA-256.

Pass `trainable_spec_ids.txt` to your external intersection tool. FREDIS exports the same newline-separated format as other tools and includes the input SHA-256 in `stats.json` for source validation.

Verification runs on CPU. `--workers` controls the number of parallel worker processes. There is no device selector or processing timeout.

## Workbench

Open Data → **Trainable Spectra**.

1. Configure reaction patterns, adducts, and preprocessing in Training Data.
2. Select the input `.msds` and a new output directory, then click **Verify and export SpecIDs**. Optionally use the current Training architecture.
3. Pass the exported `trainable_spec_ids.txt` to your external intersection tool.

The page displays progress, logs, and result counts. After stopping extraction, resume with the same settings and output directory. **Copy command** copies the CLI command for the current settings without starting extraction.
