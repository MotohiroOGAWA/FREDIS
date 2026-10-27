# Verify the complete workflow on measured spectra

A useful regression check exercises the application boundaries as well as unit
tests: measured inputs, preparation, CUDA optimization, imported reactions,
checkpoint resume, and published predictions. A small run verifies execution and
feature preservation; it does not establish predictive accuracy.

## Reproduce the small real-data check

The repository includes `tests/workflows/verify_gpu_pipeline.py`. Its selector
expects the MoNA flavonoid MSDS export used by the local verification dataset,
with original record indices 4305, 4358, 3765, and 3762. Those indices are part of
its input provenance and are not interchangeable with arbitrary exports.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tests/workflows/verify_gpu_pipeline.py \
  --source path/to/mona_pos_cf_flavonoids.msds \
  --output runs/workflow-verification \
  --device cuda
```

Select `--training-command fragment-tree` for shared settings or
`--training-command fragment-tree-detailed` for independent component settings
(the default). Run each mode in a fresh output directory. GUI command-builder
integration is verified in the separate Workbench repository.

Use `--device cpu` to exercise the same CLI workflow without CUDA. The source
must be available locally; measured data are not downloaded by the script.

## What the script checks

| Stage | Verification |
|---|---|
| Select spectra | Original peaks/metadata retained; numeric CE mapping recorded |
| Prepare base dataset | Training/validation structures exist and validate |
| Train | A compact model executes optimizer steps and writes finite weights |
| Resume | Epoch/global step continue using a resumable checkpoint |
| Prepare expanded dataset | Imported reaction assets are bundled and a new pattern actually matches |
| Fine-tune | Existing complete reaction tensors remain equal after ID remapping; MolEncoder updates |
| Single prediction | Strict checkpoint loading produces a nonempty spectrum |
| Batch prediction | Four inputs run in chunks with preparation workers and preserve record order |
| Compare predictions | Single and corresponding batch peaks agree within floating-point tolerance |

The additional carbonyl pattern is a workflow probe, rather than a validated
chemical fragmentation mechanism. Inspect `trainable_parameters.json` and import
bindings when interpreting which modules were trained.

## Keep reproducible evidence

The output directory stores selected datasets, chemistry/architecture documents,
per-stage CLI logs, prepared structures, checkpoints, prediction artifacts,
`progress.json`, and `report.json`. `--continue-run` continues an interrupted
verification using its successfully recorded stages.

Use a fresh output directory after changing the implementation or test
configuration when you need a complete new verification. Keep the raw source,
original indices, package revision, GPU/environment, and commands with results.

For an installation check without measured datasets, execute the fully bundled
[documentation examples](../development.md#execute-the-documented-examples).

## Automated regressions

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
```

Tests cover teacher persistence, chemistry-free loading/training, cumulative
path MIL, reaction atom features, nearest endpoints, tensor bucket packing,
source/sample isolation, cache replacement, gradients, single/batch agreement,
pattern import/remapping, and CLI/Workbench interactions. CUDA-specific checks
run when a GPU is available.

Run the documentation build and example checker separately; they validate user
instructions and references in addition to the model regressions.
