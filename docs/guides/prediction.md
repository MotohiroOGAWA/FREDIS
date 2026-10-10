# Predict and inspect spectra

Single prediction uses a trained source-anchored model, molecular SMILES, a supported precursor adduct and collision energy. Apply the model to load its supported adduct types before running prediction.

Batch prediction is available through the CLI:

```bash
python -m fredis.ml.specgen.predict_spectrum \
  --input source.msds --output-dir prediction-output \
  --model models/my-model/last.pt --device cpu
```

Input metadata columns are configurable. The output keeps source metadata and records model and prediction provenance. Batch failures are recorded alongside the output rather than silently omitted.

Use spectrum and fragment viewers to inspect predicted peaks and materialized fragment trees. FREDIS predicts spectra from structures; this interface does not claim to provide a general structure-candidate ranking engine.
