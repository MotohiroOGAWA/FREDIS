# Prediction API

`load_generator()` restores the checkpoint configuration and handles the training
wrapper's parameter names. Use strict loading to detect a configuration/weight
mismatch before generating spectra.

## Load a trained model

```python
import torch
from fredis.ml.specgen.predict_spectrum import load_generator

generator = load_generator(
    model_path="runs/demo-model/last.pt",
    params_path=None,
    device=torch.device("cpu"),
    strict=True,
)
```

The returned generator is in evaluation mode. `params_path=None` reads
`model_config` from the checkpoint. A raw state dictionary needs an explicit
compatible parameter file.

```{eval-rst}
.. autofunction:: fredis.ml.specgen.predict_spectrum.load_generator
```

## Predict bounded chunks

```python
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.specgen.competitive_spectrum_model import merge_candidate_peaks

sources = [Compound.from_smiles("CCO"), Compound.from_smiles("CC(N)O")]
adducts = [Adduct.parse("[M+H]+"), Adduct.parse("[M+H]+")]
energies = [20.0, 30.0]

for chunk in generator.predict_batches(sources, adducts, energies, max_samples=2):
    print(chunk.sample_input_index)
    peaks = merge_candidate_peaks(chunk.spectra)
    print(peaks.sample_index.shape, peaks.mz.shape)
```

The three input sequences must align. Chunks can group/reorder repeated sources
internally; `sample_input_index` maps each chunk's internal spectrum positions
back to the original request. `chunk.spectrum_batch` links state rows to
Compounds and formula explanations. `chunk.fragments` provides structures for
inspection.

`predict()` returns a single chunk and rejects requests requiring several packs;
use `predict_batches()` for general multi-molecule or larger input. The generator
methods disable gradient recording for prediction.

```{eval-rst}
.. autoclass:: fredis.ml.specgen.source_anchored_spectrum_predictor.SourceAnchoredFragmentSpectrumGenerator
   :members: prepare, predict, predict_batches, predict_prepared_batches, set_generation
   :undoc-members:
   :exclude-members: __new__
```

## Result objects

```{eval-rst}
.. autoclass:: fredis.ml.specgen.source_anchored_spectrum_predictor.SourceAnchoredSpectrumOutput
```

The `spectra` field is a SpectrumModelOutput. It retains physical ion probabilities,
relative intensities, conditional state probabilities, formula/m/z annotations,
fragment and sink probabilities, and optional transition diagnostics.

```{eval-rst}
.. autoclass:: fredis.ml.specgen.fragment_spectrum_model.SpectrumModelOutput
```

`merge_candidate_peaks()` sums formula-equivalent contributions, filters invalid
physical outputs, and applies output normalization. Preserve unmerged rows when
inspecting explanation/state ownership.

```{eval-rst}
.. autofunction:: fredis.ml.specgen.competitive_spectrum_model.merge_candidate_peaks
```

## Predict an MSDataset with provenance

```python
from fredis.data.core.MSDataset import MSDataset
from fredis.ml.specgen.predict_spectrum import predict_msdataset

measured = MSDataset.load("runs/demo-input/train.msds")
predicted, failures = predict_msdataset(
    dataset=measured, generator=generator,
    model_path="runs/demo-model/last.pt", db="Example",
    spec_id_column="SpecID", smiles_column="SMILES",
    adduct_type_column="AdductType", collision_energy_column="CollisionEnergy",
    precursor_mz_column="PrecursorMZ", instrument_column=None,
    include_formula_annotation=True, num_workers=1, chunk_size=1,
)
print(len(predicted), len(failures))
```

The function returns the generated MSDataset and a failure DataFrame. File/run
publication is handled by the CLI, or explicitly by your application. The shared
provenance contract adds `SourceSpecID` and `Prediction*` metadata fields.

```{eval-rst}
.. autofunction:: fredis.ml.specgen.predict_spectrum.predict_msdataset
```
