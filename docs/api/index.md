# Python API reference

The API follows the same ownership boundaries as the CLI: chemistry definitions,
prepared structures, neural models, and result datasets. The signatures below
are imported from the checked-out code during the Sphinx build.

| Area | Main entry points |
|---|---|
| [Chemistry](chemistry.md) | `Fragmenter`, reaction patterns, source matching and chemical trees |
| [Prepared data](data.md) | `ReactionStructureBuilder`, `SourceReactionStructure`, `SpectrumPredictionData` |
| [Prediction](prediction.md) | `load_generator`, `predict_batches`, `predict_msdataset` |
| [Training and reaction assets](training.md) | `create_spectrum_generator`, `train_reactions`, `inspect_model`, `export_pattern` |

These imports use the implementation's module paths. Scientific dependencies
must be installed as described in [installation](../guides/installation.md).
The [input schema](../reference/input-schema.md) is the reference for batch-local
indices and stored tensor relationships.

```{toctree}
:maxdepth: 1

chemistry
data
prediction
training
```
