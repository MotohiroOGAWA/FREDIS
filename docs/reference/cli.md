# Command-line reference

Use `python -m fredis.cli --help` to inspect command groups. Prediction has its
own module/installed entry points. Tables below are generated from the argument
parsers in this checkout; defaults track the implementation used to build this
site.

## Command map

| Task | Command | Guide |
|---|---|---|
| Prepare supervision | `fredis train create-fragment-tree-data` | [Preparation](../guides/data-preparation.md) |
| Train or resume with shared architecture | `fredis train fragment-tree` | [Training](../guides/training.md) |
| Train with independent component settings | `fredis train fragment-tree-detailed` | [Training](../guides/training.md#configure-components-independently) |
| Predict spectra | `fredis-predict-spectrum` | [Prediction](../guides/prediction.md) |
| Export a reaction model | `fredis train export-pattern` | [Fine-tuning](../guides/fine-tuning.md) |
| Search hyperparameters | `fredis train tune-fragment-tree` | [Tuning](../guides/tuning.md) |
| Export trainable SpecIDs | `fredis train trainable-ids` | [Eligibility checks](../guides/trainable-spec-ids.md) |
| Compare reaction sets | `fredis analysis reaction-pattern` | [Pattern analysis](../guides/reaction-pattern-analysis.md) |
| Assign formulas | `fredis assign-formula run` | Formula/ion annotation utility |
| Calculate descriptors | `fredis molecule descriptors` | Molecular descriptor table utility |
| Search SMARTS | `fredis molecule smarts-search` | Metadata structure filtering utility |
| Split datasets | `fredis train split-datasets` | Dataset split utility |

## Prepare training data

```{fredis-cli} prepare
```

## Train or resume

`--epochs` specifies additional epochs when resuming. Shared architecture
settings expand into the detailed training arguments. Imported frozen patterns
retain their architecture. `--hidden-dim` sets atom, molecule, reaction and
condition widths; see the [mapping](configuration.md#shared-training-arguments).

```{fredis-cli} train
```

## Train with detailed settings

Use this interface for independent molecular, pattern, Transformer, GAT,
SetTransformer and head settings, or `--model-architecture` documents. It follows
the same training path as the shared command.

```{fredis-cli} train-detailed
```

## Predict spectra

The module form is `python -m fredis.ml.specgen.predict_spectrum`. Direct prediction
requires `--smiles`, `--ce`, and `--adduct-type` when `--input` is absent.

```{fredis-cli} predict
```

## Export a reaction pattern

```{fredis-cli} export-pattern
```

## Tune training parameters

```{fredis-cli} tune
```

## Export trainable SpecIDs

```{fredis-cli} trainable-ids
```

## Reaction Pattern Analysis

```{fredis-cli} reaction-pattern
```

## Formula assignment

```{fredis-cli} assign-formula
```

## Molecular descriptors

```{fredis-cli} descriptors
```

## SMARTS search

```{fredis-cli} smarts-search
```

## Split datasets

```{fredis-cli} split
```
