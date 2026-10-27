# FREDIS documentation

**Fragmentation with Reactions that are Externally Definable for In Silico Spectra**

FREDIS learns molecular fragmentation from experimental MS/MS spectra and predicts
spectra from molecular structures. Reaction SMARTS define the chemical search;
a neural model selects compatible reaction histories and predicts physical-ion
intensities under an adduct and collision-energy condition.

This documentation describes Python package **{{ release }}**. The optional
FREDIS Workbench GUI is maintained in a separate repository.

## Start with a task

| I want to… | Start here |
|---|---|
| Run a complete small example | [Quickstart](quickstart.md) |
| Install the Python environment and VS Code tools | [Installation](guides/installation.md) |
| Turn measured spectra into training inputs | [Data preparation](guides/data-preparation.md) |
| Train, resume, or tune a model | [Training](guides/training.md) · [Parameter tuning](guides/tuning.md) |
| Reuse learned reactions and add new ones | [Fine-tuning](guides/fine-tuning.md) |
| Predict one spectrum or an entire dataset | [Prediction](guides/prediction.md) |
| Understand the neural model and tensors | [Architecture](concepts/architecture.md) · [Input schema](reference/input-schema.md) |
| Find a command option or Python call | [CLI reference](reference/cli.md) · [Python API](api/index.md) |

## The workflow

```{mermaid}
flowchart LR
    INPUT["Measured spectra<br/>SMILES, adduct, CE and peaks"] --> PREP["Prepare chemical structures<br/>Assign peaks and save supervision"]
    RULES["Reaction SMARTS<br/>Ion and hydrogen-shift rules"] --> PREP
    PREP --> TRAIN["Train or fine-tune<br/>Save a checkpoint"]
    TRAIN --> PREDICT["Predict from structure<br/>Single or batch"]
    CONDITIONS["SMILES + adduct + CE"] --> PREDICT
    PREDICT --> OUTPUT["Peaks, formulas and provenance<br/>Inspect in Workbench"]
```

Training and validation molecules remain separate. Different CE measurements of
one molecule can share chemical preparation and reaction features. A prediction
requires a checkpoint trained for the chosen reaction and ion vocabulary.

The guides distinguish runnable examples from reference material. The quickstart
uses synthetic peaks to verify installation. The real-data verification guide
runs a small selection of measured spectra; a short workflow run establishes
software behavior, rather than predictive accuracy.

```{toctree}
:maxdepth: 2
:caption: Getting started

quickstart
guides/installation
```

```{toctree}
:maxdepth: 2
:caption: Workflows

guides/data-preparation
guides/training
guides/fine-tuning
guides/prediction
guides/tuning
guides/reaction-pattern-analysis
guides/trainable-spec-ids
guides/pattern-workflow-verification
```

```{toctree}
:maxdepth: 2
:caption: Model and data

guides/fredis-model
concepts/chemistry
concepts/architecture
guides/reaction-prediction-training
guides/annotation-schema
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cli
reference/configuration
reference/input-schema
reference/file-formats
api/index
troubleshooting
development
```

Source: [MotohiroOGAWA/FREDIS](https://github.com/MotohiroOGAWA/FREDIS).
