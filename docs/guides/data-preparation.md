# Prepare fragment-tree training data

Preparation turns experimental spectra into shared chemical structures and
explicit supervision. Choose the reaction definitions, ion rules, and input
mapping here; choose neural dimensions later during [training](training.md).

## Input requirements

Supported inputs are `.msds`, MSP, MGF, and msentity spectrum tables (`.tsv`,
`.csv`, `.parquet`). Tables need a `Peak` column such as
`100.1,0.5;150.2,1.0`. A metadata-only table does not provide training peaks.

| Default column | Required value |
|---|---|
| `SMILES` | A molecular structure RDKit can parse |
| `AdductType` | A precursor adduct resolvable through the Fragmenter's main-adduct rules |
| `CollisionEnergy` | A supported numeric or textual energy representation |
| `PrecursorMZ` | A finite positive precursor m/z |
| Peak list | Experimental m/z and relative intensity values |

A stable `SpecID` is useful for provenance and required by dataset prediction
and trainable-ID export. Use explicit mappings when metadata names differ.

## Choose train and validation molecules

With separate measured datasets:

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input train.msds --validation-input validation.msds \
  --output-dir data/prepared \
  --params chemistry.fragmenter.pft \
  --validation-ratio 0.1
```

Canonical molecules appearing in training are removed from validation. With an
explicit validation input, `--validation-ratio` can cap the retained validation
molecule count relative to training. The retained records and removal counts are
reported. Multiple CE spectra for one molecule stay together.

To split one input by unique molecules instead:

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input spectra.msds --output-dir data/prepared \
  --params chemistry.fragmenter.pft \
  --validation-ratio 0.1 --test-ratio 0.1 --validation-seed 42
```

Automatic splitting requires at least two usable unique molecules. A test split
is optional. Training requires usable training and validation structures; an
output with only one populated split does not meet that requirement.

For several files or explicit split assignments, use `--datasets-json` and
`--dataset-mode assigned`. For the quickstart files:

```bash
python -m fredis.cli train create-fragment-tree-data \
  --datasets-json '[
    {"id":"train", "path":"runs/demo-input/train.msds", "split":"train",
     "mapping":{"smilesColumn":"SMILES", "adductTypeColumn":"AdductType",
                "collisionEnergyColumn":"CollisionEnergy", "precursorMzColumn":"PrecursorMZ"}},
    {"id":"validation", "path":"runs/demo-input/validation.msds", "split":"validation"}
  ]' \
  --dataset-mode assigned --output-dir runs/demo-assigned \
  --params docs/examples/chemistry.fragmenter.pft \
  --max-unique-fragment-smiles 100 --max-reaction-combinations 1000
```

Mappings omitted from an entry use the standard column names. Assigned mode
rejects a canonical molecule assigned to different splits. Workbench's dataset
table writes the same configuration. See the
[CLI options](../reference/cli.md#prepare-training-data) for all mapping flags.

## Map metadata columns

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input measured.msds --output-dir data/prepared \
  --params chemistry.fragmenter.pft --validation-ratio 0.1 \
  --smiles-column CanonicalSMILES --adduct-type-column PrecursorType \
  --collision-energy-column CE_eV --precursor-mz-column ParentMZ
```

Validation has separate `--validation-*-column` overrides when its column names
differ. The same input loader backs Workbench preview, validation, and execution.
A preview checks column presence; **Validate Dataset Values** examines every
record. The CLI performs authoritative checks again before splitting.

Malformed structures, unsupported adducts, unparseable CE values, and invalid
precursor m/z values exclude the affected records and appear in
`invalid_records.json`. A record with several invalid fields is excluded once.
Missing required columns and inputs with no usable records are errors.

## Define chemistry and imports

`--params` accepts a Fragmenter configuration, or a chemistry object containing
`fragmenter_params` and `symbols`. The configuration contains reaction SMARTS,
fragment-ion adducts, hydrogen-shift limits, reaction-depth bounds, and mass
tolerance. The [chemistry guide](../concepts/chemistry.md) explains those terms.

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input spectra.msds --output-dir data/prepared \
  --params chemistry.fragmenter.pft --validation-ratio 0.1 \
  --mass-tolerance 0.01Da,10ppm \
  --max-reaction-count 2 \
  --set precursor_candidate_max_reaction_count=1
```

Precedence is: file configuration, inline `--params-json`, named overrides,
then repeated `--set PATH=JSON`. `--set` paths are relative to the Fragmenter.
`--symbols-json '["C","N","O","P","S"]'` selects the raw atom vocabulary.
The observed/chemical adduct vocabulary is resolved from the dataset and rules.

Select pretrained reaction sources with `--pretrained-pattern-model` or
`--reaction-models-json`. Preparation bundles their dependencies in
`pretrained_models/`. See [fine-tuning](fine-tuning.md) before selecting imports.

## What preparation computes

```{mermaid}
flowchart TB
    RECORDS["Validate records and canonicalize source molecules"] --> SPLIT["Assign molecule-disjoint splits"]
    SPLIT --> REACTIONS["Find source reaction matches and pair constraints"]
    REACTIONS --> SETS["Enumerate compatible reaction sets"]
    SETS --> COMPOUNDS["Materialize and deduplicate molecular structures"]
    COMPOUNDS --> IONS["Build ion / hydrogen-shift formula explanations"]
    IONS --> MATCH["Assign measured peaks within mass tolerance"]
    MATCH --> TEACHER["Save BranchTeacher memberships<br/>and spectrum formula targets"]
    TEACHER --> SAVE["Write source structures, manifests and reports"]
```

Source reactions are identified on the original molecule. Reaction sets and
structures are shared across conditions; peak assignments preserve ambiguous
explanations. BranchTeacher stores alternative paths and negative memberships,
while spectrum targets retain CE-specific intensities. Preparation does not
save path-loss temperature weights or learned neural features.

`--minimum-relative-intensity` filters low-intensity measured peaks before
assignment. Retained spectra are normalized by their maximum intensity in the
standard preparation path. Mass tolerance controls assignment, rather than a
learned m/z regression target.

When `H` is absent from `symbols`, explicit hydrogen nodes are omitted from
encoder graphs and attached hydrogen information enters heavy-atom features.
The chemical molecule, formula, and valence are retained. Reaction atom references
are remapped to represented graph atoms.

## Bound expensive searches

| Limit | What it bounds |
|---|---|
| `--max-reaction-count` | Maximum number of primitive reactions in a set |
| `--max-reaction-combinations` | Raw combinations examined across all sizes, before RDKit materialization |
| `--max-unique-fragment-smiles` | Distinct canonical generated fragment structures, excluding the source |

The two per-source cost limits default to `-1` (unlimited); otherwise values
must be positive. A combination limit reached during depth *n* retains only
complete levels through *n − 1*. A structure-limit overflow can retry a smaller
reaction depth. A source whose precursor cannot be resolved in the retained
search is skipped. Other per-source failures are recorded and remaining sources
continue.

Choose limits using the recorded maxima and counts in `reaction_statistics.json`.
Inspect retained depth, skip reason, and search counters before interpreting
coverage. A shallower retained search can change training eligibility.

## CPU workers and output handling

`--num-workers 4 --chunk-size 1` processes independent SMILES groups in separate
Python processes. One worker runs serially. Training and validation are processed
separately; worker results retain deterministic source indices and filenames.
Parent progress combines a terminal progress bar with structured events used by
Workbench.

A fresh output directory is the normal path. `--overwrite 1` authorizes clearing
the selected preparation output before regeneration. Interactive CLI runs can
ask for confirmation; noninteractive runs require the explicit option. Input
and application directories are protected from this overwrite operation.

## Inspect the saved dataset

```text
data/prepared/
  fragment-tree.training-data-config.pft
  fragment-tree.fragmenter.pft
  split_manifest.json
  invalid_records.json
  pretrained_models/                 # present when sources were imported
  train_structures/
    fragment-tree.training-data.pft
    manifest.tsv
    reaction_statistics.json
    assignment_scores.tsv
    invalid_records.json
    skipped_sources.json
    data/*.preft.pt
  validation_structures/
    ...
```

The root configuration stores the resolved chemistry and input mappings. Each
split's manifest records prepared/skipped sources, record counts, paths, reaction
statistics, and retained depth. `.preft.pt` structures include common chemical
tables, observations, BranchTeacher, and spectrum targets.

Continue with [training](training.md). Use the [input schema](../reference/input-schema.md)
for field ownership and [file formats](../reference/file-formats.md) for saved
artifacts and Workbench entry points.
