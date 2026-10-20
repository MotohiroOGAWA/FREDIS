# FREDIS Workbench for VS Code

**Fragmentation with Reactions that are Externally Definable for In Silico Spectra**

A VS Code extension for configuring, running, and inspecting FREDIS applications. It includes Fragment Tree data preparation and end-to-end model training.

## Reaction Pattern Analysis

**Data → Reaction Pattern Analysis** measures peak assignment along a tree of
reaction pattern sets across one or more spectrum datasets. Input / Run has
Datasets, Fragmenter, **Pattern Sets** and **Analysis Tree** sections: add
pattern or pattern-set files (a single pattern becomes a one-pattern set),
edit a set's patterns, and build the tree with **+ Add Root**, **+ Add Child**,
**Edit PatternSet** and **Delete Node**. One set can appear at several nodes.
Each node analyses the union of the sets on its path; each edge compares the
parent condition with the parent condition + the child set for compounds whose
**precursor SMILES** matches a pattern the child set adds, and roots report
absolute coverage. **Save/Load Configuration** and **Copy command** include the
tree; the CLI equivalent is `python -m fredis.cli analysis reaction-pattern
--tree-config-json …` (the tree is embedded in the command; no external file). Opening a result restores the tree with its calculated nodes
marked; add nodes and run with *Add new nodes to the existing result* to
calculate only the new conditions.

A run writes `result.reactionpatternanalysis.pft` (A, HDF5 summaries),
`compounds/*.peaks.parquet` (B, one file per compound), and
`details/**/*.preft.pt` (C, fragment/reaction detail). Open the result by
clicking it in the Explorer (it has its own custom editor, so the binary file
never opens as text), with **Open With**, by dropping it on **Load
Configuration** or the **Generated Result** area, or with **Results → Open**.
Both **Input / Run** (from the saved configuration) and **Results** are
restored. Results load lazily, A → B → C:

- opening reads only A: Overall Summary, the result tree and **Tree Edges**
  (one row per node: coverage, Δ coverage, parent coverage, mean Δ, applicable
  and improved compounds, overall coverage), Score Frequency per node, and the
  Compound Summary list filtered from A;
- selecting a node shows its parent condition, child condition, added set,
  absolute and delta metrics and distribution;
- selecting a compound reads only its B Parquet: the Compound / Peak view shows
  one column per node as **NEW** (versus its parent), ✓, ✗ or — (not
  evaluated), and Compound × Tree Node rows open their distributions;
- clicking a node cell reads only the C files on that node's path.

See [`docs/guides/reaction-pattern-analysis.md`](../docs/guides/reaction-pattern-analysis.md)
for definitions, CLI usage, and the full result format.

## Research projects

Choose **Home > Open / Create Project** to keep automatic form drafts, named configuration snapshots, tracked files, notes, and run logs in a research directory. Restore settings from history, compare parameter changes, and switch between recent projects. See [Workbench projects](content/projects.md) for details.

## Reaction pattern file editors

Open `*.reaction.pft` directly in VS Code to edit one reaction pattern: its name, reactant SMARTS, and products. Ctrl/Cmd+S saves the single-pattern JSON format. `*.reactionset.pft` opens the pattern-set editor.

## Development

1. Open this directory in VS Code and press `F5` to launch an Extension Development Host, or run `npm run dev`. The F5 launch configuration automatically opens the parent FREDIS application directory, so `Open Folder` is not required on each restart.
2. Run `FREDIS: Open Workbench` from the Command Palette.
3. Select the input, edit or load the Fragmenter parameters, select an output directory, and press `Run CLI`.

In a Development Host, the extension watches JavaScript, CSS, HTML, and JSON files below `src/` and `media/`. Saving a change reloads the Development Host after approximately 250 ms. The watcher is disabled in normally installed VSIX builds.

The extension does not duplicate the data-preparation implementation. It invokes the existing Python CLI as a child process:

```bash
python -m fredis.ml.data_preparation.fragment_tree.create_fragment_tree_training_data ...
```

The `Training` tab follows the same workflow: select the generated training and validation structure directories, configure the model and optimizer, then run or copy the existing training CLI command:

```bash
python -m fredis.ml.training.fragment_tree_training.training_model ...
```

Training configurations use the dedicated `*.pfttrain.json` suffix. Load, drop, or save them from the top-right of the Fragment Tree Training page. They contain run settings and trainable Branch / Fragment Transformer / Ion-State / Intensity parameters only. Fragmenter settings, Reaction Patterns, ion-adduct rules, Symbols, molecular-encoder parameters, and adduct ordering are inherited from prepared datasets and checkpoints, checked for compatibility, and displayed read-only. The Workbench only passes supported training arguments to the CLI; resolved model metadata remains owned by the Python implementation and its checkpoints.

Set `fredis.pythonPath` when the required Python environment is not available as `python`. The FREDIS project directory defaults to the parent directory of this extension and can be overridden with `fredis.applicationRoot`.

## Batch spectrum prediction

In **Predict Spectrum**, the single-molecule preview remains available and a
separate **Batch MSDataset prediction** section accepts an input `.msds` and an
output directory. Select the trained checkpoint and device, specify the source column names,
DB label, chemical precompute batch size, parallel prepare workers and compounds per worker chunk, then run
or copy the CLI command. The
Workbench invokes:

```bash
python -m fredis.ml.specgen.predict_spectrum --input input.msds --output-dir prediction-output --model model.pt --db MoNA
```

The checkpoint is loaded once. FREDIS creates a temporary directory next to the
output and invokes a separate Python subprocess worker for each compound batch.
Each worker reconstructs only the non-neural chemistry context, not a copy of
the full prediction model. Neural prediction then expands only model-selected fragments and
finishes that compound before moving on. Compact `tqdm` status is shown in the
tab without appending every refresh to the log; the operation can be stopped
from the Workbench. Output
metadata preserves source values and adds interoperable prediction provenance;
the final `.msds` is written below that directory, and run details plus
record-level failures are written under `<output-dir>/run/`.

**Load Configuration** and **Save Configuration** preserve the model/device,
single-spectrum SMILES, adduct and collision energy, plus every batch prediction
path, column mapping, DB label, chemical batch size, precompute worker count,
temporary-cache option, and overwrite option.

## Configuration and results

- `Save Configuration` and `Load Configuration` export and import Workbench settings as JSON.
- Fragmenter parameters use a reusable form component with separate `Load Fragmenter` and `Save Fragmenter` actions. Add/remove AdductType rules and fragment-ion adducts with the +/− controls. `Edit Reaction Pattern Set` opens the existing pattern editor; `Apply to Fragmenter` applies its changes.
- Every run also writes `fragment-tree.pft` with input/output settings and embedded `fragmenterParams` to its output directory. Load this file to restore the run. Both Run CLI and Copy Command pass the edited values using `--params-json`; loading a Fragmenter file never makes it an output destination.
- Every output directory receives a `fragment-tree.pft` result manifest. Opening it in Explorer displays the run status and structure manifests in the FREDIS result viewer.
- Select a JSON or TSV entry in the result viewer to open it in the standard VS Code editor.
- Select an individual `.preft.pt` structure to inspect its fragment-tree drawing,
  samples, peaks, selected
  reaction depth, formula groups, terminal nodes, ordered edge paths, and
  expansion nodes directly inside the result viewer.
- The Workbench header includes `Open Result` for reopening any generated
  `fragment-tree.pft` file.

## Reaction Pattern Set editor

Select `Reaction Pattern Set` on the left side of the Workbench navigation. The tab provides its own `Load Configuration` and `Save Configuration` actions for `*.reactionset.pft` documents. It allows you to:

- edit the pattern-set name;
- add and remove patterns;
- edit each pattern name and `reactant_smarts` value;
- add and remove products;
- edit each product name and `smarts` value.
- load or save an individual pattern as `*.reaction.pft`;
- build a pattern visually from a SMILES structure using RDKit;
- draw single, double, triple, and aromatic bonds in a PubChem-style structure view;
- select atoms and bonds individually or with a freehand lasso, and clear the selection explicitly;
- show element symbols and source atom indices on every atom;
- open a dedicated VS Code periodic-table panel with the current elements preselected and return the applied selection;
- use the source element initially or choose any non-hydrogen atom;
- combine single, double, triple, and aromatic bond types with OR (initially the source bond type), and optionally require a ring bond;
- edit the reactant graph directly in Product Transformation to retain/delete atoms and preserve, change, or delete bonds.

The editor reads and writes this JSON shape:

```json
{
  "reaction_pattern_set": {
    "name": "single_bond_cleavage_pattern_set",
    "patterns": [
      {
        "name": "single_bond_cleavage",
        "reactant_smarts": "[!#1:1]-[!#1:2]",
        "products": [
          {
            "name": "single_bond_cleavage",
            "smarts": "[!#1:1]"
          }
        ]
      }
    ]
  }
}
```

`Apply Reactant` and `Add Product` validate their generated SMARTS/SMIRKS with FREDIS `_ReactionPattern.from_rules()`. Validation failures appear as VS Code error notifications.

Opening a `*.reactionset.pft` file directly in Explorer also uses the dedicated structured editor. Directly opened documents support VS Code save, undo, and redo. Use `Reopen Editor With... > Text Editor` when raw JSON editing is preferred.

## Single-SMILES reaction viewer

Open `Visualization → Reaction Viewer`, load a pattern set, and select either the complete set or one pattern. Every mode lists reaction Patterns on the left and matches under `Matched Reactants`; there is no separate `Matched Reactions` stage. Selecting a reactant opens its product choices. Clicking a bond filters the list to reactants containing that bond, and clicking it again clears the filter. In `Stepwise reactions`, each selected reaction can be removed independently and no reaction-count input is required. `Exhaustive candidates` enumerates valid unordered reaction sets up to `Max reactions`; `-1` removes the explicit reaction-count limit. This page reuses the exact Reaction Preview UI instance from the Reaction Pattern editor.

## Package

```bash
npm install
npm run check
npm run package
```

Packages are written to `artifacts/fredis_workbench-<version>.vsix`, keeping generated files out of the extension root. Install the generated file with `Extensions: Install from VSIX...` in VS Code.

### Assignment score distribution

The result viewer reads all sample rows from `assignment_scores.tsv` in the result
folder and its `train_structures` / `validation_structures` folders. Use **Add
Result…** to load one or more additional Data Preparation result files. Checked
datasets are overlaid in the same chart; each dataset can be renamed, hidden,
removed, assigned separate bar and line colors, and given its own opacity and visual
bar width. Opacity and bar width default to 1 and 100%, respectively. The histogram
bin count is configurable independently from those visual bar widths (default: 10).
The semi-transparent bars show sample counts on the left axis, while matching lines
show cumulative percentages on the right axis. Bins
include their lower boundary, with score 1 included in the final bin. Missing,
nonnumeric, and out-of-range scores are excluded and their count is displayed.

Chart title, subtitle, axis titles, legend, and tick labels can be renamed where
applicable, resized, spaced, or hidden. Minimum outer spacing and image size are
editable; the canvas grows automatically when visible labels need more room.
Background and text colors are also editable. **Save Settings…** writes the chart,
series, and referenced result paths to a `*.scorechart.json` file, and **Load
Settings…** restores the complete comparison. **Save PNG** saves the rendered
comparison; **Save TSV** exports the dataset name, bin bounds, counts, cumulative
counts, and cumulative percentages for all visible datasets. Use **Refresh** after
score files are generated or updated.

### SMARTS Search

Select **SMARTS Search**, browse for an `.msds` file, enter a SMARTS query
(for example `c1ccccc1`), and press **Run CLI**. The SMILES column defaults to
`SMILES` and can be changed. **Cancel** stops an active search.

Results show matching records and matching unique molecules, with counts and
percentages. Record percentages use all valid SMILES records as the denominator;
unique percentages use valid molecules deduplicated by RDKit canonical isomeric
SMILES. Missing and invalid SMILES are excluded and reported separately. A record
is counted once regardless of the number of substructure matches. Matching uses
RDKit's default substructure behavior (chirality is not required). An empty valid
set displays `N/A` for its percentage.

The Workbench only invokes the FREDIS CLI and displays its JSON output. Search
and counting logic live in the Python application, not the extension. **Copy
Command** copies the same command and arguments used by **Run CLI**, using
`fredis.pythonPath`. Run the copied command from the application directory shown
in the status message (`mnt/app` by default).

The CLI can also be used independently from the application directory:

```bash
python -m fredis.cli molecule smarts-search --input /path/to/data.msds --smarts 'c1ccccc1' --smiles-column SMILES
```

An installed FREDIS environment also supports `fredis molecule smarts-search`.
Results go to stdout as JSON; errors go to stderr with a nonzero exit code.

Run `python scripts/check-smarts-search.py` with that environment to check the
counting behavior.

Run `node scripts/check-smarts-cli.js` to verify CLI results, failures, and copied
command quoting with the FREDIS Python environment available as `python`.

Search results include separate **Detected compounds** and **Not detected
compounds** lists. Each row shows a canonical isomeric SMILES and the number of
source records for that molecule. Lists show 50 molecules per page and support
SMILES text filtering. Missing and invalid SMILES remain excluded from both lists.

Workbench runs and copied commands include `--include-compounds`. This CLI flag
adds a `compounds` array to the JSON result; each entry contains `smiles`,
`matched`, and `records`. Without the flag, CLI output remains summary-only.

Use **Add reactant SMARTS** / **Remove** to edit multiple queries (at least one
is required). The summary compares every query with **Any reactant SMARTS (OR)**,
for both record counts/percentages and unique-molecule counts/percentages. The OR
row counts each record or molecule only once even when multiple queries match.
All rows use the same valid-SMILES denominators. Select **Compound lists for** to
inspect the detected/undetected lists for the union or an individual query.

Repeat `--smarts` when using the CLI:

```bash
python -m fredis.cli molecule smarts-search --input data.msds --smarts O --smarts N --include-compounds
```

Top-level JSON counts describe the OR union. `patterns` contains each query's
counts, percentages and zero-based `index`. Compound entries include
`matchedPatterns` (matching query indices); `matched` indicates any query match.
The file is loaded once and each distinct input SMILES is parsed once per run.

### Fragment Tree Fine-tuning

Fine-tuning is part of **Fragment Tree Training**. Choose **Fine-tune patterns**,
select a base checkpoint, and use training/validation structures regenerated
with the complete expanded reaction pattern set. The training command uses
`--fine-tune-checkpoint` and `--adapter-width`; resume an interrupted expansion
from the same page. Existing base parameters remain frozen while new category
parameters and projection adapters train.

See `../fredis/ml/training/fragment_tree_training/README.md` for CLI usage,
checkpoint/resume behavior, and the exact expansion architecture.

### Training metrics

Open a training job in **Training Jobs** and choose **Metrics**, or choose
**Results** to open its complete training report. You can also open a run's
`training.pft` directly from Explorer. Select a metric and click **＋ Add chart** to add charts; use
**×** to remove them. Click **Add all charts** to add every available chart at
once. Enter a search in **Filter charts** to use **Add matching charts** instead.
**Clear charts** removes all selected charts. Set **Layout** to **Vertical** to stack all charts in one
column, or **Grid** to arrange them side by side. The layout selection is retained
in the webview state. **Auto refresh (15s)** updates charts while the tab is visible.
The viewer reads `metrics.tsv` and `metric_distributions.tsv` directly. Distribution charts combine min–max and q1–q3 bands with a median line over global
steps. Mean charts group adducts (or CE ranges) for comparison. Each chart combines
train, train_window, and validation, with colors and line styles plus toggles in the
legend. These are recorded summary statistics, rather than raw sample histograms. Non-finite and missing values are omitted.
Hover over a chart to inspect its nearest recorded step and value. The run path and
selected charts are retained in the webview state.

### Spectrum viewer

Open **Visualization → Spectrum Viewer** to browse or drop MSDS, MSP, MGF, TSV
or CSV datasets directly in the Workbench. The viewer supports multiple loaded
datasets, metadata editing, filtering, sorting, export, spectrum comparison and
similarity calculation. The complete msentity viewer runs inside the Workbench:
its UI, structure preview, optional RDKit SMARTS filtering and Python backend
are shared directly, so msentity updates apply when a new Workbench is opened.
Structure and chemistry settings use the msentity extension settings; Python
and page size use the FREDIS settings. The viewer is loaded directly from the
`fredis/libs/msentity/vscode-extension` submodule. Installed msentity extensions
are not used. Packaging runs `build:spectrum-viewer` to include a snapshot of
that submodule, so no separate msentity VSIX download is required.

When dragging from the VS Code Explorer, keep Shift held while releasing the
mouse onto a drop field/button; release Shift after the drop. VS Code owns the
editor drop overlay before the Webview receives any events. The shared document guard protects file drags entering headings/whitespace
before a drop zone.

### Workbench dashboard

Open `FREDIS: Open Workbench` or use the FREDIS Activity Bar. The dashboard
provides quick actions, environment diagnostics and recent training jobs.
The navigation retains the existing preprocessing, prediction, fragment viewer,
SMARTS and evaluation tools. Checkpoint initialization and adapter training are
configured directly in the training workflow.

Training supports AdamW weight decay, gradient clipping, normalized branch MIL,
depth-diverse weak negatives, and physical-ion intensity loss.

Data Preparation and Training expose individual parameter fields. Import Parameters
accepts a Fragmenter JSON through Browse or drag-and-drop and expands it into
the form. The file is optional; execution uses a snapshot of the edited fields.
Advanced buttons reveal less frequently changed settings. Hover over a parameter
label for 500ms to see its description.
The default CLI is `python -m fredis.cli train fragment-tree`; it jointly
trains a newly initialized Mol Encoder. Training options include
`--weight-decay`, `--gradient-clip`, `--branch-weight`, `--negative-weight`,
`--branch-mil-temperature`, `--intensity-loss-weight` and
`--cosine-loss-weight`, all in **Optimization and Loss**. Weak negative reaction
weight defaults to `0.5`; cosine loss weight defaults to `1`. Hover over each
label for its description. Mouse-wheel scrolling does not change numeric input
values; edit the field or use its step arrows. Spectrum generation
settings (branch path threshold, fragment node budget, ion and peak thresholds)
form their own **Generation Params** section, separate from the architecture.

Training Jobs shows live stdout/stderr, epoch progress, searchable logs and a
link to the metrics charts. Training writes `metrics.tsv` and emits JSON Lines
`epoch_end` events. Metadata and bounded log previews live in VS Code workspace
storage. Training continues when its panel closes and uses detached processes
so it can survive extension host restarts. Restored running PIDs are checked;
if the process disappeared, its exit status is reported as unknown. Stopping a
restored process requires verifying it in the terminal to avoid killing a reused
PID. Existing batch prediction retains its original lifecycle.

Run `npm run check:workbench` for dashboard, CSP and CLI argument checks.

CLI configuration precedence is: preset defaults → `--params` file → inline
`--params-json` → section JSON options → named options → repeated `--set PATH=JSON`.
For example, `--params model.json --max-reaction-count 2 --mass-tolerance 0.02Da`
overrides those two values from the file. Any nested value, including pattern or
adduct-rule array entries, can be overridden with `--set`, e.g.
`--set 'fragmenter_params.fragment_ion_tree_builder.reaction_pattern_set.patterns.0.name="custom"'`.
These JSON configuration options apply to dataset preparation, not `train fragment-tree`.
Fragment-tree training uses explicit Mol Encoder/Reaction/Post Model options.
Fragmenter, adducts and reaction patterns are inherited from the train/validation datasets.
JSON configuration arguments
and `--set` are rejected by the training CLI. Dataset column mapping is configurable in preparation.


### Home and dataset preparation

Home shares the FREDIS introduction with the [documentation](https://github.com/MotohiroOGAWA/FREDIS/tree/main/docs).
Quick Start and the jobs, samples and models tabs open workflows, copy paths and
reveal results. Compact forms launch training and single-spectrum prediction.
A compatible pretrained checkpoint loads its configuration and initializes new
training with a fresh optimizer. Resume in the full Training form continues a run.

Training Data has Input Dataset, Fragmentation, Assignment, Output and Run steps.
Browse or drop training and optional validation inputs to inspect real records,
summary statistics, molecular structures and spectra. Column mapping checks
column presence immediately. Click **Validate Dataset Values** for RDKit SMILES,
registered main adducts, parsed collision energy and numeric precursor m/z.
Fragmentation settings use a clickable JSON drop area, periodic-table Symbols
and unique-node / total-transition limits; model dimensions are training-only.
Without validation input, FREDIS splits by unique SMILES.
Run stays visible; missing requirements appear in red and disable execution.

Document, GitHub, theme and Help stay available in the toolbar. From the project
root, install the docs extra and build with `python -m sphinx -W -b html docs
docs/_build/html`. Document opens the Sphinx/Furo pages when built and otherwise
opens the shared introduction.


### Default values and output root

Open **Settings → Preferences** and configure `fredis.workbench.defaultOutputDirectory`.
It defaults to empty; output fields remain blank until a root is configured or a
path is entered. New workflows use named timestamp subdirectories of that root.
Edit `fredis.workbench.dataDefaults`, `trainingDefaults` or `predictionDefaults`
in VS Code settings to configure initial form values. New Workbench forms apply those defaults, including nested parameter
objects. Output paths are generated separately. See the
[defaults guide](https://github.com/MotohiroOGAWA/FREDIS/blob/main/docs/guides/workbench.md).

Intensities are always scaled so each spectrum's maximum is 1. Data **Output → Parallel Processing**
provides Worker Processes and Chunk Size (`1` each by default). The CLI supports
`--num-workers` and `--chunk-size`. Stop terminates the preparation process and its workers.

### Source-anchored training workbench

Open **FREDIS: Start Training** (or **Training → New Training**) to configure
`fredis.cli train fragment-tree`, implemented by
`fredis/ml/training/fragment_tree_training/training.py`.
The training page provides three run modes:

- **New model** trains a newly initialized Mol Encoder jointly with the
  fragment-tree model. Its architecture is edited in the same model-parameter
  UI as the reaction and spectrum models.
- **Resume training** supplies `--resume` to restore the optimizer and epoch.
- **Fine-tune patterns** supplies `--fine-tune-checkpoint`,
  and `--adapter-width`; patterns are inherited from the datasets. Include every old pattern and
  regenerate training and validation structures with the expanded configuration.

The page groups dataset paths, model configuration, training settings and output
beside a live run summary, environment/input checks and weighted loss summary.
Dataset cards recursively count `.preft.pt` files and their bytes without loading
tensors. Estimated optimizer steps are `ceil(training files / batch size) × epochs`.
Dataset schema, saved molecular targets, checkpoint compatibility and actual
memory requirements are verified by the training runtime. **Inspect Dataset**
reveals the directory in Explorer. **Start Training** opens the existing job view
with logs, progress and metrics. **Copy Command** uses the same CLI builder.

### Saving Workbench files

FREDIS configuration files (`*.pft`) use manual saving by default, even when
VS Code Auto Save is enabled globally. Edits remain unsaved until Ctrl+S or
Save All, and closing a modified document follows VS Code’s normal save/discard
flow. This is contributed through the `fredis-workbench-file` language setting;
ordinary text/source files keep their existing Auto Save behavior. An explicit
user/workspace language override can still override this default. Reload the
VS Code window after updating the extension to pick up the new contribution.
