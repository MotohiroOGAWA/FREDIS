# FREDIS Workbench

Install the packaged VSIX in VS Code, open the FREDIS workspace and run **FREDIS: Open Workbench**. Configure `fredis.pythonPath` and, if automatic discovery does not locate the project, `fredis.applicationRoot`.

The sidebar contains **Home**, **Data**, **Training**, **Prediction**, **Visualization** and **Settings**. Documentation, GitHub, theme and help controls remain in the top toolbar.

The low-emphasis **Data → Data Evaluation…** item opens the standalone
Evaluation window rather than adding another full Workbench page. It evaluates
`.mssim` cosine similarities by metadata column, collision energy or molecular
descriptor, and can compare multiple tools or datasets with grouped box/violin
plots. The same window remains available from **FREDIS: Open Evaluation** in the
command palette.

## Home

The product description comes from the same overview included in this documentation. Quick-start actions lead to data preparation, training, prediction and visualization. The resource tabs show real jobs, sample datasets and discovered checkpoints; use their actions to select a path, copy it or inspect results.

Quick training requires prepared train/validation structure directories, output, epochs, batch size and learning rate. An optional compatible pretrained checkpoint initializes a fine-tuning run. Quick prediction transfers model, SMILES, adduct and collision energy to the full prediction workflow and uses the model's supported conditions.

## Data preparation

The tabs are **Input Dataset**, **Fragmentation**, **Assignment**, **Output** and **Run**. Dataset inspection displays file size, record counts, unique SMILES, adduct distribution and bounded record/spectrum previews. Column validation checks train and validation datasets. When validation input is omitted, select the unique-SMILES split ratio and seed.

On **Fragmentation**, the compact right-aligned **Load** / **Save** row directly below the tabs manages the fragmenter config only and also accepts drag-and-drop. It does not accept model configuration JSON.

The fixed bottom-right **Run** button is always clickable. Clicking it checks dataset validation and required settings; red text explains any unmet requirements and processing does not start. Clicking during an active job also displays a message and prevents duplicate runs. The backend validates settings again before launching processing.

## Reaction patterns

Open **Data → Reaction Patterns**, or run **FREDIS: Edit Reaction Patterns** from the command palette. Creation controls appear above the pattern list. **Add Pattern** creates an editable reactant/product SMARTS entry. **Add Pattern Visually** opens the existing structure builder: draw a source molecule, select atoms and bonds, configure their queries, then **Apply Reactant**. While dragging with the left mouse button held down, a colored dashed line follows the mouse in both the source and product views. When drawing a selection loop, a bond is selected only when both of its endpoint atoms are enclosed. Draw and map each product, edit bond types, and use **Add Product**. Existing entries also support visual editing. **Clear All Patterns** removes every pattern while keeping the pattern-set name.

Import a pattern set or an individual pattern by dropping its JSON file onto the corresponding import control, or click that control to browse. **Save Configuration** exports the complete pattern set; individual entries can also be saved separately. Import errors appear in the editor.

To update a preparation run, open **Training Data → Fragmentation → Fragmenter Parameters** and expand **Reaction Pattern Set**. The same complete editor is embedded in place and starts with that run's current pattern set; changes are applied to the preparation configuration immediately. Saving a standalone pattern set is still separate from the current preparation run.

## Fragment ion adduct rules

Open **Data → Adduct Rules**, or run **FREDIS: Edit Fragment Ion Adduct Rules**. The shared editor manages the rule-set name and the hydrogen-shift range and each rule's precursor adduct and fragment-ion adducts. Rules are collapsed initially. **Clear All Rules** removes every rule while keeping the rule-set name. Complete sets use `*.adductset.pft`; individual rules use `*.adduct.pft`, and both open in their dedicated FREDIS custom editor.

The same editor is embedded under the collapsed **Training Data → Fragmentation → Fragmenter Parameters → Fragment Ion Adduct Rule Set** summary. Expanding it edits the current preparation configuration in place without navigating to another tab. **Reaction Pattern Set** uses its existing shared editor in the same way; both summaries remain collapsed by default.

## Fragmenter configurations

The **Fragmentation** settings of **Data → Training Data** (reaction pattern set, adduct rule set, reaction limits and mass tolerance) can be saved with **Save** and restored with **Load**. Fragmenter configurations use `*.fragmenter.pft` and open in the dedicated **FREDIS Fragmenter Config** editor, which shows the same Fragmentation form with the embedded Reaction Pattern Set and Adduct Rule Set editors; edits are written to the file and saved with Ctrl+S. **Save As…** writes a copy, and **Load** (or dropping a file) imports another fragmenter config into the open file. Options missing from an older file are filled from the project preset.

## Fragment tree visualization

Open **Visualization → Fragment Tree**, enter a SMILES, and choose **Generate Fragment Tree**. Only the FragmentTreeBuilder settings are needed: **Max Reaction Count**, the **Reaction Pattern Set** (edited in the same embedded editor as Data Preparation, with its own Load/Save), and the search limits **Max Unique Fragment SMILES** and **Max Reaction Combinations** (-1 is unlimited). Adduct rules and precursor settings do not affect the tree. **Use Training Data settings** copies these four values from Data Preparation. Click a node to depict its fragment, or an edge to list each reaction-state transition; reaction links jump to the primitive reaction table. Click a Fragment Nodes column header to sort ascending, and again for descending; the table can also be filtered by SMILES or formula.

## Spectrum viewer

Open **Visualization → Spectrum Viewer**, then browse for or drop an MSDS, MSP, MGF, TSV or CSV dataset. Multiple datasets can remain open in the same viewer. The table supports metadata editing, filtering, sorting, column selection, export, spectrum comparison and similarity calculation. Set `fredis.workbench.spectrumViewerPageSize` to show between 1 and 500 rows per page (20 by default).

Choose the column containing SMILES values with **SMILES column**, then use a row's **Structure** button to open its molecular depiction beside the Workbench. The structure panel is always rendered on a white canvas so bonds and atom labels remain visible with both light and dark VS Code themes. It uses mmkit `Compound`, `Adduct` and `Formula` objects to derive the neutral formula and exact mass, canonical adduct type, precursor formula, calculated precursor m/z, charge and adduct mass shift. If the row contains `PrecursorMZ` (or a supported alias), the observed value is shown as **Precursor m/z** alongside the calculated value.

## Reaction viewer

Open **Visualization → Reaction Viewer** to apply reaction patterns to one SMILES with the pattern set currently loaded in **Reaction Patterns**. Select the full set or one pattern and choose either mode:

- **Stepwise reactions** lists reaction Patterns on the left and their currently valid matches under **Matched Reactants**. Select a reactant to display its product choices. Clicking a highlighted bond filters the reactant list to matches containing that bond; clicking it again clears the filter. Every selected reaction has its own **Remove** button. Stepwise has no reaction-count input and continues until no compatible reaction remains.
- **Exhaustive candidates** generates all valid unordered reaction sets up to **Max reactions** and displays their fragments and reaction SMIRKS. Set **Max reactions** to `-1` to search without an explicit reaction-count limit (the finite primitive-reaction count remains the effective upper bound).

The viewer and **Reaction Patterns → Reaction Preview** are the same UI instance. The stepwise mask uses the production fragment-tree rules: every reaction retains every other reaction's Source match, no two reactions change the same Source bond, duplicate/redundant reactions are excluded, the retained atom intersection is nonempty, and the reaction limit is respected. Candidate sets must also produce an RDKit-valid fragment.

## Themes and help

Theme preference is saved in the webview's workspace state. The default follows the VS Code theme; the toolbar toggles light/dark overrides. Help explains the current page and links to the product guide.


## Workbench defaults

Open **Settings → Preferences** to edit FREDIS settings. **Default Output Directory**
is initially empty, so output fields start blank. When configured, new forms use
`<root>/preparation-<timestamp>`, `<root>/training-<timestamp>` and
`<root>/prediction-<timestamp>`. Relative roots resolve against the FREDIS project.
Existing form edits and restored job/configuration paths remain under your control.

**Save as Workbench Defaults** on Data, Training or Prediction stores the current
parameter values in VS Code workspace settings (user settings if no workspace is
open). Defaults apply to newly opened Workbench forms. Generated output paths are
excluded from these snapshots. You can also edit the three default objects directly:

```json
{
  "fredis.workbench.defaultOutputDirectory": "/data/fredis-runs",
  "fredis.workbench.dataDefaults": {
    "numWorkers": 4,
    "chunkSize": 1,
    "maxUniqueFragmentSmiles": 1000,
    "maxReactionCombinations": 50000,
    "symbols": ["C", "N", "O", "P", "S"],
    "fragmenterParams": {"mass_tolerance": "0.02Da,10ppm"}
  },
  "fredis.workbench.trainingDefaults": {"epochs": 100, "batchSize": 16, "lr": 0.0001},
  "fredis.workbench.predictionDefaults": {"ce": 30, "adductType": "[M+H]+"}
}
```

Nested parameter objects merge with the built-in preset; explicit form edits and
loaded configurations take priority.

## Parallel data preparation

Open **Data → Output → Parallel Processing**. **Worker Processes** selects the
number of independent subprocesses processing SMILES groups; `1` runs serially.
**Chunk Size** selects how many groups are handled by each subprocess command.
Data preparation uses the shared `fredis/utils/parallel_subprocess.py` runner. Chunks are capped to
keep enough work for the available workers. Training and validation run separately.
The CLI equivalents are `--num-workers 4 --chunk-size 1`. Stopping preparation also
terminates its worker processes.

## Fragment Tree Dataset Results

Use the preparation page's top-right **Load Configuration** drop area to select `fragment-tree.pft` by clicking or dragging from the filesystem or Explorer. The preparation heading does not display a fixed preset name.

The result viewer has a toolbar and light/dark toggle, without a sidebar. Select an output file to inspect its stored fragment trees. **Structure Detail** shows total nodes, edges, transitions, spectra, branch groups, physical-ion candidates, explanations, primitive reactions and teacher states. The tree and table display one entry per materialized fragment node, with unique node IDs. In-memory trees without assigned persistent IDs use their local node indices, indicated in the legend. Diamonds mark precursors; orange circles mark terminal branch nodes.

Click a node or its table entry to open a separate molecule panel. Later selections update the same panel. Both molecule views and the tree have zoom controls. Expand a node's **Reaction sets** to inspect alternative normalized histories. Reaction links select the corresponding entry in **Reaction Reference**, which provides reaction pattern, reactant/reaction and product molecule IDs, query-ordered Source atom maps, retained/discarded atom maps and changed/cut bonds. The mapped Original Source drawing and saved SMARTS definitions provide the reference for these IDs. Click an edge to inspect its source and target nodes, added reaction and parent/target reaction sets.


Each selected spectrum shows its ion adduct followed by the main adduct in parentheses,
for example `[M+H-H2O]+([M+H]+)`. Collision energy is rounded to one decimal place
using half-up rounding. Detected fragments use an orange outline with no fill;
precursors use a green diamond outline.

The **Peak Assignments** table contains one row per retained experimental peak,
including its m/z and intensity. Annotation cells remain empty for unmatched peaks.
Matches include fragment node IDs, SMILES, formula, theoretical m/z, mass error
in ppm, hydrogen shift relative to the main adduct, and ion adduct. Multiple
annotations stay in the same peak row. Intensities reflect the preparation
normalization setting.

Preparation saves two intensity-coverage scores for each sample: assigned peak
intensity divided by total peak intensity, and the same ratio after removing
peaks matching the precursor m/z within the configured mass tolerance from both
the numerator and denominator. Each peak contributes once even when it has
multiple matches. A zero denominator produces an unavailable score. Scores are
stored in the structure and in `assignment_scores.tsv` directly under each split
directory. Existing datasets without saved peak records must be regenerated
to display the peak table and both scores.


Open `train_structures/fragment-tree.pft` or
`validation_structures/fragment-tree.pft` to view that split independently.
These files also retain the preparation settings for import. The Python
`preparation_config.json` remains at the output root. A training-only CLI run
uses `train_structures` as well.

**Assignment Score Distribution** shows a histogram for the selected split.
Choose **All peaks** or **Without precursor peaks** and change **Bins** from 1
to 100. Scores of 100% belong to the final bin; unavailable scores are counted
separately. Hover over a bar to see its interval and sample count. Light mode
uses a white page background.


The Output Files table includes rejected sample counts, a rejection-log link
when records were skipped, and total stored node/edge counts across the samples
in each structure. Both assignment-score columns show the mean of available
per-sample scores; select the structure and a spectrum to see individual scores.
A zero count or score is displayed explicitly. Missing values in older
manifests are recovered from saved structures and assignment-score TSV files.
Workbench light mode uses white page, panel, toolbar and navigation surfaces.
