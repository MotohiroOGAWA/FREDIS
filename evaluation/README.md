# FREDIS Evaluation

Column-oriented similarity evaluation used by FREDIS Workbench. An msentity
`.mssim` file is joined to one `.msds` (or CSV, TSV, JSON, JSONL/NDJSON, or
Parquet) metadata source. The join column is configurable and defaults to
`SpecID` when that column exists in both files.

```bash
python -m evaluation inspect --input result.mssim --metadata observed.msds
python -m evaluation summarize --input result.mssim --metadata observed.msds \
  --join-column SourceSpecID --metadata-join-column SpecID \
  --group-column AdductType --output-image adduct-boxplot.svg
python -m evaluation summarize --input result.mssim --metadata observed.msds \
  --join-column IDENTIFIER --group-column CollisionEnergy \
  --transform collision-energy --precursor-mz-column PrecursorMZ \
  --mode numeric --bins 0,10,20 --output-image collision-energy-boxplot.svg
python -m evaluation summarize --input result.mssim --metadata observed.msds \
  --group-column SMILES --transform chemical --smiles-column SMILES \
  --chemical-descriptor HeavyAtomCount --mode numeric --bins 0,10,20 \
  --output-image heavy-atom-count-boxplot.svg
python -m evaluation summarize --input result.mssim --metadata observed.msds \
  --group-column SMILES --transform smarts --smiles-column SMILES \
  --smarts-json '[{"label":"Alcohol","smarts":"[OX2H]"},{"label":"Aromatic","smarts":"[c]"}]' \
  --output-image smarts-boxplot.svg
```

Each group is represented as a Tukey box plot of the `.mssim`
`cosine_similarity` values. SVG output is transparent by default. Use
`--opaque` for a white background and `--width`, `--height`, and `--color` to
customize its appearance.
The result and metadata join keys may have different names. When neither is
specified, Evaluation first tries same-name `SourceSpecID`, then
`.mssim.SourceSpecID` to metadata `SpecID`, then same-name `SpecID`, then
same-name `IDENTIFIER`. Metadata rows whose join key is missing are ignored.
Both column and grouped SVGs also support graph opacity independently of the
background, plus separate x-axis label, y-axis label, and title font sizes.
The CLI options are `--graph-opacity`, `--x-label-size`, `--y-label-size`, and
`--title-size`; the same values are editable and persisted by the Workbench.

Column values can be left unchanged, parsed as collision energy using FREDIS'
existing CE-to-eV parser, or derived from SMILES using the supported molecular
descriptors. The Workbench keeps edits as a draft and does not recalculate the
chart until **Generate box plot** is pressed.

SMARTS grouping evaluates every query independently. A spectrum can therefore
contribute to more than one group when its compound matches multiple queries.
Alongside the score distribution, the result reports matching spectrum rows
and unique canonical compounds for each SMARTS. In the Workbench, enter one
query per line as `label = SMARTS` (with spaces around `=`); a line containing
only SMARTS uses the SMARTS itself as its label. A bond expression such as
`C(=O)` is not treated as a separator because it has no surrounding spaces.

## Grouped multi-tool comparison

`compare` creates a conventional vertical box plot from multiple `.mssim`
files. Groups define the x-axis (for example, databases), while series define
tools and retain the same color across every group. A group/series pair can be
marked explicitly as `noData` when the tool produced no result.

```json
{
  "title": "MS/MS similarity comparison",
  "groups": [
    {"id": "mona", "name": "MoNA"},
    {"id": "massbank", "name": "MassBank"}
  ],
  "series": [
    {"id": "fredis", "name": "FREDIS", "color": "#36c5a2"},
    {"id": "fiora", "name": "FIORA", "color": "#e67722"}
  ],
  "entries": [
    {"groupId": "mona", "seriesId": "fredis", "path": "mona-fredis.mssim"},
    {"groupId": "mona", "seriesId": "fiora", "noData": true},
    {"groupId": "massbank", "seriesId": "fredis", "path": "massbank-fredis.mssim"},
    {"groupId": "massbank", "seriesId": "fiora", "path": "massbank-fiora.mssim"}
  ],
  "width": 1200,
  "height": 650,
  "transparent": true
}
```

```bash
python -m evaluation compare --config comparison.json \
  --output-image comparison.svg
```

The Evaluation window exposes the same feature under **Grouped Box Plot**.
Groups and series can be added, renamed, removed, and reordered independently.
The image controls separately configure the gap between groups, the much
smaller gap between series inside a group, and box width. Box width `0` is
automatic and expands the boxes when the image width increases; a positive
pixel value fixes the width and centers the tightly packed boxes in each group.

## Editable settings

Both **Column Evaluation** and **Grouped Box Plot** provide **Load Settings**
and **Save Settings** actions. The resulting JSON stores input paths, grouping
and ordering, enabled categories, series colors, no-data selections, and image
settings. Loading it restores the form instead of only restoring a generated
image, so the evaluation can be adjusted and run again.

Evaluation setting files use the versioned envelope below. New Evaluation
views should use the same envelope with their own `kind` and editable `config`
payload.
Grouped settings saved by the Workbench can also be passed directly to
`python -m evaluation compare --config`.

```json
{
  "schema": "fredis.evaluation.config",
  "schemaVersion": 1,
  "kind": "column",
  "config": {}
}
```

### Training Data Filtering (first Evaluation tab)

Specify metadata (`.msds`, CSV, TSV, JSON, JSONL, or Parquet), load its
columns, and select the SpecID and classification columns (for example `Class`,
`Superclass`, or `Subclass`). Add or remove tools with a name, a text-file path
(one SpecID per line), and a color. Evaluate shows overall and per-category
retention counts, percentages, exclusions, and a PNG chart.

All percentages use the metadata SpecIDs in that category as denominator.
`Common` is the intersection across every configured tool. Missing categories
appear as `(Missing)`. Metadata IDs must be nonempty and unique; duplicate IDs
in tool lists are counted once. Diagnostics report duplicate list entries and
IDs absent from metadata. This audits retention patterns, not the reasons each
tool excluded a spectrum.

Click Evaluate and choose a directory to save the evaluation. This writes
`results.json` (all computed counts) and `settings.evaltrainable.pft`, whose
`resultPath` is relative to the settings file. Moving the directory keeps this
link valid. Opening the settings file loads the cached counts automatically,
without reading metadata or tool ID lists. If the result is missing or unreadable,
the settings remain available and the UI reports the problem.

Save Settings writes only the configuration file, display options, and a relative
reference to the existing result, when available. It does not calculate or copy
results. Save settings again after changing display options to preserve them.

Result display has Graph and Data list pages. The list can be filtered by
metadata, N, per-tool retention/counts, or all-tool statistics. Filtering only
narrows the list; it never changes which categories are checked for the graph.
Each list row starts with a Show checkbox, category name, original N, and a
small per-tool retention distribution. Select/Unselect all in filtered list only
changes currently visible rows; selections hidden by a filter remain intact.
The checkbox selection and sort order drive graphs and exports. Overall has a
separate visibility checkbox. Settings preserve selections, filters, sort
priorities, and the active page.

The default chart places categories side by side on the x-axis with grouped
vertical bars; horizontal bars remain available. Common's color can be changed
with its palette or HEX input. Charts expand as the number of categories grows,
keeping readable bar/label sizes in a scrollable viewport rather than shrinking
the entire chart to fit the page width.

Sort priorities can directly reference any metadata column independently of the
Category column: for example Kingdom ascending, Superclass ascending, then N
descending. There is no metadata-column preselection. New sort/filter columns
are loaded from the metadata source when needed and added to the saved result
cache; tool lists and retention calculations are not rerun. Cached columns work
without the original source. Loading a new column requires the source and
verifies its SpecIDs and category assignments against the evaluation fingerprint.
For older caches, use Load columns to refresh the list of available column names.

Sort rules apply from top to bottom and can be added, removed, or reordered.
A header click replaces the sort order; Shift-click appends a priority. Filters
combine with AND. Apply display refreshes the list and the checked graph.
Fields include original N, each tool's retained/excluded N and retention rate,
and the minimum/maximum/mean/median rate across actual tools. These are retention
percentages, not prediction accuracy. All-tool statistics exclude Common and
include hidden tools. Common remains the intersection of all evaluated tools.
Overall stays first when shown. Missing sort values go last; tied priorities
use category names as a stable tie-breaker. Multiple metadata values per category
appear joined with ` | `.

Export TSV, Export PNG, and Export SVG independently write the selected view to
a chosen directory. TSV writes per-tool `retention.tsv`, category-oriented
`categories.tsv` (checked categories with Data list columns), and full `id_diagnostics.tsv`.
PNG writes `retention.png`; SVG writes `retention.svg`. Exports never read source
data or recalculate retention. Both graph formats preserve category order,
selection, chart orientation, and colors; original denominators remain unchanged.

The CLI exposes the same calculation:

```sh
python -m evaluation filtering --request-json '{"metadata":"source.msds","idColumn":"SpecID","groupColumn":"Class","tools":[{"name":"FREDIS","path":"trainable_spec_ids.txt","color":"#36c5a2"}]}' --output-dir audit
```

Rendering and export can also use a cached result through `filtering-view`:

```sh
python -m evaluation filtering-view --request-json '{"result":{"rows":[],"diagnostics":[],"series":[]},"view":{"showOverall":false}}' --output-dir figures --format png
```

