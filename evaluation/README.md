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
