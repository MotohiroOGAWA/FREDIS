# Reaction Pattern Analysis

Reaction Pattern Analysis measures how many experimental MS/MS peaks FREDIS can
assign under cumulative combinations of reaction pattern sets, arranged as a
user-defined tree. It uses the normal Fragmenter and the same spectrum
readers, record validation and assignment chemistry as fragment-tree data
preparation.

## Pattern sets and the analysis tree

Every analysis unit is a **ReactionPatternSet**. A single reaction pattern is
used as a one-pattern set named after the pattern. Nothing is selected by
default: the patterns of the model/Fragmenter configuration are not used unless
they are added as a set.

The **Tree Configuration** keeps the pattern-set list and the tree apart, so one
set can appear at any number of nodes:

```json
{
  "patternSets": [
    {"id": "nonring_single_bond", "name": "nonring_single_bond", "patterns": [ ... ]},
    {"id": "mclafferty", "name": "mclafferty", "patterns": [ ... ]},
    {"id": "flavonoids", "name": "flavonoids", "patterns": [ ... ]}
  ],
  "nodes": [
    {"id": "n1", "patternSet": "nonring_single_bond", "parent": null},
    {"id": "n2", "patternSet": "mclafferty", "parent": "n1"},
    {"id": "n3", "patternSet": "flavonoids", "parent": "n1"},
    {"id": "n4", "patternSet": "mclafferty", "parent": null}
  ]
}
```

```text
nonring_single_bond            condition: nonring_single_bond
├── mclafferty                 condition: nonring_single_bond + mclafferty
└── flavonoids                 condition: nonring_single_bond + flavonoids
mclafferty                     condition: mclafferty            (another root)
```

- Several **roots** form independent series; depth is unlimited.
- A node's **condition** is the union of the pattern sets on its path:
  `A → B → C` analyses `A`, `A + B` and `A + B + C`.
- Every node with a parent is an **edge**: *parent condition → parent
  condition + child set*. A root has no parent and reports absolute metrics
  only; it is never treated as a difference from an empty set.
- Nodes that reach the same pattern union (`A → B` and `B → A`) share one
  condition, which is calculated once.

The CLI and Workbench both build this model; `tree.py` validates it (unknown
sets, cycles, duplicate ids, two different patterns with the same name) and
resolves stable pattern ids, conditions and node rows for the engine.

## What is calculated

```text
Input datasets
    ↓
Unique compounds              (canonical SMILES; all spectra of every dataset)
    ↓
Edge applicability            (precursor SMILES vs. the patterns each child set adds)
    ↓
Per-compound plan             (calculate · identical to parent · reuse from an existing result)
    ↓
One task per compound         (Source reactions once per pattern; conditions parents first)
    ↓
Node / edge comparison → sample rows → distributions
    ↓
A → B → C result files
```

### Unique compounds

Every valid spectrum is grouped by canonical SMILES, across datasets, adducts
and collision energies. Each group gets a stable `compound_id`
(`c` + the first 16 hex digits of SHA-256 of the canonical SMILES), which is
also used for file names. Each spectrum gets a `sample_id` and its position in
the compound (`sample_index`); a peak is identified by
`(sample_id, peak_index)`, never by its m/z.

### Fragmentation reactions are generated once

Source reactions are matched only on the original precursor. Each compound is
one worker task: the reactions of every pattern used anywhere in the tree are
enumerated **once per (compound, pattern)** and cached for the task, so a set
placed at several nodes is never matched again.

- A **root** condition is a full fragment-tree build from the union of its
  patterns' reactions.
- A **child** condition is seeded with the parent condition's reaction-sequence
  compounds below `max_reaction_count` and may only add reactions of the patterns
  the child set introduces. Every history containing a new reaction has a
  parent-only prefix shorter than `max_reaction_count`, so this equals a full
  build of the child condition; parent reactions are not rerun.
  `max_reaction_count` still limits every history.

### Edge applicability

An edge applies to a compound when its **precursor SMILES** matches the
`reactant_smarts` of a pattern the child set adds. Otherwise the child set adds
no reaction for that compound and the child condition *is* the parent condition:
nothing is calculated, no edge row is written, and the node inherits the
parent's assignments (`compound_condition.alias_of`). A matching edge is
calculated and saved even when it assigns no new peak (Δ = 0).

### Meaning of contribution

All annotations are binary per peak; multiple fragments or alternative paths
never count a peak twice (assignment counts are kept separately).

```text
assigned(condition)  = the condition's tree assigns the peak
                       (a child condition includes every parent assignment)
new(edge)            = assigned(child) AND NOT assigned(parent)
peak coverage        = assigned peak count / peak count
intensity coverage   = assigned peak intensity / total peak intensity
Δ metric (edge)      = child condition metric − parent condition metric
```

Absolute child metrics are always kept next to the deltas. **Without
precursor** removes peaks within mass tolerance of the theoretical precursor
m/z from numerator and denominator; every value is stored with and without the
precursor.

Coverage is first computed per sample (spectrum). A compound × node is then
summarized by the distribution of its sample values: n, **mean**, std
(population, ddof = 0), min, q25, median, q75, max, and a histogram (coverage:
0.0–1.0 in 20 bins; counts/intensities: 0–max in 20 bins). The *weighted*
values (totals ÷ totals) are stored as `weighted_*`. Node rows report the
compared spectra (edge: applicable compounds) and `overall_*` coverage of the
node's condition over every spectrum.

Only spectra retained by the builder enter the analysis, exactly as in
Training Data. A failed compound or condition never aborts the run; it is
recorded with `status` (`completed` / `skipped` / `failed`), `skip_category`,
`reason` and `error`. Children of a condition that did not complete are
`skipped` with `skip_category = parent_unavailable`.

## Adding nodes to an existing result

Recalculation is never needed to grow a tree. Run again with **extend**: the
result in the output directory is kept, every condition it already calculated
is reused, and only new conditions are calculated.

- Conditions are identified by the **content** of their patterns, so renaming
  sets or moving nodes reuses them; changing a set's patterns creates new
  conditions.
- Every saved tree keeps the reaction-sequence compounds that seed children, so a
  new child below any calculated node is built incrementally.
- Sources, chemistry configuration and limits must equal the existing result's
  (they default to it); otherwise the run stops and explains which settings
  differ.
- Pattern ids stay stable across runs, because saved reactions contain them.
- B is written to `compounds.new/` and swapped in together with A, so an
  interrupted extension leaves the previous result intact.

## Result format: A → B → C

```text
output/
├── result.reactionpatternanalysis.pft          A
├── compounds/
│   └── <compound_id>.peaks.parquet      B
├── details/conditions/<condition_key>/  C
│   └── <compound_id>.preft.pt
├── config.reaction-pattern-analysis.pft         run configuration (JSON, includes the tree)
├── tree.reaction-pattern-analysis.json          Tree Configuration (usable with --tree-config)
└── reaction-pattern-analysis-chemistry-config.json  resolved chemistry (Fragmenter and symbols)
```

| | File | Format | Role |
|---|---|---|---|
| **A** | `result.reactionpatternanalysis.pft` | standard HDF5 | tree, analysis-wide summaries, distributions and indexes |
| **B** | `compounds/*.peaks.parquet` | standard Parquet | one compound, all spectra, peak-level binary summary per condition |
| **C** | `details/conditions/**/*.preft.pt` | FREDIS `.preft.pt` | detailed fragment/reaction/assignment data per compound and condition |

All paths stored in A are relative to the result directory, so the whole
directory can be moved or copied to another machine.

### A: `result.reactionpatternanalysis.pft` (HDF5)

The extension is `.pft`, but the content is a plain HDF5 file (no wrapper,
no pickled objects). Each table is an HDF5 group with one 1-D
dataset per column; its `columns` and `row_count` attributes describe it.
Large tables are chunked and gzip-compressed.

| Path | One row per | Notes |
|---|---|---|
| root attrs, `/metadata` | — | `schema = fredis.reaction-pattern-analysis`, `schema_version = 3`, `result_format = hdf5-tree-v1`, `analysis_status`, `created_at`, `fredis_version`, `git_commit_sha`; `/metadata/overall_summary` (JSON) |
| `/configuration/json` | — | `run` (the Workbench Input / Run form), `tree` (Tree Configuration), `chemistryConfig` (Fragmenter with the analysis-wide pattern library, and symbols), columns, limits |
| `/datasets` | dataset | id, label, original path, column mapping, record counts |
| `/pattern_sets` | used pattern set | id, name, pattern ids, number of nodes using it |
| `/patterns` | pattern | stable `pattern_id`, name, reactant SMARTS, definition JSON, sets |
| `/conditions` | distinct condition | `condition_key` (content hash), pattern ids, label, C directory |
| `/nodes` | tree node (depth-first) | `node_row`, id, parent, set, depth, condition, added pattern ids, path and condition labels |
| `/compounds` | compound (sorted by id) | SMILES, counts, evaluated nodes, improving edges, new-peak union, B path, status |
| `/compound_node_summary` | compound × root, applicable compound × edge | status, totals, weighted and mean coverage (child, parent, Δ), new peaks/intensity, with/without precursor |
| `/compound_node_distribution` | compound × node × metric × with/without precursor | n, mean, std, min, q25, median, q75, max, `histogram_json` |
| `/node_summary` | node | applicable / calculated / skipped / failed / improved counts, weighted and mean coverage (child, parent, Δ), new peaks, overall coverage |
| `/node_distribution` | node × metric × with/without precursor | over all sample rows of the node |
| `/sample_summary` | evaluated spectrum | compound, dataset, record, sample index, adduct, CE, precursor m/z, peak/intensity totals |
| `/sample_node_summary` | spectrum × root, spectrum × applicable edge | assigned counts and coverage; parent, new and Δ values (NaN for roots) |
| `/compound_condition` | compound × condition | status, kept `sample_index` list, own C path, seed C path, `alias_of` |
| `/indexes/*` | — | per-compound row ranges and per-node row lists |

Distribution metrics: `peak_coverage`, `intensity_coverage` (all nodes) and
`parent_peak_coverage`, `parent_intensity_coverage`, `delta_peak_coverage`,
`delta_intensity_coverage`, `newly_assigned_peak_count`,
`newly_assigned_intensity` (edges).

A is written as `result.reactionpatternanalysis.pft.tmp` with
`analysis_status = running` and atomically renamed after it is marked
`completed`.

### B: `compounds/<compound_id>.peaks.parquet`

| Columns | Meaning |
|---|---|
| `sample_id`, `dataset_id`, `record_index`, `sample_index`, `peak_index` | identity |
| `compound_id`, `smiles`, `adduct`, `collision_energy`, `precursor_mz` | spectrum metadata |
| `mz`, `intensity`, `relative_intensity`, `is_precursor` | peak |
| `assigned::<condition_id>`, `count::<condition_id>` | per condition resolved for this compound |

An edge's newly assigned peaks are `assigned::<child> & ~assigned::<parent>`.
B never stores fragment structures, formula candidates or reaction paths.

### C: `details/conditions/<condition_key>/<compound_id>.preft.pt`

The same `.preft.pt` structure files as Training Data, one per compound and
calculated condition, with the saved reaction registry, per-sample annotations
and `reaction_sequence_compounds` (seeds for child conditions). A child tree
holds what its condition adds; peak details of a node combine the trees on its
path.

## Workbench

Open **Data → Reaction Pattern Analysis**. **Input / Run** has Datasets, Fragmenter,
**Pattern Sets**, **Analysis Tree** and Execution sections.

- **Pattern Sets**: add pattern or pattern-set files (Browse, Add, or
  drag-and-drop), or a configuration's patterns with *Add its patterns as a
  set*. **Edit** a set to view its patterns, remove patterns, add patterns from
  files, or rename it. Sets used by tree nodes cannot be deleted.
- **Analysis Tree**: **+ Add Root**, and per node **+ Add Child**, **Edit
  PatternSet** and **Delete Node** (with its subtree). Each node shows its set
  and the comparison it represents; a child whose patterns are all already on
  its path is flagged.
- **Execution**: *Add new nodes to the existing result* extends the result in
  the output directory. When a non-empty output directory holds a result,
  Workbench asks whether to add nodes to it or to delete and recalculate.

Opening a result restores **Input / Run** — sources, settings and the tree,
with calculated nodes marked — so nodes can be added and run without starting
over. Results are loaded lazily:

```text
open result       → A only (tree, node summary, compound list, score frequency)
select compound   → that compound's B Parquet only
select peak cell  → the C files on the node's path only
```

- **Overall Summary** and **Tree Edges**: the result tree and one sortable,
  filterable row per node — peak/intensity coverage, Δ coverage, parent
  coverage, mean Δ, applicable / calculated / improved compounds, new peaks,
  and overall coverage. Selecting a node shows its parent condition, child
  condition, added set, absolute and delta metrics and sample distribution, and
  restricts the compound list to it.
- **Data Filter / Compound Summary**: filters from A (SMILES, node, node and
  compound status, node coverage and Δ coverage (mean), dataset, adduct,
  collision energy, precursor m/z).
- **Score Frequency**: one series per node with its absolute score.
- **Compound × Tree Node**: per-node metrics; a row opens its distribution.
- **Compound / Peak View** (B): one column per node (**NEW** vs. its parent,
  ✓, ✗, or — when not evaluated) with peak filters.
- **Peak / Node Detail** and **Path / MS2 Depth Analysis** (C).

## CLI

```bash
python -m fredis.cli analysis reaction-pattern \
  --source-spec '{"path":"nist-positive.msds","kind":"dataset","columns":{"smiles":"SMILES","adduct":"AdductType","collision_energy":"CollisionEnergy","precursor_mz":"PrecursorMZ"}}' \
  --output-dir results/reaction-pattern-analysis \
  --params single_bond_cleavage_pos.fragmenter.pft \
  --tree-config tree.json \
  --max-reaction-count 3 --mass-tolerance "20 ppm" \
  --num-workers 8 --chunk-size 4
```

`--tree-config-json '<json>'` passes the same Tree Configuration inline, so the
command needs no external file (Workbench runs and **Copy command** use it).
The tree can also be given option by option, merged with either:

```bash
  --pattern-set nonring_single_bond.reactionset.pft \
  --pattern-set mclafferty=McLafferty.reaction.pft \
  --node n1=nonring_single_bond \
  --node n2=mclafferty@n1
```

`--pattern-set [ID=]FILE` adds a pattern-set file, or a single pattern as a
one-pattern set; `--node ID=SET[@PARENT]` places a set as a root or below
`PARENT`. Add nodes to an existing result with `--extend 1`; sources, model
configuration, limits and tree default to the existing result's, and
`--pattern-set` / `--node` add to its tree:

```bash
python -m fredis.cli analysis reaction-pattern --output-dir results/reaction-pattern-analysis --extend 1 \
  --pattern-set flavonoid.reactionset.pft --node n5=flavonoid@n1
```

Standard error shows the ten phases with `tqdm` counters. Standard output ends
with `{"ok": true, "result": "/path/to/result.reactionpatternanalysis.pft"}`.

## Python API

```python
from fredis.analysis.reaction_pattern import ReactionPatternAnalysisResult, generate_reaction_pattern_analysis

with ReactionPatternAnalysisResult("output/result.reactionpatternanalysis.pft") as result:
    result.node_summary()                                   # A: one row per node / edge
    result.compound_list({"nodeRow": 1}, sort="mean_delta_peak_coverage", descending=True)
    result.compound_node_distribution(compound_id, 1)
    result.compound_peaks(compound_id)                      # B
    result.peak_detail(compound_id, sample_id, peak_index, 1)  # C
```
