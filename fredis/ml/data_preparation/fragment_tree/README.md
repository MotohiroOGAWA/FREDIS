# Preparing fragment-tree training data

This directory turns experimental MS/MS spectra into `.preft.pt` training
structures for the fragment-tree model. Preparation is the **chemistry
boundary** of the project: it parses molecules, enumerates reactions,
runs RDKit, annotates peaks with reaction combinations and stores everything the
model needs as tensors. Training loads these tensors directly and never calls
RDKit.

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input train.msds --validation-input validation.msds \
  --output-dir prepared --params single_bond_cleavage_pos.fragmenter.pft \
  --num-workers 4 --chunk-size 4 \
  --max-unique-fragment-smiles 1000 --max-reaction-combinations 50000 \
  --overwrite 1
```

Contents:

1. [Pipeline overview](#pipeline-overview)
2. [Input dataset](#input-dataset)
3. [Train / validation / test split](#train--validation--test-split)
4. [Processing unit: one unique SMILES](#processing-unit-one-unique-smiles)
5. [Search limits](#search-limits)
6. [Multiprocessing](#multiprocessing)
7. [Building the fragment tree](#building-the-fragment-tree)
8. [Ions, precursor and peak assignment](#ions-precursor-and-peak-assignment)
9. [Teacher structure](#teacher-structure)
10. [Skip conditions](#skip-conditions)
11. [Output files](#output-files)
12. [`manifest.tsv`](#manifesttsv)
13. [`reaction_statistics.json`](#reaction_statisticsjson)
14. [Statistics reference](#statistics-reference)
15. [Code map](#code-map)

---

## Pipeline overview

For every unique source SMILES:

```text
records with that SMILES (all adducts and collision energies)
  │
  ├─ reaction generation      SMARTS patterns matched once on the source
  ├─ reaction combination search       pure enumeration by increasing depth ──┐ max_reaction_combinations
  │    └─ pruning                     conflicts, invalidation, duplicate sets │
  ├─ RDKit materialization           one reaction per unique effect      │
  │    └─ canonical SMILES            unique fragment set ────────────────┘ max_unique_fragment_smiles
  ├─ fragment tree                    nodes merged by canonical SMILES
  ├─ fragment ion tree                hydrogen shifts, fragment-ion adducts, formula candidates
  ├─ precursor resolution             records with an unreachable precursor are dropped
  ├─ peak annotation                  peaks → formulas → reaction combinations on the source
  ├─ teacher structure                teacher states, positive / weak-negative reactions
  ├─ materialized teacher trees       ion-condition states and intensity targets
  └─ data/<stem>.preft.pt + one manifest row
```

Combination enumeration and RDKit materialization are separate phases. Enumerate
reaction ID sets in increasing depth, without compiling or running reactions.
If the cumulative combination budget is exceeded partway through depth n,
discard that entire depth and all deeper levels. Only the fully enumerated
levels through n - 1 are materialized. Overflow in depth 1 retains Source only.
There is no combination-search restart or replay of retained reactions.
The structure-count limit is checked later, during materialization.

## Input dataset

`--input` (and optionally `--validation-input`) accepts `.msds`, `.msp`,
`.mgf`, `.tsv`, `.csv` and `.parquet`. Table formats need a `Peak` column in
msentity spectrum-table syntax.

Required metadata columns, renamable with `--smiles-column`,
`--adduct-type-column`, `--collision-energy-column` and
`--precursor-mz-column` (and `--validation-*-column` for a validation file
with different names):

| Column (default) | Check |
|---|---|
| `SMILES` | Parsed by RDKit. Every atom must be in the configured element `symbols` (plus H). |
| `AdductType` | Parsed, charge ±1, and contains exactly one main adduct registered in the fragmenter. |
| `CollisionEnergy` | Converts to a finite, non-negative eV value (`NCE=…%` uses precursor m/z). |
| `PrecursorMZ` | Finite and positive. |

A record that fails any check is **excluded**, not fatal. Excluded records are
listed with their original index, id (`SpecID` if present) and reasons in
`invalid_records.json`. A missing column, or an input with no usable records,
stops the run.

Peaks are then preprocessed per record: peaks below
`--minimum-relative-intensity` × the spectrum's maximum are dropped, and the
remaining intensities are scaled so that the maximum is 1.

## Train / validation / test split

Workbench uses a **dataset list**. Add any number of files; each card has its
own column mapping, file statistics, dataset summary, validation progress and
preview. The summary heading includes its assigned role in assignment mode.

- **Assign each dataset**: select Train, Validation or Test using the radios on
  each card. Files with the same role are combined, and at least one Train
  dataset is required. No ratio-based cap or automatic holdout is applied.
  Canonical molecules shared across different roles cause an error before
  preparation; overlapping spectra within one role are retained.
- **Combine and split**: combine all files, ignoring their remembered roles,
  then split the union by unique RDKit canonical SMILES. Duplicate molecules
  across files stay in one split; spectra are retained. Validation and Test
  fractions and the seed apply only to this mode.

**Check Split Summary** on the input step and **Preflight Check** on the run
step use the same plan as generation and show separate Train / Validation /
Test molecule and spectrum counts. Each card's values can also be checked
individually as part of **Validate Dataset Values**.

The CLI accepts the same list using `--datasets-json` and `--dataset-mode`:

```sh
python -m fredis.cli train create-fragment-tree-data \
  --datasets-json '[{"path":"a.msds","split":"train"},{"path":"b.msds","split":"validation"},{"path":"c.msds","split":"test"}]' \
  --dataset-mode assigned --output-dir prepared \
  --params fredis/presets/fragment/single_bond_cleavage_pos.fragmenter.pft
```

Each list entry can include `id` and `mapping` with `smilesColumn`,
`adductTypeColumn`, `collisionEnergyColumn`, and `precursorMzColumn`.
`--dataset-mode automatic --validation-ratio 0.1 --test-ratio 0.1` instead
splits the union approximately 80 / 10 / 10. Paths and per-file mappings are
saved in preparation settings; original record positions and source paths
are retained as metadata. Old single-input configurations open as a list;
a separate validation input becomes an explicitly assigned Validation item.
Legacy CLI flags below remain supported.


Splits use **RDKit canonical isomeric SMILES**. Different SMILES spellings of
one molecule stay together, across adducts and collision energies. Atom-map
annotations are ignored; stereochemistry, isotopes and charge are preserved.
This does not normalize tautomers or remove salts. Original input SMILES and
spectra are retained.

| Options | Behaviour |
|---|---|
| neither ratio | Training only; structures are written to `train_structures/`. |
| `--validation-ratio r` | Reserve a fraction of input canonical molecules for validation. |
| `--test-ratio t` | Reserve a fraction of input canonical molecules for final testing (default 0). |
| both ratios | Both fractions refer to the original input molecule count; train receives the remainder. For example, `--validation-ratio 0.1 --test-ratio 0.1` gives approximately 80 / 10 / 10. |
| `--validation-input` | Use a separate validation dataset. CLI removes molecules also present in the original input, including those assigned to test. Workbench preflight rejects overlapping inputs. |
| `--validation-input` + `--validation-ratio r` | Cap validation at `max(1, round(original input canonical molecule count × r))` molecules after overlap removal. A zero or omitted ratio means no cap. Test is still taken from the original input. |

Ratios must be finite and in `[0, 1)`; automatic validation and test ratios
must sum to less than 1. Zero disables an automatic split. Each enabled split
gets at least one molecule; insufficient distinct canonical molecules is an
error. Counts use Python `round(n * ratio)`; if rounding leaves no training
molecule, the largest held-out count is reduced (validation first on a tie).

Canonical keys are sorted before shuffling with `--validation-seed` (default 0),
so the assignment is reproducible even if input records are reordered.
`split_manifest.json` saves the canonical molecule-to-split mapping and seed.
Settings, including `testRatio`, are restored from each split's
`fragment-tree.training-data.pft`.

Workbench's **Preflight Check** reports molecule and spectrum counts for each
split after excluding invalid metadata, using the same assignment logic as
preparation. Counts precede structure generation; molecules skipped during
generation are reported by each split's existing statistics.

Outputs are `train_structures/`, `validation_structures/` (when enabled), and
`test_structures/` (when enabled). Train and validation remain the inputs to
the training workflow. Reserve test for evaluation of the selected final
model; it is not automatically passed to training or model selection.

Overlap removal and shortfalls are reported as JSON events on stdout:
`validation_overlap_removed`, `validation_below_target`, `validation_empty`.
If separate validation ends up empty, CLI continues with training only.

### Check and export split spectrum datasets

**Check Split Summary** displays progress for each input's loading and record
validation, followed by canonical molecule splitting. On success,
**Export Split Datasets…** lets you select a destination directory and writes
`train.msds`, `validation.msds`, and `test.msds` there. A split with zero records
is omitted. `dataset_split_manifest.json` records the split assignments and
source information. Existing split files are never overwritten.

The export reuses the checked molecule assignments and eligible record indexes;
it does not repeat metadata validation. If a source file changes after checking,
check the split again. Editing the input list or configuration disables export
until the new configuration has been checked. Checked plans are temporary and
are removed when the Workbench closes. Both checking and exporting use the same
CLI command as standalone workflows:

```sh
python -m fredis.cli train split-datasets \
  --datasets-json '[{"path":"a.msds"},{"path":"b.msds"}]' \
  --dataset-mode automatic --validation-ratio 0.1 --test-ratio 0.1 --seed 7 \
  --params fredis/presets/fragment/single_bond_cleavage_pos.fragmenter.pft \
  --output-dir split-datasets
```

Use `--dataset-mode assigned` and each entry's `split` role to keep explicit
assignments instead. Per-file `mapping` works as in data preparation.
For separate checking and export, replace `--output-dir` with
`--plan-output checked-split.json`, then run:

```sh
python -m fredis.cli train split-datasets \
  --checked-plan checked-split.json --output-dir split-datasets
```

The CLI emits machine-readable `FREDIS_PROGRESS` events to stderr and one JSON
result to stdout. `--config-stdin` accepts the Workbench configuration directly.

## Processing unit: one unique SMILES

Records are grouped by their **exact SMILES string**. There is no
canonicalization, so two different strings for the same molecule become two
sources. One group becomes one *source*: one fragment tree, one `.preft.pt`
file and one `manifest.tsv` row. Collision energies and adducts of that
molecule share the source.

Inside a source, records are split into **branch groups** by normalized main
adduct (for example `[M+H]+` and `[M+Na]+`). Collision energy is only a
per-sample condition feature, so a reaction observed at one energy is never a
negative at another energy of the same branch group.

## Search limits

Two per-source limits bound the cost of one molecule. Both default to `-1`
(unlimited); otherwise they must be positive integers. A source that exceeds
either one is **skipped as a whole**. Its data is never truncated.

### `max_unique_fragment_smiles`

CLI `--max-unique-fragment-smiles`, Workbench **Max Unique Fragment Smiles**
(`maxUniqueFragmentSmiles` in saved configurations).

- Counts distinct canonical SMILES of materialized fragments. **The source
  molecule itself is not counted**, even if a fragment happens to equal it.
- It is checked every time RDKit generates a fragment. When the count reaches
  `limit + 1`, the source stops at once: no further combination is requested
  and no further reaction runs.
- It does not count duplicates: two histories producing the same SMILES
  count once.
- A value equal to a source's own unlimited `num_unique_fragment_smiles` keeps
  that source's tree identical; one less skips it.

### `max_reaction_combinations`

CLI `--max-reaction-combinations`, Workbench **Max Reaction Combinations**
(`maxReactionCombinations`).

- Bounds `ReactionSequenceSearch.stats.num_raw_combinations`: every
  (parent, candidate reaction) pair examined, over all combination sizes from 1
  to `max_reaction_count`, including pairs that pruning then rejects.
- It is checked inside `ReactionSequenceSearch`, as soon as the count reaches
  `limit + 1` and before that combination is yielded. The combination that
  crosses the limit never reaches RDKit. No candidate from that incomplete
  depth is materialized, including candidates found before the overflow.
- Retained search counters describe only the complete lower levels. The observed
  overflow counters (including limit + 1) are recorded in `limit_attempts`;
  `completed_max_reaction_count` records the retained depth.
- This is the cheaper limit: it stops molecules with many primitive reactions
  before they spend time in RDKit.

### Choosing values

Run once with unlimited values, or generous ones, and read the maxima in
`reaction_statistics.json` together with the per-source columns in
`manifest.tsv`. For example, `max_raw_combinations` and
`max_raw_combinations_smiles` show the most expensive prepared molecule.

Both limits are CLI options only; a `--params` file holds chemistry, not limits.

## Multiprocessing

With `--num-workers 1` (default), or fewer than two sources, sources are
prepared serially in the main process.

Otherwise:

- Sources are sent in chunks of `--chunk-size` SMILES groups, capped at
  `groups // workers` so every worker has work.
- Chunks run in subprocesses (`subprocess_worker.py`) through
  `fredis.utils.parallel_subprocess.run_parallel_subprocesses`. Each subprocess
  builds its own RDKit/fragmenter context.
- At most `num_workers` chunk files exist at once, in a temporary
  `fredis-preparation-*` directory inside the split directory. Each is deleted
  as soon as its result is read.
- Results are collected as chunks finish.
  - Output file names come from each source's index, so they match serial
    preparation.
  - Maxima ties resolve to the smallest SMILES, so `reaction_statistics.json`
    does not depend on completion order.
  - The rows of `manifest.tsv` are sorted by SMILES.
- If a worker process dies, its chunk is re-run one source per process, and
  only a source that still crashes is skipped (`worker_crash`).
- Skip records, including limit skips and their partial statistics, travel
  back from the workers exactly as in serial mode.

## Building the fragment tree

### Reaction generation

Each reaction pattern's reactant SMARTS is matched **once** on the atom-mapped
source (`GetSubstructMatches(uniquify=False)`). Every match × product template
becomes a primitive `ReactionInstance`. A reaction records:

- the atoms it retains and discards,
- the bonds it cuts and any bond-order updates,
- the matched atoms and bonds it depends on.

Templates that would create atoms or bonds, or that change aromatic bonds, are
unsupported. No-op reactions and reactions that retain nothing are discarded. The
number of primitive reactions is `num_primitive_reactions`.

### Reaction combination search

`ReactionSequenceSearch` enumerates unordered combinations of primitive reactions
with at most `max_reaction_count` reactions (from the fragmenter's
`fragment_ion_tree_builder.max_reaction_count`). A priority queue expands parents
in order of increasing normalized reaction count, and each child adds a primitive
reaction with a larger index than the parent's cursor.

Every (parent, candidate reaction) pair examined counts as one raw combination
(`num_raw_combinations`). This count covers all sizes: with
`max_reaction_count=3`, it includes 1-reaction, 2-reaction and 3-reaction
combinations.

### Retaining complete reaction-count levels

During training data preparation, reaching a search limit retains the largest reaction count that can be enumerated exhaustively within the limit. For example, if the search exceeds a limit at reaction count 3, all results from count 3 and above are discarded, and annotation uses the complete tree through count 2. If even count 1 cannot be completed, annotation uses count 0 (Source only). Spectra whose precursors cannot be resolved in the retained tree are still excluded.

`manifest.tsv` and `.preft.pt` metadata under `metadata.search_stats` record `requested_max_reaction_count`, `completed_max_reaction_count`, `search_truncated`, and `discarded_from_reaction_count`. The saved metadata also includes `limit_attempts`, containing each failed attempt's reaction-count bound, limit name, limit value, and observed counters. Ordinary search counters describe the complete retained tree.

The same behavior applies to trainable SpecID verification, which uses `ReactionStructureBuilder`. Combination enumeration runs once: overflow at depth n retains complete levels through n - 1 before any RDKit execution. The separate unique-structure limit can still require a lower-depth materialization retry, because unique structures are known only after execution. If existing seeds exceed the retained bound, the source is excluded with a recorded reason rather than silently dropping those seeds.

### Pruning

Most raw combinations never reach RDKit:

| Rule | Counter | Meaning |
|---|---|---|
| Hard conflict | `num_hard_conflict_pruned` | Two reactions change the same bond. |
| Invalidated reaction | `num_invalidated_reaction_pruned` | One reaction discards atoms that another reaction's match needs. Reactions are simultaneous, so no order can fix this. |
| Duplicate sequence | `num_duplicate_sequence_pruned` | The normalized history equals its parent (skipped), or it was already visited. An already-visited history is still yielded once per new parent, as an alternative route. |
| Duplicate effect | `num_duplicate_effect_pruned` | A different history has the same net effect (retained atoms, cuts, bond updates) as one already materialized, so the cached product is reused without running RDKit. |

Combinations that retain no atoms, or that normalize to more than
`max_reaction_count` reactions, are skipped without a counter.

### RDKit materialization

Each new effect is compiled into a source-specific reaction and run once:
`num_compiled_sequences` and `num_rdkit_run_reactants` count these.

- If the reaction gives exactly one product, it counts toward
  `num_generated_fragments`.
- If sanitization fails (for example a valence violation), the effect is
  cached as invalid and every history with that effect is dropped silently.
- More than one product is an error, and the source is skipped.

### Unique fragments and tree nodes

After each successful materialization, the product's **canonical SMILES**
(without atom maps) is added to a set that **excludes the source SMILES**. The
set's size is `num_unique_fragment_smiles`. The `max_unique_fragment_smiles`
limit is checked right here, immediately after each new SMILES.

Tree nodes are merged by the same canonical SMILES; node 0 is the source.
Edges connect nodes and carry every distinct reaction transition between them.

- Every distinct reaction history that reaches a node is stored as a
  transition on its incoming edge.
- A materialized fragment whose parent history never produced a node (and
  which has no alternative predecessor) is not linked into the tree. It still
  counts in the statistics.

In the common case, tree nodes = 1 + `num_unique_fragment_smiles`.

## Ions, precursor and peak assignment

### Ion and formula candidates

`FragmentIonTreeBuilder` extends the tree with ion chemistry from the
fragmenter's adduct rule set:

- **Hydrogen shifts:** every hydrogen shift ΔH from the rule set's
  `hydrogen_shift_min` to `hydrogen_shift_max`.
- **Fragment-ion adducts:** one per `fragment_ion_adducts` entry of the adduct
  rule. They apply to every neutral node; charged nodes get none.
- **Formula candidates:** per main adduct and for each node × hydrogen shift
  × fragment-ion adduct, the ion formula and its exact mass, sorted by mass.

### Precursor resolution

For each record, the precursor formula is its adduct applied to the source
formula. A tree node can be the precursor if its formula plus a hydrogen shift plus the
main adduct equals the precursor formula. Every reaction combination that
reaches such a node with at most `precursor_candidate_max_reaction_count`
reactions (default 0, which means only the source itself) is a precursor
combination.

- Records with no such precursor are dropped. The manifest reports them as
  `rejected_sample_count`, with the reason `N of M records dropped: unresolved
  precursor reaction sequence`.
- If no record of the source is left, the source is skipped
  (`unresolved_precursor`).

### Peak annotation

Per main adduct, every peak is matched to every formula candidate within
`mass_tolerance`. A matched candidate (node, fragment-ion adduct, hydrogen
shift) is explained by every reaction combination that reaches the node and
contains a precursor combination. Reactions in a combination are applied to
the source simultaneously, so annotations carry no order. Per peak only the
explanations with the **fewest reactions** are kept.

A peak is flagged as the precursor peak if it is within tolerance of the
*theoretical* precursor m/z, not the dataset's `PrecursorMZ`. Per sample:

- `assignmentScore` = matched intensity / total intensity.
- `assignmentScoreWithoutPrecursor` = the same ratio without precursor peaks.

## Teacher structure

Each observed peak can have several assigned formulas, Compounds and reaction
paths. Preparation first builds the supported paths with
`prepare_source_reactions`. It materializes the chemistry and peak annotations,
then `SourceReactionStructure.from_preparation()` packs the assigned paths into
[BranchTeacher](../../input/SCHEMA.md#branchteacher).

BranchTeacher stores exact reaction-history transitions, the target fragment
endpoints for each peak, and peak/depth positive and negative memberships.
Intermediate states remain supported when a later step explains the peak.
A transition supporting any observed peak in the Source/main-adduct group is
excluded from every negative bag in that group. Terminal negative alternatives
occupy the next depth. Root-only negative units remain possible for groups
without assigned fragment paths. Paths share molecular structures and neural
state features through integer references.

Numerical class weights, temperature and depth/peak normalization are computed
by the training loss. At depth two the positive score includes both steps'
log probabilities; independent one-step successes cannot replace a connected
path. Training reads these memberships directly without rebuilding teachers.

Spectrum supervision remains separate: measurements and peak assignments are
saved in `annotations`, and `targets` stores the maximum assigned measured
intensity per (spectrum, Formula). Unobserved valid candidates have zero targets;
invalid candidates are masked. Spectrum's cross-entropy and cosine objectives
are unchanged. Descriptors are computed by the model when configured.

The saved payload uses `fredis.annotated-spectrum-training`, without a version.
It contains shared Source/Reaction/ReactionSet/Compound tables, chemical graphs,
adduct/formula explanations, spectrum candidates, observations, assignments,
BranchTeacher and spectrum targets. Regenerate data when this contract changes.

### Candidate graph storage

Default: save positive and weak-negative edges and their endpoint nodes, plus
source roots. CLI `--save-all-edges` / Workbench `saveAllEdges: true` additionally
saves ignored edges and candidate nodes. The Workbench checkbox defaults to off
and is persisted in preparation configurations. Both modes use identical branch
supervision; candidate coverage for spectrum prediction may differ.

## Skip conditions

A failure in one source never stops the run. The source is recorded as skipped
and preparation continues, in serial and in parallel mode alike.

| `category` | Cause | `reason` example |
|---|---|---|
| `limit_exceeded` | A search limit was exceeded. `limit` names it. | `max_reaction_combinations exceeded: limit=50000, observed>50000` |
| `unresolved_precursor` | No record of the source has a reachable precursor. | `No sample in this group has a valid precursor reaction sequence from Source` |
| `error` | Any other exception while preparing the source, for example an unsupported element or invalid peak values. `error` holds the traceback. | `ValueError: Unsupported atom symbol: 'As' ...` |
| `worker_crash` | The worker process died (native crash, OOM kill) even when the source was run on its own. `error` holds the stderr tail. | `Worker process exited with status -9` |

Limit skips are deliberate filtering, so they are kept separate from errors in
`category` / `skip_category`, in `num_skipped_sources_by_category`, and in
`num_limit_skipped_sources`.

A skipped source keeps the search statistics observed **up to the moment it
stopped**:

- For a limit skip, they are lower bounds, and the limited value is exactly
  `limit + 1`.
- For `unresolved_precursor`, and for errors raised after the tree was built,
  they are the full statistics.
- For `worker_crash`, and for errors raised before the tree search, they are
  empty.

Record-level exclusions (invalid metadata, see [Input dataset](#input-dataset))
happen before grouping and are reported in `invalid_records.json`, not as
skipped sources.

## Output files

```text
<output-dir>/
  preparation_config.json      resolved CLI settings, chemistry_config and Workbench form state
  invalid_records.json         excluded records per input: {"train": [...], "validation": [...]}
  train_structures/
    fragment-tree.training-data.pft          Workbench configuration of this split (status: running → completed)
    data/smiles_<index:06d>_<sha1[:16]>.preft.pt
    manifest.tsv               one row per source (completed or skipped)
    assignment_scores.tsv      one row per kept sample
    skipped_sources.json       one entry per skipped source
    reaction_statistics.json     split-level counts, search maxima, limits, configuration
    invalid_records.json       re-inspection of this split (normally [])
  validation_structures/       same layout, when validation exists
```

- A `.preft.pt` file holds the structure plus `metadata = {smiles,
  record_indexes, sample_annotations}`.
- `record_indexes` are positions within the split's metadata-valid dataset.
- If the output directory exists, the CLI requires `--overwrite 1` (or
  interactive confirmation). With overwrite, the whole directory is cleared
  first. It refuses to clear a directory that contains the inputs, the
  parameter file or the application.

`skipped_sources.json` entries:

```json
{
  "source_index": 13,
  "smiles": "CCC(C)C1NC(=O)...",
  "record_indexes": [13],
  "reason": "max_unique_fragment_smiles exceeded: limit=60, observed>60",
  "category": "limit_exceeded",
  "limit": "max_unique_fragment_smiles",
  "search_stats": {"num_primitive_reactions": 50, "num_raw_combinations": 373,
                   "num_hard_conflict_pruned": 253, "...": "...",
                   "num_unique_fragment_smiles": 61}
}
```

`search_stats` is the full `ReactionSequenceSearchStats`, including the pruning
counters, or `null` if the tree search never ran. `error` entries add an
`error` field with the traceback.

### Open preparation settings in Workbench

Each preparation run writes `fragment-tree.training-data-config.pft` in its
output root. Click this file in VS Code Explorer to open **Data / Training Data**
in the FREDIS Workbench with its datasets, column mappings, split settings and
Fragmenter settings restored. Opening it does not start preparation.
**Save Configuration** uses the same `.training-data-config.pft` extension.

`preparation_config.json` remains available for compatibility and can still be
loaded with **Load Configuration**. Existing JSON settings can also be copied
to a filename ending in `.training-data-config.pft` without changing their
contents. The per-split `fragment-tree.training-data.pft` files continue to open
the generated-data results viewer.

## `manifest.tsv`

Tab-separated, one row per source, sorted by SMILES.

| Column | Meaning |
|---|---|
| `file` | `.preft.pt` name under `data/` (empty for skipped sources) |
| `smiles` | Source SMILES |
| `record_indexes` | JSON list of kept record indexes (all input records for skipped sources) |
| `num_input_records` | Records of this source |
| `num_valid_samples` | Records kept in the structure |
| `rejected_sample_count` | Records dropped (unresolved precursor), or all records of a skipped source |
| `rejection_log` | `skipped_sources.json` for skipped sources |
| `num_branch_groups` | Distinct main adducts |
| `num_teacher_nodes` | Teacher states, including one empty root per branch group |
| `num_transition_states` | Teacher states plus one-step next states of valid reactions |
| `num_positive_transitions` | Unique teacher (parent, child, reaction) steps |
| `num_ion_states` | Structural-node/ion-condition/sample states |
| `max_teacher_reaction_count` | Largest number of reactions in a teacher state |
| `num_nodes`, `num_edges` | Nodes and edges of the **materialized teacher trees**, not the full search tree |
| `assignment_score`, `assignment_score_without_precursor` | Mean per-sample assignment scores |
| `num_primitive_reactions` … `num_unique_fragment_smiles` | Reaction search statistics (see [reference](#statistics-reference)); partial for limit skips, empty when unavailable |
| `status` | `completed` or `skipped` |
| `skip_category` | Empty, or `limit_exceeded` / `unresolved_precursor` / `error` / `worker_crash` |
| `reason` | Skip reason, or the partial-drop note for a completed source |

## `reaction_statistics.json`

Split-level summary:

```json
{
  "architecture": "fragment-tree-competitive-ion",
  "max_primitive_reactions": 83,
  "max_primitive_reactions_smiles": "…",
  "max_raw_combinations": 91234,
  "max_raw_combinations_smiles": "…",
  "max_compiled_sequences": 15231,
  "max_compiled_sequences_smiles": "…",
  "max_rdkit_run_reactants": 15231,
  "max_rdkit_run_reactants_smiles": "…",
  "max_generated_fragments": 15231,
  "max_generated_fragments_smiles": "…",
  "max_observed_unique_fragment_smiles": 812,
  "max_observed_unique_fragment_smiles_smiles": "…",
  "search_limits": {"max_unique_fragment_smiles": 1000, "max_reaction_combinations": 100000},
  "num_skipped_sources_by_category": {"error": 2, "limit_exceeded": 41},
  "num_limit_skipped_sources": {"max_unique_fragment_smiles": 30, "max_reaction_combinations": 11},
  "num_sources": 9000,
  "num_samples": 120000,
  "…": "…"
}
```

- `max_*` values are the largest per-source search statistics among
  **completed** sources, each with the SMILES that reached it (`*_smiles`;
  ties go to the lexicographically smallest SMILES). They describe the data
  that was actually prepared. Partial statistics of skipped sources are in
  `manifest.tsv` and `skipped_sources.json`.
- The largest unique-fragment count is named
  `max_observed_unique_fragment_smiles`, so it cannot be mistaken for the
  `max_unique_fragment_smiles` **limit** in `search_limits`.
- With no completed source, maxima and their SMILES are `null`.
- Other keys:
  - `num_teacher_nodes`, `num_positive_transitions`,
    `mean_teacher_nodes_per_sample`, `mean_positive_edges_per_sample`
  - `num_branch_groups`, `num_transition_states`,
    `num_ion_states`,
    `max_teacher_reaction_count`
  - `num_sources`, `num_samples`, `num_metadata_valid_records`,
    `num_skipped_sources`, `num_skipped_records`, `num_rejected_records`,
    `num_excluded_records`
  - `num_workers`, `chunk_size`, `chemistry_config`,
    `minimum_relative_intensity`

## Statistics reference

Per-source statistics come from `ReactionSequenceSearchStats`, filled while the
source's fragment tree is built.

| Statistic | In manifest | Meaning |
|---|---|---|
| `num_primitive_reactions` | yes | Primitive reactions matched on the source |
| `num_raw_combinations` | yes | (parent, reaction) pairs examined by the search over all sizes, before pruning; bounded by `max_reaction_combinations` |
| `num_hard_conflict_pruned` | skipped only | Pairs rejected because two reactions change the same bond |
| `num_invalidated_reaction_pruned` | skipped only | Pairs rejected because one reaction removes atoms another needs |
| `num_duplicate_sequence_pruned` | skipped only | Pairs that normalized to their parent or to an already visited history |
| `num_duplicate_effect_pruned` | skipped only | Histories that reused a cached product of an identical effect |
| `num_compiled_sequences` | yes | Unique effects compiled into a reaction |
| `num_rdkit_run_reactants` | yes | RDKit `RunReactants` calls (one per compiled effect) |
| `num_generated_fragments` | yes | Successful single-product reactions |
| `num_unique_fragment_smiles` | yes | Distinct canonical fragment SMILES, excluding the source; bounded by `max_unique_fragment_smiles` |

"skipped only" means the value appears in `skipped_sources.json` but has no
manifest column.

Typical relations:

- `num_raw_combinations` ≥ `num_compiled_sequences` = `num_rdkit_run_reactants`
  ≥ `num_generated_fragments` ≥ `num_unique_fragment_smiles`.
- `num_rdkit_run_reactants` − `num_generated_fragments` is the number of
  effects that failed sanitization.

## Code map

| File | Role |
|---|---|
| `create_training_data.py` | CLI, splits, grouping, serial/parallel orchestration, skip handling, all output files |
| `subprocess_worker.py` | Worker process for one chunk of sources |
| `record_validation.py` | Record-level metadata checks (`invalid_records.json`) |
| `datasets.py` | Input loading, `split_by_smiles`, `dedupe_validation` |
| `context.py` | Fragmenter / graph builder / tensorizer context, `validate_limits` |
| `manifest_summary.py` | Structure-level manifest columns |
| `fredis/ml/input/structure_builder.py` | `ReactionStructureBuilder`: tree, precursor, assignment, teacher, targets; `last_search_stats` |
| `fredis/ml/input/source_reaction_structure.py` | Teacher structure tensors and `.preft.pt` saving |
| `fredis/domain/fragment/reaction/ReactionSequenceSearch.py` | Combination search, statistics, `max_reaction_combinations`, `FragmentTreeLimitExceeded` |
| `fredis/domain/fragment/tree/FragmentTreeBuilder.py` | Materialization, unique SMILES, `max_unique_fragment_smiles`, tree construction |
| `fredis/domain/fragment/ion_tree/FragmentIonTreeBuilder.py` | Ion candidates; attaches `_search_stats` to the ion tree |

User-facing guides: [data preparation](../../../../docs/guides/data-preparation.md)
and the optional Workbench GUI in its separate repository. Training details are in
[`fragment_tree_training/README.md`](../../training/fragment_tree_training/README.md).
