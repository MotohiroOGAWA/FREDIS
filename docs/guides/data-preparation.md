# Prepare fragment-tree training data

Choose experimental spectra with SMILES, precursor adduct, collision energy, precursor m/z and peak lists. Supported inputs are `.msds`, `.msp`, `.mgf`, `.tsv`, `.csv` and `.parquet`. Tables require a `Peak` column in msentity spectrum-table syntax. Preview and column validation use the same input loader as preparation.

## Train and validation datasets

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input train.msds --validation-input validation.msds \
  --output-dir data/prepared \
  --params single_bond_cleavage_pos.fragmenter.pft
```

Data preparation needs only chemistry: the Fragmenter (`--params` / `--params-json`)
and the element symbols of the atom features (`--symbols-json`, default
`C,N,O,P,S,F,Cl,Br,I`). The Fragmenter is required; there is no model preset.
Model architecture (dimensions, layers, heads) is chosen later, when training,
so one prepared dataset can train models of any size.
`fredis/presets/fragment/single_bond_cleavage_pos.fragmenter.pft` is an example
Fragmenter and the Workbench's initial Fragmentation settings.

With an explicit validation dataset, outputs are written to `train_structures` and `validation_structures`. To create a held-out split by unique SMILES, replace `--validation-input` with `--validation-ratio 0.1 --validation-seed 0`. Molecules are kept within a single split. At least two distinct SMILES are needed for automatic splitting.

Without either validation option, the CLI retains the single-dataset behavior and writes structures directly to the output directory.

## Column mapping

Map actual metadata names with `--smiles-column`, `--adduct-type-column`, `--collision-energy-column` and `--precursor-mz-column`. Workbench shows whether each mapped column exists in both selected datasets and whether the values are valid. Selecting a file loads the preview and checks column presence only. Click **Validate Dataset Values** to check every record; separate Train and Validation progress bars continue to completion without the preview timeout. SMILES are parsed by RDKit; adducts must resolve to a main adduct registered in the fragmenter; collision energy uses the FREDIS parser, including supported strings; precursor m/z must convert to a finite positive number. Input, mapping or fragmentation changes invalidate this check. The manual check is informative rather than a prerequisite for **Copy Command** or **Run**; Preflight and the CLI still perform the authoritative checks.

## Fragmentation and assignment

Edit the maximum reaction count and precursor reaction limit directly. **Reaction Pattern Set** and **Fragment Ion Adduct Rule Set** are collapsed summaries that expand their shared dedicated editors inline, without leaving Fragmentation. The collapsed Pattern summary shows its set name; the collapsed adduct summary shows its set name and comma-separated main adducts. Adduct-rule sets can also be created under **Data → Adduct Rules**; sets use `*.adductset.pft` and individual rules use `*.adduct.pft`. Data Preparation exposes element **Symbols** through a periodic table; it requires no neural model dimensions or separate adduct embedding list. The adduct embedding order is resolved automatically from the Fragmenter rules and observed dataset metadata; there is no CLI option for setting it independently. The CLI accepts `--symbols-json '["C","N","O","P","S"]'`.

The small **Load** / **Save** controls at the upper right of Fragmentation import and export only the fragmenter configuration. You can also drop a fragmenter config JSON anywhere on that compact control row. Model configuration JSON belongs to the model and training workflows and is intentionally rejected here.

Two per-source search limits bound the cost of one molecule. Both default to `-1` (unlimited) and otherwise must be positive; a source that exceeds either is skipped as a whole rather than silently truncated.

- `--max-unique-fragment-smiles N` counts the distinct canonical SMILES of materialized fragments, **excluding the source molecule itself**. It is checked each time RDKit generates a fragment; the source stops as soon as the count exceeds `N`.
- `--max-reaction-combinations N` bounds the raw reaction combinations examined by the reaction search (`num_raw_combinations`), over every combination size from 1 to `max_reaction_count`. It is checked inside the search, before the next combination can reach RDKit.

See [the preparation README](../../fredis/ml/data_preparation/fragment_tree/README.md#search-limits) for the exact semantics. Mass tolerance controls peak-to-formula assignment, for example `--mass-tolerance 0.01Da,10ppm`. The minimum relative peak intensity option filters low-intensity experimental peaks before assignment. Normalization is enabled by default and scales each spectrum's maximum retained intensity to 1.

## Parallel processing

Use `--num-workers 4 --chunk-size 1` to prepare independent SMILES groups in subprocesses. Workbench exposes these under **Output → Parallel Processing**. One worker is the default and runs serially. Worker outputs retain the same source indices and filenames as serial preparation.

## Configuration precedence

`--params` file values are followed by inline `--params-json`, named options and repeated `--set PATH=JSON` overrides, in that order. `--set` paths are relative to the Fragmenter, e.g. `--set fragment_ion_tree_builder.max_reaction_count=2`. `--params` accepts a Fragmenter file or a chemistry configuration (`fragmenter_params` plus `symbols`).

```bash
python -m fredis.cli train create-fragment-tree-data \
  --input train.msds --output-dir data/prepared \
  --params single_bond_cleavage_pos.fragmenter.pft --max-reaction-count 2 \
  --mass-tolerance 0.02Da \
  --max-unique-fragment-smiles 1000 --max-reaction-combinations 50000
```

Workbench passes the edited configuration snapshot, so a later form edit takes priority over an imported file.

## Outputs

Structures are saved as `.preft.pt` files under `train_structures/data` and `validation_structures/data`, with source and record metadata. Single-dataset outputs use `<output>/data`. Each structure directory contains `manifest.tsv` listing SMILES, files, record counts, per-source reaction search statistics (`num_primitive_reactions`, `num_raw_combinations`, `num_compiled_sequences`, `num_rdkit_run_reactants`, `num_generated_fragments`, `num_unique_fragment_smiles`) and skip category/reason for the result tables. `reaction_statistics.json` records counts, configuration, the configured search limits and the largest search statistics among prepared sources together with the SMILES that reached them (`max_raw_combinations`, `max_raw_combinations_smiles`, ...). Use these maxima to choose search limits for a new run. If the Output Directory already exists and overwrite is disabled, the interactive CLI asks for confirmation before loading or validating the datasets; a non-interactive run fails immediately. Workbench presents the equivalent modal confirmation before its run-time validation or processing begins. With overwrite enabled, the entire specified Output Directory is cleared before generating fresh outputs. Input and application directories cannot be used as overwrite destinations. JSON Lines progress events let Workbench display the current source and split.

Invalid SMILES, unsupported or malformed adducts, unparseable collision energies and invalid precursor m/z values exclude the affected records. They do not block preparation when usable records remain. **Invalid Records** in Workbench shows the original zero-based record number, ID, column value and exclusion reason, with search and pagination. A record with multiple invalid fields is excluded once. Required columns must still exist, and each input must contain usable records. Automatic splitting requires at least two usable unique SMILES.

The CLI applies the same checks before splitting or collecting adduct conditions. `invalid_records.json` in the output root lists excluded records from each original input for review.

Exceeding `max_reaction_combinations` partway through depth n retains only
fully enumerated levels through n - 1. Depth 1 overflow retains Source only.
This happens before RDKit materialization, without re-enumeration or replay.
`ReactionStructureBuilder` can also retry a smaller depth on the separate
unique-structure limit. Sources whose precursors cannot be resolved in the
retained tree are skipped. Other per-source errors and worker failures are
recorded in `skipped_sources.json` and do not stop the remaining sources.
`metadata.search_stats` and manifest depth columns identify the retained depth;
`limit_attempts` records observed overflow counters. Aggregate statistics
separate prepared, metadata-valid and skipped record counts.

The parent process displays a `tqdm` progress bar on stderr, including prepared/skipped source counts, and also emits JSON Lines for Workbench. Progress advances for each attempted source, including skipped sources.

Resolved preparation settings are saved as `preparation_config.json` directly in Output Directory; its `chemistry_config` (`fragmenter_params`, `symbols`, `adduct_type_strs`) is what training inherits before computation in both single-dataset and train/validation modes. Preparation configuration files are not written inside `train_structures` or `validation_structures`. Workbench additionally saves `fragment-tree.pft` in the output root before starting the CLI; use **Load Configuration** to restore the preparation form from this file.

When `H` is absent from **Symbols**, explicit hydrogen atoms are omitted from encoder graphs; attached explicit hydrogen neighbors are included in the heavy atom's hydrogen-count feature. The original chemical molecule, formula and valence are preserved. This also permits hydrogen-only materialized fragments to have an empty encoder graph. Selecting `H` keeps hydrogen nodes in the graph. Source reaction indexes are remapped to the represented graph atoms.

Parallel preparation uses `fredis.utils.parallel_subprocess.run_parallel_subprocesses` to launch independent Python commands for chunks of SMILES groups. Each subprocess initializes its own RDKit preparation context. The common utility displays tqdm progress and notifies the parent after a successful chunk; the parent updates source counters and collects saved results. Parallel progress consumes completed chunks immediately, so a slow earlier chunk does not hold back updates from later completed chunks. Output filenames and the returned file list remain deterministic. The counter advances when a chunk completes; a molecule still being computed is not counted as complete.

Both root configuration files can be loaded by Workbench. Loading restores training/validation input paths, output path, column mapping, split settings, intensity options, overwrite, worker/chunk settings, symbols, search limits (`maxUniqueFragmentSmiles`, `maxReactionCombinations`) and all fragmentation parameters. Python snake_case keys from `preparation_config.json` are converted to the Workbench form keys. Dataset value checks must be rerun after loading.

Source reaction combinations are enumerated as reaction ID sets before chemical
execution. Compatibility checks and deduplication use precomputed source
relations. The incomplete over-budget depth is discarded; retained candidates
are then materialized once per unique effect. Canonical SMILES merges identical
Compound structures. The unique-SMILES limit stops later materialization when
its own bound is exceeded.

The result viewer reads each split manifest to display the structures. Training recursively discovers `.preft.pt` files, so select `train_structures` and `validation_structures`.
