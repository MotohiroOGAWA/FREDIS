"""Run a Reaction Pattern Analysis over a PatternSet tree and write the A/B/C result.

Phase 1  input/source resolution              Phase 6  condition calculations
Phase 2  tree, pattern ids and conditions      Phase 7  condition outcomes
Phase 3  spectra and unique compounds          Phase 8  node/edge comparison and summaries
Phase 4  precursor SMARTS applicability        Phase 9  A / B / C persistence
Phase 5  per-compound plan (reuse, calculate)  Phase 10 finalize result
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable

import numpy as np

from . import applicability, chemistry, executor, parquet_storage, summary
from .comparison import ConditionOutcome, compare_compound, empty_outcome
from .compounds import CompoundGroup, group_compounds, mark_precursor_peaks
from .constants import (
    COMPLETED, COMPOUND_DIR, DEFAULT_COLUMNS, FAILED, CHEMISTRY_CONFIG_NAME, RESULT_NAME, RUN_CONFIG_NAME, SKIP_ERROR,
    SKIP_PARENT_UNAVAILABLE, SKIPPED, TREE_CONFIG_NAME,
)
from .hdf5_storage import ResultWriter
from .previous import PreviousResult, chemistry_signature, compare_signatures
from .sources import expand_dataset_inputs, load_samples, normalize_analysis_sources
from .tree import PatternRegistry, TreeConfiguration, resolve_tree
from .workspace import initialize_output_directory

PHASES = 10


def _section(index: int, title: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"\n=== Reaction Pattern Analysis: Phase {index}/{PHASES} {title}{suffix} ===", file=sys.stderr, flush=True)


def _note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_configuration(output: Path, source_rows: Iterable[dict], tree: TreeConfiguration, chemistry_config: dict, *,
                      max_unique_fragment_smiles: int, max_reaction_combinations: int,
                      minimum_relative_intensity: float, num_workers: int,
                      chunk_size: int, overwrite: bool = False, extend: bool = False) -> dict:
    """The Workbench Input / Run form values that reproduce this run."""
    builder = chemistry_config["fragmenter_params"]["fragment_ion_tree_builder"]
    return {
        "sources": [{key: value for key, value in row.items() if key != "inspected"} for row in source_rows],
        "params": str(output / CHEMISTRY_CONFIG_NAME), "outputDir": str(output), "tree": tree.to_dict(),
        "maxReactionCount": builder.get("max_reaction_count", 2),
        "massTolerance": chemistry_config["fragmenter_params"].get("mass_tolerance", "20 ppm"),
        "maxUniqueFragmentSmiles": max_unique_fragment_smiles, "maxReactionCombinations": max_reaction_combinations,
        "minimumRelativeIntensity": minimum_relative_intensity,
        "numWorkers": num_workers, "chunkSize": chunk_size, "overwrite": overwrite, "extend": extend,
    }


def _write_run_configuration(output: Path, chemistry_config: dict, run: dict) -> None:
    (output / CHEMISTRY_CONFIG_NAME).write_text(json.dumps(chemistry_config, ensure_ascii=False, indent=2) + "\n")
    (output / TREE_CONFIG_NAME).write_text(json.dumps(run["tree"], ensure_ascii=False, indent=2) + "\n")
    (output / RUN_CONFIG_NAME).write_text(json.dumps({
        "schema": "fredis.workbench.configuration", "schemaVersion": 1, "kind": "reaction-pattern-analysis", "config": run,
    }, ensure_ascii=False, indent=2) + "\n")


def _library_config(chemistry_config: dict, registry: PatternRegistry) -> dict:
    """The chemistry configuration whose pattern set is the analysis-wide library (stable ids)."""
    config = copy.deepcopy(chemistry_config)
    builder = config["fragmenter_params"]["fragment_ion_tree_builder"]
    builder["reaction_pattern_set"] = {"name": "reaction_pattern_analysis_library", "patterns": registry.pattern_documents()}
    return config


def _compound_input(group: CompoundGroup) -> chemistry.CompoundInput:
    samples = group.samples
    return chemistry.CompoundInput(
        compound_id=group.compound_id, smiles=group.canonical_smiles,
        source_positions=list(range(len(samples))),
        sample_ids=[sample.sample_id for sample in samples],
        dataset_ids=[sample.dataset_id for sample in samples],
        record_indexes=[sample.record_index for sample in samples],
        adducts=[sample.adduct for sample in samples],
        collision_energies=[sample.collision_energy for sample in samples],
        peaks_mz=[sample.mz.tolist() for sample in samples],
        peaks_intensity=[sample.intensity.tolist() for sample in samples])


def _detail_file(output: Path, condition, compound_id: str) -> str:
    return str(output / condition.detail_directory / f"{compound_id}.preft.pt")


class _Plan:
    """How each condition of one compound is resolved."""

    def __init__(self):
        self.kind: dict[int, str] = {}           # "alias" | "reused" | "compute"
        self.alias: dict[int, int] = {}          # condition -> effective condition
        self.reused: dict[int, object] = {}      # condition -> PreviousOutcome
        self.steps: list[chemistry.ConditionStep] = []
        self.applicable: dict[str, bool] = {}

    def effective(self, condition_id: int) -> int:
        while condition_id in self.alias:
            condition_id = self.alias[condition_id]
        return condition_id


def _plan_compound(group: CompoundGroup, tree, matched: set[int], previous: dict, output: Path) -> _Plan:
    plan = _Plan()
    for node in tree.nodes:
        condition = tree.condition(node.condition_id)
        is_root = node.parent_condition_id is None
        applicable = is_root or bool(set(node.added_pattern_ids) & matched)
        plan.applicable[node.node_id] = applicable
        if node.condition_id in plan.kind:
            continue
        if not applicable:
            plan.kind[node.condition_id] = "alias"
            plan.alias[node.condition_id] = node.parent_condition_id
        elif condition.key in previous:
            plan.kind[node.condition_id] = "reused"
            plan.reused[node.condition_id] = previous[condition.key]
        else:
            plan.kind[node.condition_id] = "compute"
            parent = None if is_root else plan.effective(node.parent_condition_id)
            parent_ids = frozenset() if parent is None else tree.condition(parent).pattern_ids
            plan.steps.append(chemistry.ConditionStep(
                condition_id=node.condition_id, parent_condition_id=parent,
                pattern_ids=tuple(sorted(condition.pattern_ids)),
                added_pattern_ids=tuple(sorted(condition.pattern_ids - parent_ids)),
                output_file=_detail_file(output, condition, group.compound_id)))
    return plan


def _check_identity(group: CompoundGroup, previous: PreviousResult, outcomes: dict) -> None:
    known = previous.samples.get(group.compound_id, {})
    for outcome in outcomes.values():
        for sample_index in outcome.kept:
            sample = group.samples[sample_index] if sample_index < len(group.samples) else None
            if sample is None or known.get(sample_index) != (sample.dataset_id, sample.record_index):
                raise ValueError(f"The spectra of compound {group.compound_id} differ from the existing result. "
                                 "Use the same sources and settings to add nodes, or run without extending.")


def _reused_outcome(group: CompoundGroup, prior, masks: dict, condition_id_text: int) -> ConditionOutcome:
    outcome = empty_outcome(group, prior.status, skip_category=prior.skip_category, reason=prior.reason,
                            error=prior.error, kept=tuple(prior.kept), file=prior.detail_path,
                            seed_file=prior.seed_path)
    if prior.status != COMPLETED:
        return outcome
    for index in prior.kept:
        assigned, counts = masks[(index, condition_id_text)]
        if len(assigned) != len(group.samples[index].mz):
            raise ValueError(f"The peaks of compound {group.compound_id} differ from the existing result.")
        outcome.assigned[index] = assigned
        outcome.counts[index] = counts
    return outcome


def _computed_outcome(group: CompoundGroup, result: dict, parent: ConditionOutcome | None) -> ConditionOutcome:
    outcome = empty_outcome(group, result["status"], skip_category=result.get("skip_category", ""),
                            reason=result.get("reason", ""), error=result.get("error", ""),
                            search_stats=result.get("search_stats"))
    if result["status"] != COMPLETED:
        return outcome
    kept = tuple(int(value) for value in result["kept"])
    if len(kept) != len(result["counts"]):
        raise ValueError("Chemistry result kept/count lengths differ")
    outcome.kept, outcome.file, outcome.seed_file = kept, result["file"], result["file"]
    for index, values in zip(kept, result["counts"]):
        counts = np.asarray(values, dtype=np.int32)
        if parent is not None:
            # The seeded child tree may omit parent histories at full depth;
            # the condition always includes every parent assignment.
            outcome.assigned[index] = parent.assigned[index] | (counts > 0)
            outcome.counts[index] = np.maximum(parent.counts[index], counts)
        else:
            outcome.assigned[index] = counts > 0
            outcome.counts[index] = counts
    return outcome


def _alias_outcome(target: ConditionOutcome, target_id: int) -> ConditionOutcome:
    return ConditionOutcome(status=target.status, skip_category=target.skip_category, reason=target.reason,
                            error=target.error, kept=target.kept, assigned=target.assigned, counts=target.counts,
                            file="", seed_file=target.seed_file, alias_of=target_id)


def generate_reaction_pattern_analysis(*, sources: Iterable[dict] = (), output_dir: str | Path,
                               chemistry_config: dict | None = None, tree: TreeConfiguration | dict | None = None, smiles_column: str = "SMILES",
                               adduct_type_column: str = "AdductType", collision_energy_column: str = "CollisionEnergy",
                               precursor_mz_column: str = "PrecursorMZ", minimum_relative_intensity: float = 0.0,
                               max_unique_fragment_smiles: int = -1,
                               max_reaction_combinations: int = -1, num_workers: int = 1, chunk_size: int = 1,
                               overwrite: bool = False, extend: bool = False) -> Path:
    """Generate ``<output_dir>/result.reactionpatternanalysis.pft`` and its B/C files.

    With ``extend`` the result already in ``output_dir`` is kept: conditions
    it calculated are reused and only new ones are calculated.  Sources,
    chemistry configuration and tree default to the existing result's.
    """
    started = time.monotonic()
    columns = {"smiles": smiles_column, "adduct": adduct_type_column,
               "collision_energy": collision_energy_column, "precursor_mz": precursor_mz_column}
    output = Path(output_dir).resolve()
    previous = None
    if extend:
        if not (output / RESULT_NAME).exists():
            raise FileNotFoundError(f"No Reaction Pattern Analysis result to extend in {output}")
        previous = PreviousResult(output / RESULT_NAME)
    requested = [dict(value, path=str(Path(value["path"]).resolve())) if value.get("path") else dict(value)
                  for value in sources]
    if not requested and previous is not None:
        requested = copy.deepcopy(previous.run.get("sources") or [])
    if not requested:
        # Checked before the output directory is created or cleared.
        raise ValueError("No input spectra: add at least one --source <dataset|prepared|result> "
                         "(repeatable) or --source-spec.")
    if chemistry_config is None:
        if previous is None:
            raise ValueError("A Fragmenter configuration is required.")
        chemistry_config = previous.chemistry_config
    if tree is None:
        if previous is None:
            raise ValueError("A Tree Configuration with at least one node is required.")
        tree = previous.tree
    tree = tree if isinstance(tree, TreeConfiguration) else TreeConfiguration.from_dict(tree)

    # Phase 1 ---------------------------------------------------------------
    _section(1, "Input/source resolution", str(output))
    settings = dict(max_unique_fragment_smiles=max_unique_fragment_smiles,
                    max_reaction_combinations=max_reaction_combinations,
                    minimum_relative_intensity=minimum_relative_intensity,
                    num_workers=num_workers, chunk_size=chunk_size)
    if previous is not None:
        signature = {"maxUniqueFragmentSmiles": max_unique_fragment_smiles,
                     "maxReactionCombinations": max_reaction_combinations,
                     "minimumRelativeIntensity": minimum_relative_intensity}
        differences = compare_signatures(chemistry_signature(previous.chemistry_config, previous.run),
                                         chemistry_signature(chemistry_config, signature))
        if differences:
            raise ValueError("Nodes can only be added with the settings of the existing result; these differ: "
                             + ", ".join(differences) + ". Run without extending to recalculate everything.")
    else:
        overwrite = initialize_output_directory(output, overwrite=overwrite,
                                                 protected_paths=[row["path"] for row in requested if row.get("path")])
    source_rows = normalize_analysis_sources(sources=requested, columns=columns, show_progress=True)

    # Phase 2 ---------------------------------------------------------------
    registry = PatternRegistry(previous.patterns if previous is not None else ())
    resolved = resolve_tree(tree, registry)
    _section(2, "Tree, pattern ids and conditions",
             f"{len(resolved.nodes)} node(s), {len(resolved.conditions)} condition(s)")
    library_config = _library_config(chemistry_config, registry)
    run = run_configuration(output, source_rows, tree, chemistry_config, overwrite=overwrite, extend=extend, **settings)
    _write_run_configuration(output, chemistry_config, run)
    for node in resolved.nodes:
        _note(f"  {'  ' * node.depth}{node.pattern_set_name}  [{node.node_id}] → {node.condition_label}")

    # Phase 3 ---------------------------------------------------------------
    _section(3, "Spectra and unique compounds")
    from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
    fragmenter = create_preparation_context(library_config).fragmenter
    symbols = library_config["symbols"]
    samples, dataset_rows = [], []
    for dataset in expand_dataset_inputs(source_rows):
        loaded, row = load_samples(dataset, fragmenter=fragmenter, symbols=symbols,
                                   minimum_relative_intensity=minimum_relative_intensity)
        samples.extend(loaded)
        dataset_rows.append(row)
        _note(f"Dataset {row['dataset_id']} {dataset.path.name}: {row['valid_record_count']:,} valid of "
              f"{row['record_count']:,} records ({row['skipped_record_count']:,} skipped).")
    if previous is not None:
        before = [(row["original_path"], row["valid_record_count"]) for row in previous.datasets]
        after = [(row["original_path"], row["valid_record_count"]) for row in dataset_rows]
        if before != after:
            raise ValueError("The datasets differ from the existing result. Add nodes with the same sources, "
                             "or run without extending.")
    groups = group_compounds(samples)
    mark_precursor_peaks(groups, fragmenter.mass_tolerance)
    _note(f"{len(samples):,} spectra belong to {len(groups):,} unique compounds.")

    # Phase 4 ---------------------------------------------------------------
    _section(4, "Precursor SMARTS applicability")
    added = sorted({pid for node in resolved.nodes for pid in node.added_pattern_ids
                    if node.parent_condition_id is not None})
    compiled = applicability.compile_reactant_smarts(
        [dict(registry.definition(pid), pattern_id=pid) for pid in added])
    matched = applicability.match_precursors({group.compound_id: group.canonical_smiles for group in groups}, compiled)

    # Phase 5 ---------------------------------------------------------------
    _section(5, "Per-compound plan")
    plans = {}
    for group in groups:
        prior = previous.compound_outcomes(group.compound_id) if previous is not None else {}
        plans[group.compound_id] = _plan_compound(group, resolved, set(matched.get(group.compound_id, ())), prior, output)
    steps = sum(len(plan.steps) for plan in plans.values())
    reused = sum(len(plan.reused) for plan in plans.values())
    _note(f"{steps:,} condition tree(s) to calculate; {reused:,} reused from the existing result.")

    # Phase 6 ---------------------------------------------------------------
    _section(6, "Condition calculations", f"{sum(1 for plan in plans.values() if plan.steps):,} compound(s)")
    by_id = {group.compound_id: group for group in groups}
    tasks = []
    for group in groups:
        plan = plans[group.compound_id]
        if not plan.steps:
            continue
        resolved_conditions = {
            condition_id: chemistry.ResolvedCondition(condition_id, prior.status, tuple(prior.kept), prior.seed_path)
            for condition_id, prior in plan.reused.items()}
        tasks.append(chemistry.ChemistryTask(_compound_input(group), plan.steps, resolved_conditions))
    options = chemistry.ChemistryOptions(
        chemistry_config=library_config,
        max_unique_fragment_smiles=max_unique_fragment_smiles, max_reaction_combinations=max_reaction_combinations)
    computed: dict[str, dict[int, dict]] = {}

    def record(result: dict) -> None:
        computed[result["compound_id"]] = {int(step["condition_id"]): step for step in result["steps"]}

    executor.execute(tasks, options, num_workers=num_workers, chunk_size=chunk_size,
                     description="Condition trees", on_result=record)

    # Phase 7 ---------------------------------------------------------------
    _section(7, "Condition outcomes")
    comparisons = []
    for group in groups:
        plan = plans[group.compound_id]
        outcomes: dict[int, ConditionOutcome] = {}
        if plan.reused:
            prior_rows = list(plan.reused.items())
            columns_by_condition = {condition_id: (prior.assigned_column, prior.count_column)
                                    for condition_id, prior in prior_rows}
            peak_file = previous.peak_file(group.compound_id)
            masks = parquet_storage.read_condition_masks(peak_file, columns_by_condition) if peak_file else {}
            for condition_id, prior in prior_rows:
                outcomes[condition_id] = _reused_outcome(group, prior, masks, condition_id)
            _check_identity(group, previous, outcomes)
        results = computed.get(group.compound_id, {})
        for node in resolved.nodes:
            condition_id = node.condition_id
            if condition_id in outcomes:
                continue
            kind = plan.kind[condition_id]
            if kind == "alias":
                target = plan.effective(condition_id)
                outcomes[condition_id] = _alias_outcome(outcomes[target], target)
            elif kind == "compute":
                step = next(item for item in plan.steps if item.condition_id == condition_id)
                result = results.get(condition_id) or dict(status=FAILED, skip_category=SKIP_ERROR,
                                                           reason="No calculation result was returned.")
                parent = outcomes.get(step.parent_condition_id) if step.parent_condition_id is not None else None
                if parent is not None and not parent.completed and result["status"] == COMPLETED:
                    result = dict(status=SKIPPED, skip_category=SKIP_PARENT_UNAVAILABLE,
                                  reason="The parent condition is unavailable for this compound.")
                outcomes[condition_id] = _computed_outcome(group, result, parent)
        comparisons.append(compare_compound(group, resolved.nodes, outcomes, plan.applicable))

    # Phase 8 ---------------------------------------------------------------
    _section(8, "Node/edge comparison and summaries")
    relative = lambda value: str(Path(value).resolve().relative_to(output)) if value else ""
    tables, overall = summary.summarize(comparisons, tree=resolved, datasets=dataset_rows, relative_path=relative)
    overall.update(calculatedConditionTrees=steps, reusedConditionTrees=reused)

    # Phase 9 ---------------------------------------------------------------
    _section(9, "A / B / C persistence")
    # An extension stages B next to the existing one and swaps it in with A,
    # so an interrupted run leaves the previous result intact.
    staging = output / (COMPOUND_DIR + ".new") if previous is not None else output
    if previous is not None and staging.exists():
        shutil.rmtree(staging)
    for comparison in comparisons:
        target = staging / Path(comparison.group.peak_summary_file).name if previous is not None else output
        if previous is not None:
            staging.mkdir(parents=True, exist_ok=True)
            frame = parquet_storage.peak_frame(comparison)
            frame.to_parquet(target, engine="pyarrow", index=False)
        else:
            parquet_storage.write_peaks(comparison, output)
    configuration = {
        "run": run, "tree": tree.to_dict(), "chemistryConfig": library_config, "columns": {**DEFAULT_COLUMNS, **columns},
        "sources": run["sources"], "maxReactionCount": run["maxReactionCount"], "massTolerance": run["massTolerance"],
        "maxUniqueFragmentSmiles": max_unique_fragment_smiles, "maxReactionCombinations": max_reaction_combinations,
        "minimumRelativeIntensity": minimum_relative_intensity,
        "numWorkers": num_workers, "chunkSize": chunk_size,
    }
    result_path = output / RESULT_NAME
    writer = ResultWriter(result_path)
    try:
        writer.write_json("configuration", configuration)
        writer.write_overall_summary(dict(overall, elapsedSeconds=time.monotonic() - started))
        for name in ("datasets", "pattern_sets", "patterns", "conditions", "nodes", "compounds",
                     "compound_node_summary", "compound_node_distribution", "node_summary", "node_distribution",
                     "sample_summary", "sample_node_summary", "compound_condition"):
            writer.write_table(name, tables.get(name))
        writer.write_indexes(len(comparisons), tables.tables)
    except BaseException:
        writer.abort()
        raise

    # Phase 10 --------------------------------------------------------------
    _section(10, "Finalize result")
    if previous is not None:
        current = output / COMPOUND_DIR
        retired = output / (COMPOUND_DIR + ".old")
        if retired.exists():
            shutil.rmtree(retired)
        if current.exists():
            current.rename(retired)
        staging.mkdir(parents=True, exist_ok=True)
        staging.rename(current)
        writer.finalize()
        shutil.rmtree(retired, ignore_errors=True)
    else:
        writer.finalize()
    _note(f"Reaction Pattern Analysis complete: {result_path}")
    return result_path
