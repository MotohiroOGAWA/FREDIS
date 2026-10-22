"""Prepare CE-independent branch groups and physical-ion tensors."""

from __future__ import annotations
import argparse, json, math, sys, os, pickle, tempfile, shutil, csv, traceback
from collections import Counter
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from fredis.utils.parallel_subprocess import run_parallel_subprocesses
from fredis.domain.fragment.tree.FragmentTreeBuilder import FragmentTreeLimitExceeded
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.libs.msentity.msentity import MSDataset
from fredis.domain.mass.parse_ce import parse_ce_to_ev
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.input.source_reaction_structure import (
    save_fragment_tree_structure,
    make_structure_file_stem,
    UnresolvedPrecursorError,
)
from .manifest_summary import structure_manifest_fields
from .context import create_preparation_context, validate_limits
from fredis.cli.base import output_directory
from fredis.cli.chemistry_options import (
    configure_chemistry_options,
    resolve_chemistry_options,
)
from .pretrained_models import stage_pretrained_models, relocate_sources
from .datasets import load_spectrum_dataset, split_dataset, canonical_keys, dedupe_validation
from .record_validation import inspect_records

# Fragment-tree search statistics copied from each source's search_stats into
# manifest.tsv (see README.md for their meaning).
SEARCH_STAT_FIELDS = (
    "num_primitive_reactions",
    "num_raw_combinations",
    "num_compiled_sequences",
    "num_rdkit_run_reactants",
    "num_generated_fragments",
    "num_unique_fragment_smiles",
    "requested_max_reaction_count",
    "completed_max_reaction_count",
    "search_truncated",
    "discarded_from_reaction_count",
)
# Why a source was skipped: a configured search limit, an unreachable
# precursor, any other exception while preparing it, or its worker process dying.
SKIP_LIMIT_EXCEEDED = "limit_exceeded"
SKIP_UNRESOLVED_PRECURSOR = "unresolved_precursor"
SKIP_ERROR = "error"
SKIP_WORKER_CRASH = "worker_crash"


def _saved_reaction_registry(reactions, fragmenter):
    """Persist viewer metadata in the same file so loading never reruns RDKit."""
    patterns = {
        pattern.pattern_id: pattern
        for pattern in fragmenter.fragment_ion_tree_builder.reaction_pattern_set
    }
    rows = []
    for index, reaction in enumerate(reactions):
        pattern = patterns.get(reaction.reaction_pattern_id)
        rule = (
            next(
                (
                    item
                    for item in pattern.reaction_rules
                    if item.id == reaction.reaction_rule_id
                ),
                None,
            )
            if pattern
            else None
        )
        rows.append(
            dict(
                id=index,
                reactionPatternId=reaction.reaction_pattern_id,
                reactionPatternName=pattern.name if pattern else "",
                reactantSmarts=pattern.reactant_smarts if pattern else "",
                reactionRuleId=reaction.reaction_rule_id,
                reactionRuleName=rule.source_rule.name if rule else "",
                reactionRuleProductSmarts=rule.source_rule.smarts if rule else "",
                productMoleculeId=reaction.product_molecule_id,
                sourceAtomMaps=list(reaction.source_atom_maps),
                retainedAtomMaps=sorted(reaction.retained_atom_maps),
                discardedAtomMaps=sorted(reaction.discarded_atom_maps),
                cutBondMaps=sorted(reaction.cut_bond_maps),
                changedBondMaps=sorted(reaction.changed_bond_maps),
            )
        )
    return rows


def _reset_output(output, overwrite, protected_paths=()):
    output = Path(output).resolve()
    if overwrite and output.exists():
        for protected in [
            Path(__file__).resolve(),
            *map(lambda value: Path(value).resolve(), protected_paths),
        ]:
            if protected.is_relative_to(output):
                raise ValueError(
                    "Output directory contains input or application files; choose a separate output directory."
                )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _confirm_existing_output(args):
    output = Path(args.output_dir).resolve()
    if args.overwrite or not output.exists():
        return
    if not output.is_dir():
        raise FileExistsError(
            f"Output path already exists and is not a directory: {output}"
        )
    message = f"Output directory already exists: {output}"
    if sys.stdin.isatty():
        answer = (
            input(message + "\nOverwrite it before processing? [y/N] ").strip().lower()
        )
        if answer in ("y", "yes"):
            args.overwrite = 1
            return
        raise FileExistsError("Cancelled because the output directory already exists.")
    raise FileExistsError(
        message + ". Re-run with --overwrite 1 or choose another directory."
    )


def _prepare_group(task, builder, options):
    tree, smiles, rows, dataset, *seed_rows = task
    output = Path(options["output_dir"])
    adduct_type_column = options["adduct_type_column"]
    collision_energy_column = options["collision_energy_column"]
    precursor_mz_column = options["precursor_mz_column"]
    minimum_relative_intensity = options["minimum_relative_intensity"]
    adducts = []
    energies = []
    mzs = []
    intensities = []
    for index in range(len(dataset)):
        record = dataset[index]
        adducts.append(Adduct.parse(str(dataset[adduct_type_column].iloc[index])))
        precursor_mz = float(dataset[precursor_mz_column].iloc[index])
        if not math.isfinite(precursor_mz) or precursor_mz <= 0:
            raise ValueError(f"Invalid precursor m/z at record {rows[index]}")
        ev = parse_ce_to_ev(dataset[collision_energy_column].iloc[index], precursor_mz)
        if ev is None or not math.isfinite(float(ev)) or float(ev) < 0:
            raise ValueError(f"Cannot convert CE at record {rows[index]}")
        energies.append(float(ev))
        peaks = record.peaks
        values = [(float(peak.mz), float(peak.intensity)) for peak in peaks]
        if any(
            not math.isfinite(mz) or mz <= 0 or not math.isfinite(value) or value < 0
            for mz, value in values
        ):
            raise ValueError(f"Invalid peak values at record {rows[index]}")
        maximum = max((value for _, value in values), default=0.0)
        retained = [
            (mz, value)
            for mz, value in values
            if value >= maximum * minimum_relative_intensity
        ]
        mzs.append([mz for mz, _ in retained])
        # Intensities are always relative to the record's base peak (max 1.0).
        intensities.append(
            [value / maximum if maximum > 0 else value for _, value in retained]
        )
    seed_compounds = {
        sequence: Compound.from_smiles(mapped_smiles)
        for sequence, mapped_smiles in (seed_rows[0] if seed_rows else ())
    }
    incremental = {}
    if options.get("addable_pattern_ids") is not None:
        incremental = dict(
            seed_compounds=seed_compounds,
            addable_pattern_ids=options["addable_pattern_ids"],
        )
    structure, kept = builder.build(
        Compound.from_smiles(smiles), adducts, energies, mzs, intensities,
        save_all_edges=options.get("save_all_edges", False), **incremental
    )
    kept_rows = [rows[index] for index in kept]
    path = (
        output / "data" / (make_structure_file_stem(smiles, index=tree) + ".preft.pt")
    )
    reaction_sequence_compounds = [
        (sequence, compound.mapped_smiles)
        for sequence, compound in builder.last_reaction_sequence_compounds.items()
    ]
    provenance = {}
    if "_fredis_dataset_path" in dataset.columns and "_fredis_source_record" in dataset.columns:
        paths, indexes = dataset["_fredis_dataset_path"], dataset["_fredis_source_record"]
        provenance["source_records"] = [dict(path=str(paths.iloc[index]), record_index=int(indexes.iloc[index]))
                                        for index in kept]
    save_fragment_tree_structure(
        structure=structure,
        output_file=path,
        metadata=dict(
            smiles=smiles,
            **provenance,
            record_indexes=kept_rows,
            search_stats=builder.last_search_stats,
            reaction_sequence_compounds=reaction_sequence_compounds,
            reaction_registry=_saved_reaction_registry(
                builder.last_reactions, builder.generator.fragmenter
            ),
        ),
    )
    annotations = structure.sample_annotations
    return (
        path,
        kept_rows,
        annotations,
        structure_manifest_fields(structure, annotations),
        builder.last_search_stats,
    )


def _skipped(task, category, reason, search_stats, **extra):
    return dict(
        skipped=dict(
            source_index=task[0],
            smiles=task[1],
            record_indexes=task[2],
            reason=reason,
            category=category,
            search_stats=search_stats,
            **extra,
        )
    )


def _prepare_group_safely(task, builder, options):
    # Cleared here as well as in build(): an error raised before the tree
    # search must not report the previous source's statistics.
    builder.last_search_stats = None
    try:
        path, kept_rows, annotations, summary, search_stats = _prepare_group(
            task, builder, options
        )
        scores = [
            dict(
                structure_file=path.name,
                sample_index=index,
                record_index=record_index,
                assignment_score=sample["assignmentScore"],
                assignment_score_without_precursor=sample[
                    "assignmentScoreWithoutPrecursor"
                ],
            )
            for index, (record_index, sample) in enumerate(zip(kept_rows, annotations))
        ]
        rejected = len(task[2]) - len(kept_rows)
        reason = (
            f"{rejected} of {len(task[2])} records dropped: unresolved precursor reaction sequence"
            if rejected
            else ""
        )
        return dict(
            path=path,
            record_count=len(kept_rows),
            input_record_count=len(task[2]),
            rejected_record_count=rejected,
            smiles=task[1],
            record_indexes=kept_rows,
            assignment_scores=scores,
            summary=summary,
            reason=reason,
            search_stats=search_stats,
        )
    except FragmentTreeLimitExceeded as error:
        # Statistics stop at the moment the limit was hit, so they are lower
        # bounds for the source; the offending one is observed>limit.
        return _skipped(
            task, SKIP_LIMIT_EXCEEDED, str(error), error.stats, limit=error.limit_name
        )
    except UnresolvedPrecursorError as error:
        return _skipped(
            task, SKIP_UNRESOLVED_PRECURSOR, str(error), builder.last_search_stats
        )
    except Exception as error:
        # One malformed or unsupported molecule must not abort a whole
        # (possibly hours-long) run: record it as skipped, with the traceback
        # kept in skipped_sources.json for diagnosis, and carry on.
        return _skipped(
            task,
            SKIP_ERROR,
            f"{type(error).__name__}: {error}",
            builder.last_search_stats,
            error=traceback.format_exc(),
        )


def _search_maxima(completed_search_stats):
    """Largest search statistic over completed sources, with the SMILES that reached it.

    num_X becomes max_X (num_unique_fragment_smiles becomes
    max_observed_unique_fragment_smiles, so it cannot be mistaken for the
    max_unique_fragment_smiles limit). Ties keep the lexicographically
    smallest SMILES, so the result does not depend on worker completion order.
    """
    result = {}
    ordered = sorted(completed_search_stats, key=lambda item: item[0])
    for field in SEARCH_STAT_FIELDS:
        key = (
            "max_observed_unique_fragment_smiles"
            if field == "num_unique_fragment_smiles"
            else "max_" + field[len("num_") :]
        )
        best = max(
            (
                (stats[field], smiles)
                for smiles, stats in ordered
                if stats.get(field) is not None
            ),
            key=lambda item: item[0],
            default=(None, None),
        )
        result[key] = best[0]
        result[key + "_smiles"] = best[1]
    return result


def create_reaction_training_data(
    *,
    dataset: MSDataset,
    chemistry_config: dict,
    output_dir: str | Path,
    smiles_column: str = "SMILES",
    adduct_type_column: str = "AdductType",
    collision_energy_column: str = "CollisionEnergy",
    precursor_mz_column: str = "PrecursorMZ",
    minimum_relative_intensity: float = 0.0,
    overwrite: bool = True,
    save_all_edges: bool = False,
    split: str = "dataset",
    max_unique_fragment_smiles: int = -1,
    max_reaction_combinations: int = -1,
    num_workers: int = 1,
    chunk_size: int = 1,
    preparation_config: dict | None = None,
    _check_existing: bool = True,
    seed_reaction_compounds_by_smiles: (
        dict[str, list[tuple[object, str]]] | None
    ) = None,
    addable_pattern_ids: frozenset[int] | None = None,
    source_record_indexes: list[int] | tuple[int, ...] | None = None,
) -> list[Path]:
    if (
        not math.isfinite(minimum_relative_intensity)
        or not 0 <= minimum_relative_intensity <= 1
    ):
        raise ValueError("Minimum relative intensity must be between 0 and 1.")
    validate_limits(max_unique_fragment_smiles, max_reaction_combinations)
    if (
        type(num_workers) is not int
        or num_workers < 1
        or type(chunk_size) is not int
        or chunk_size < 1
    ):
        raise ValueError("Worker processes and chunk size must be positive integers.")
    if source_record_indexes is not None and len(source_record_indexes) != len(dataset):
        raise ValueError(
            "source_record_indexes must contain one index per input spectrum."
        )
    inspection = inspect_records(
        dataset,
        create_preparation_context(chemistry_config).fragmenter,
        dict(
            smilesColumn=smiles_column,
            adductTypeColumn=adduct_type_column,
            collisionEnergyColumn=collision_energy_column,
            precursorMzColumn=precursor_mz_column,
        ),
        symbols=chemistry_config["symbols"],
    )
    original_rows = inspection["validIndexes"]
    source_rows = (
        [source_record_indexes[index] for index in original_rows]
        if source_record_indexes is not None
        else original_rows
    )
    dataset = dataset[original_rows]
    if not len(dataset):
        raise ValueError("The dataset contains no valid records to prepare.")
    generator = create_preparation_context(chemistry_config)
    builder = ReactionStructureBuilder(
        generator,
        max_unique_fragment_smiles=max_unique_fragment_smiles,
        max_reaction_combinations=max_reaction_combinations,
    )
    grouped = {}
    for index, smiles in enumerate(dataset[smiles_column].tolist()):
        grouped.setdefault(str(smiles), []).append(index)
    output = Path(output_dir)
    # main() already verified (and, with --overwrite, cleared) the whole output tree
    # before any split started computing; skip this redundant re-check there via
    # _check_existing=False so it can never abort a run mid-way, after training's
    # (possibly expensive) split has already completed, over a directory main()
    # already accounted for. overwrite still stays False for that caller so this
    # call's own _reset_output does not delete the split config main() just wrote.
    if _check_existing and not overwrite:
        existing = list(output.rglob("*.preft.pt"))
        if existing:
            raise FileExistsError(
                "Output already contains training structures. Enable overwrite or choose another directory."
            )
    if preparation_config is None:
        with stage_pretrained_models(chemistry_config) as staged:
            _reset_output(output, overwrite)
            chemistry_config = staged.install(output)
    else:
        _reset_output(output, overwrite)
    (output / "invalid_records.json").write_text(
        json.dumps(inspection["invalidRecords"], indent=2)
    )
    options = dict(
        output_dir=str(output),
        save_all_edges=save_all_edges,
        adduct_type_column=adduct_type_column,
        collision_energy_column=collision_energy_column,
        precursor_mz_column=precursor_mz_column,
        minimum_relative_intensity=minimum_relative_intensity,
        max_unique_fragment_smiles=max_unique_fragment_smiles,
        max_reaction_combinations=max_reaction_combinations,
        addable_pattern_ids=addable_pattern_ids,
    )
    seed_reaction_compounds_by_smiles = seed_reaction_compounds_by_smiles or {}
    tasks = (
        (
            tree,
            smiles,
            [source_rows[index] for index in rows],
            dataset[rows],
            seed_reaction_compounds_by_smiles.get(smiles, ()),
        )
        for tree, (smiles, rows) in enumerate(grouped.items())
    )
    if preparation_config is None:
        (output / "preparation_config.json").write_text(
            json.dumps(
                dict(
                    split=split,
                    chemistry_config=chemistry_config,
                    smiles_column=smiles_column,
                    num_workers=num_workers,
                    chunk_size=chunk_size,
                    overwrite=overwrite,
                    **options,
                ),
                indent=2,
            )
        )
    files = []
    skipped = []
    prepared_records = 0
    manifest_rows = []
    score_rows = []
    completed_search_stats = []

    def search_fields(stats):
        return {key: (stats or {}).get(key) for key in SEARCH_STAT_FIELDS}

    completed_sources = 0

    def collect_result(result):
        nonlocal prepared_records, completed_sources
        completed_sources += 1
        # No per-source printing: the tqdm bar's own postfix (below) already
        # shows prepared/skipped counts live without competing with its
        # redraws, and skipped_sources.json / manifest.tsv already report
        # every skip and completion once the run finishes.
        if "skipped" in result:
            source = result["skipped"]
            manifest_rows.append(
                dict(
                    file="",
                    smiles=source["smiles"],
                    record_indexes=json.dumps(source["record_indexes"]),
                    num_input_records=len(source["record_indexes"]),
                    num_valid_samples=0,
                    rejected_sample_count=len(source["record_indexes"]),
                    rejection_log="skipped_sources.json",
                    num_nodes=0,
                    num_edges=0,
                    assignment_score=None,
                    assignment_score_without_precursor=None,
                    **search_fields(source.get("search_stats")),
                    status="skipped",
                    skip_category=source.get("category", SKIP_ERROR),
                    reason=source["reason"],
                )
            )
            skipped.append(result["skipped"])
        else:
            files.append(Path(result["path"]))
            prepared_records += result["record_count"]
            score_rows.extend(result["assignment_scores"])
            completed_search_stats.append(
                (result["smiles"], result.get("search_stats") or {})
            )
            manifest_rows.append(
                dict(
                    file=Path(result["path"]).name,
                    smiles=result["smiles"],
                    record_indexes=json.dumps(result["record_indexes"]),
                    num_input_records=result["input_record_count"],
                    num_valid_samples=result["record_count"],
                    rejected_sample_count=result["rejected_record_count"],
                    rejection_log="",
                    **result["summary"],
                    **search_fields(result.get("search_stats")),
                    status="completed",
                    skip_category="",
                    reason=result["reason"],
                )
            )

    if num_workers == 1 or len(grouped) < 2:
        with tqdm(
            total=len(grouped),
            desc=f"Fragment trees ({split})",
            unit="tree",
            file=sys.stderr,
        ) as progress:
            for task in tasks:
                result = _prepare_group_safely(task, builder, options)
                collect_result(result)
                progress.set_postfix(
                    prepared=len(files), skipped=len(skipped), refresh=False
                )
                progress.update(1)
    else:
        # Use the shared subprocess runner, but only ever materialize up to
        # num_workers chunks' worth of task files at once (deleting each as
        # soon as its result is consumed), instead of writing every chunk
        # for the whole dataset upfront and keeping them all until the run
        # finishes. Each task file duplicates real per-record data (SMILES,
        # every kept peak) from the in-memory dataset, so writing all of
        # them at once for a NIST-scale dataset can need disk space on the
        # order of the whole dataset's size, in --output-dir, before a
        # single subprocess even starts -- on top of the growing .preft.pt
        # outputs, which is what actually exhausted disk on a large run.
        # Task files still live under output_dir (not the system temp dir):
        # output_dir is where the caller has already provisioned space for
        # this run's outputs, and bounding the window keeps that assumption
        # reasonable instead of needing the whole dataset duplicated there.
        with tempfile.TemporaryDirectory(
            prefix="fredis-preparation-", dir=str(output)
        ) as directory:
            effective_chunk_size = min(chunk_size, max(1, len(grouped) // num_workers))
            total_chunks = math.ceil(len(grouped) / effective_chunk_size)
            wave_size = max(1, num_workers)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            root = str(Path(__file__).resolve().parents[4])
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [root, env.get("PYTHONPATH")])
            )
            with tqdm(
                total=total_chunks,
                desc=f"Fragment trees ({split})",
                unit="chunk",
                file=sys.stderr,
            ) as progress:
                worker_module = (
                    "fredis.ml.data_preparation.fragment_tree.subprocess_worker"
                )

                def remove_task_files(command):
                    for flag in ("--task", "--result"):
                        Path(command[command.index(flag) + 1]).unlink(missing_ok=True)

                def consume_results(command):
                    try:
                        for result in json.loads(
                            Path(command[command.index("--result") + 1]).read_text()
                        ):
                            collect_result(result)
                    finally:
                        remove_task_files(command)

                def on_complete(command):
                    consume_results(command)
                    progress.set_postfix(
                        prepared=len(files), skipped=len(skipped), refresh=False
                    )

                def retry_individually(command):
                    # The worker process itself died (a Python exception is
                    # already recorded as skipped by _prepare_group_safely),
                    # e.g. a native crash or an OOM kill. Re-run each SMILES
                    # group of the chunk in its own process so that only the
                    # offending group is skipped, not the whole chunk or run.
                    task_file = Path(command[command.index("--task") + 1])
                    with task_file.open("rb") as stream:
                        payload = pickle.load(stream)
                    remove_task_files(command)
                    singles = {}
                    for position, task in enumerate(payload["tasks"]):
                        single_file = task_file.with_name(
                            f"{task_file.stem}-{position}.pkl"
                        )
                        with single_file.open("wb") as stream:
                            pickle.dump(
                                {**payload, "tasks": [task]},
                                stream,
                                pickle.HIGHEST_PROTOCOL,
                            )
                        singles[str(single_file)] = task

                    def on_single_error(single_command, error):
                        remove_task_files(single_command)
                        tree, smiles, rows, *_ = singles[
                            single_command[single_command.index("--task") + 1]
                        ]
                        collect_result(
                            _skipped(
                                (tree, smiles, rows),
                                SKIP_WORKER_CRASH,
                                f"Worker process exited with status {error.returncode}",
                                None,
                                error=(error.stderr or "")[-4000:],
                            )
                        )

                    with tqdm(disable=True) as quiet:
                        run_parallel_subprocesses(
                            [
                                [
                                    sys.executable,
                                    "-m",
                                    worker_module,
                                    "--task",
                                    single,
                                    "--result",
                                    str(Path(single).with_suffix(".json")),
                                ]
                                for single in singles
                            ],
                            max_workers=min(num_workers, len(singles)),
                            print_output=False,
                            env=env,
                            on_complete=consume_results,
                            on_error=on_single_error,
                            progress=quiet,
                        )

                task_id = 0
                while True:
                    commands = []
                    for _ in range(wave_size):
                        chunk = list(islice(tasks, effective_chunk_size))
                        if not chunk:
                            break
                        task_file = Path(directory) / f"task-{task_id}.pkl"
                        task_id += 1
                        result_file = task_file.with_suffix(".json")
                        with task_file.open("wb") as stream:
                            pickle.dump(
                                dict(
                                    tasks=chunk,
                                    chemistry_config=chemistry_config,
                                    options=options,
                                ),
                                stream,
                                pickle.HIGHEST_PROTOCOL,
                            )
                        commands.append(
                            [
                                sys.executable,
                                "-m",
                                worker_module,
                                "--task",
                                str(task_file),
                                "--result",
                                str(result_file),
                            ]
                        )
                    if not commands:
                        break
                    crashed = []
                    run_parallel_subprocesses(
                        commands,
                        max_workers=min(num_workers, len(commands)),
                        print_output=False,
                        env=env,
                        on_complete=on_complete,
                        on_error=lambda command, error: crashed.append(command),
                        progress=progress,
                    )
                    for command in crashed:
                        retry_individually(command)
                    if crashed:
                        progress.set_postfix(
                            prepared=len(files), skipped=len(skipped), refresh=False
                        )
    with (output / "manifest.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "file",
                "smiles",
                "record_indexes",
                "num_input_records",
                "num_valid_samples",
                "rejected_sample_count",
                "rejection_log",
                "num_branch_groups",
                "num_teacher_nodes",
                "num_transition_states",
                "num_positive_transitions",
                "num_ion_states",
                "max_teacher_reaction_count",
                "num_nodes",
                "num_edges",
                "assignment_score",
                "assignment_score_without_precursor",
                *SEARCH_STAT_FIELDS,
                "status",
                "skip_category",
                "reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda row: row["smiles"]))
    with (output / "assignment_scores.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "structure_file",
                "sample_index",
                "record_index",
                "assignment_score",
                "assignment_score_without_precursor",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                score_rows, key=lambda row: (row["structure_file"], row["sample_index"])
            )
        )
    (output / "skipped_sources.json").write_text(json.dumps(skipped, indent=2))
    (output / "reaction_statistics.json").write_text(
        json.dumps(
            dict(
                architecture=generator.architecture,
                **_search_maxima(completed_search_stats),
                search_limits=dict(
                    max_unique_fragment_smiles=max_unique_fragment_smiles,
                    max_reaction_combinations=max_reaction_combinations,
                ),
                num_skipped_sources_by_category=dict(
                    sorted(
                        Counter(
                            source.get("category", SKIP_ERROR) for source in skipped
                        ).items()
                    )
                ),
                num_limit_skipped_sources=dict(
                    sorted(
                        Counter(
                            source["limit"]
                            for source in skipped
                            if source.get("category") == SKIP_LIMIT_EXCEEDED
                        ).items()
                    )
                ),
                num_teacher_nodes=sum(
                    row.get("num_teacher_nodes", 0) or 0 for row in manifest_rows
                ),
                num_positive_transitions=sum(
                    row.get("num_positive_transitions", 0) or 0 for row in manifest_rows
                ),
                mean_teacher_nodes_per_sample=sum(
                    row.get("num_teacher_nodes", 0) or 0 for row in manifest_rows
                )
                / max(prepared_records, 1),
                mean_positive_edges_per_sample=sum(
                    row.get("num_positive_transitions", 0) or 0 for row in manifest_rows
                )
                / max(prepared_records, 1),
                num_branch_groups=sum(
                    row.get("num_branch_groups", 0) or 0 for row in manifest_rows
                ),
                num_transition_states=sum(
                    row.get("num_transition_states", 0) or 0 for row in manifest_rows
                ),
                num_ion_states=sum(
                    row.get("num_ion_states", 0) or 0
                    for row in manifest_rows
                ),
                max_teacher_reaction_count=max(
                    (
                        row.get("max_teacher_reaction_count", 0) or 0
                        for row in manifest_rows
                    ),
                    default=0,
                ),
                num_sources=len(files),
                num_samples=prepared_records,
                num_metadata_valid_records=len(dataset),
                num_skipped_sources=len(skipped),
                num_skipped_records=sum(
                    len(source["record_indexes"]) for source in skipped
                ),
                num_rejected_records=sum(
                    row["rejected_sample_count"]
                    for row in manifest_rows
                    if row["status"] == "completed"
                ),
                num_excluded_records=inspection["excludedRecords"],
                num_workers=num_workers,
                chunk_size=chunk_size,
                chemistry_config=chemistry_config,
                minimum_relative_intensity=minimum_relative_intensity,
            ),
            indent=2,
        )
    )
    return sorted(files)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Legacy single input; use --datasets-json for dataset lists.")
    parser.add_argument("--datasets-json", help="JSON list of dataset paths, per-file mappings and split assignments.")
    parser.add_argument("--dataset-mode", choices=("automatic", "assigned"), default="automatic")
    parser.add_argument("--output-dir", required=True, type=output_directory)
    parser.add_argument(
        "--save-all-edges", action="store_true",
        help="Save all candidate edges and nodes; edges beyond the root and positive-path endpoints are ignored by branch loss.",
    )
    configure_chemistry_options(parser)
    parser.add_argument(
        "--pretrained-pattern-model",
        action="append",
        help="Add every pretrained reaction pattern from a prediction checkpoint or .reaction-model.pt; repeatable",
    )
    parser.add_argument(
        "--reaction-models-json",
        help="Reaction definitions and pretrained selections; new pattern architecture is configured during training",
    )
    for name, default in (
        ("smiles", "SMILES"),
        ("adduct-type", "AdductType"),
        ("collision-energy", "CollisionEnergy"),
        ("precursor-mz", "PrecursorMZ"),
    ):
        parser.add_argument(
            "--" + name + "-column", default=default, help="Dataset metadata column."
        )
        parser.add_argument(
            "--validation-" + name + "-column",
            help="Validation dataset column, if it differs from --" + name + "-column.",
        )
    parser.add_argument(
        "--max-unique-fragment-smiles",
        type=int,
        default=-1,
        help="Skip a source once its materialized fragments exceed this many distinct canonical SMILES (the source itself is not counted); -1 is unlimited.",
    )
    parser.add_argument(
        "--max-reaction-combinations",
        type=int,
        default=-1,
        help="Skip a source once its reaction search examines more than this many raw reaction combinations (all sizes up to max_reaction_count), before RDKit runs them; -1 is unlimited.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel worker processes per split; 1 runs serially.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="SMILES groups dispatched per worker chunk.",
    )
    parser.add_argument(
        "--validation-input",
        help="Optional held-out spectrum dataset. Produces train_structures and validation_structures.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        help="Split unique SMILES when validation input is omitted. With --validation-input, instead caps validation to this fraction of training's unique SMILES count after removing molecules shared with training.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Fraction of input canonical molecules reserved for final testing; default 0.")
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--minimum-relative-intensity", type=float, default=0.0)
    parser.add_argument(
        "--overwrite",
        type=int,
        choices=(0, 1),
        default=0,
        help="Allow replacing existing structure files (0 or 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.input and not args.datasets_json:
        raise ValueError("Supply --datasets-json or --input.")
    if args.datasets_json and (args.input or args.validation_input):
        raise ValueError("Use dataset lists or legacy inputs, not both.")
    # Decide whether an existing destination may be replaced before loading or
    # validating either dataset; those checks can be expensive on large files.
    _confirm_existing_output(args)
    config = resolve_chemistry_options(args)
    records = json.loads(args.reaction_models_json) if args.reaction_models_json else []
    if args.pretrained_pattern_model:
        from fredis.ml.training.fragment_tree_training.pattern_models import (
            inspect_model,
        )

        for path in args.pretrained_pattern_model:
            records.extend(
                dict(
                    pattern=item["pattern"],
                    source=path,
                    source_pattern_id=item["pattern"]["pattern_id"],
                )
                for item in inspect_model(path)["patterns"]
            )
    if records:
        from fredis.ml.training.fragment_tree_training.pattern_models import (
            prepare_pattern_models,
        )

        config = prepare_pattern_models(config, records)
    validate_limits(args.max_unique_fragment_smiles, args.max_reaction_combinations)
    plan = None
    if args.datasets_json:
        from .dataset_plan import DEFAULT_MAPPING, plan_datasets

        plan = plan_datasets(json.loads(args.datasets_json), args.dataset_mode,
                             create_preparation_context(config).fragmenter, config["symbols"],
                             args.validation_ratio or 0, args.test_ratio, args.validation_seed)
        dataset = plan["combined"]
        reports = plan["reports"]
        validation_mapping = DEFAULT_MAPPING
        args.smiles_column = "SMILES"
        args.adduct_type_column = "AdductType"
        args.collision_energy_column = "CollisionEnergy"
        args.precursor_mz_column = "PrecursorMZ"
    else:
        dataset = load_spectrum_dataset(args.input)
        # Exclude invalid metadata before collecting adducts or splitting molecules.
        validation_dataset = (
            load_spectrum_dataset(args.validation_input) if args.validation_input else None
        )
        checker = create_preparation_context(config).fragmenter
        mapping = dict(
            smilesColumn=args.smiles_column,
            adductTypeColumn=args.adduct_type_column,
            collisionEnergyColumn=args.collision_energy_column,
            precursorMzColumn=args.precursor_mz_column,
        )
        # Falls back to the training column name for anything not explicitly
        # overridden; only takes effect when validation is a separate dataset
        # (--validation-input) since an auto ratio split reuses the same columns.
        validation_mapping = dict(
            smilesColumn=args.validation_smiles_column or args.smiles_column,
            adductTypeColumn=args.validation_adduct_type_column or args.adduct_type_column,
            collisionEnergyColumn=args.validation_collision_energy_column
            or args.collision_energy_column,
            precursorMzColumn=args.validation_precursor_mz_column
            or args.precursor_mz_column,
        )
        reports = {}
        for name, data in (("train", dataset), ("validation", validation_dataset)):
            if data is None:
                continue
            report = inspect_records(
                data,
                checker,
                mapping if name == "train" else validation_mapping,
                symbols=config["symbols"],
            )
            reports[name] = report["invalidRecords"]
            filtered = data[report["validIndexes"]]
            if not len(filtered):
                raise ValueError(name + " dataset contains no valid records to prepare.")
            if name == "train":
                dataset = filtered
            else:
                validation_dataset = filtered
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            dict(
                event="excluded_records",
                counts={name: len(rows) for name, rows in reports.items()},
            )
        ),
        flush=True,
    )
    config["adduct_type_strs"] = list(
        create_preparation_context(config).adduct_type_strs
    )
    kwargs = dict(
        chemistry_config=config,
        smiles_column=args.smiles_column,
        adduct_type_column=args.adduct_type_column,
        collision_energy_column=args.collision_energy_column,
        precursor_mz_column=args.precursor_mz_column,
        minimum_relative_intensity=args.minimum_relative_intensity,
        overwrite=bool(args.overwrite),
        save_all_edges=args.save_all_edges,
        max_unique_fragment_smiles=args.max_unique_fragment_smiles,
        max_reaction_combinations=args.max_reaction_combinations,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    preparation_config = dict(
        input=args.input,
        validation_input=args.validation_input,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        validation_seed=args.validation_seed,
        output_dir=args.output_dir,
        # Blank means "same as the training column"; only a real override is persisted,
        # so reloading a saved configuration keeps showing it as inherited, not explicit.
        validation_smiles_column=args.validation_smiles_column or "",
        validation_adduct_type_column=args.validation_adduct_type_column or "",
        validation_collision_energy_column=args.validation_collision_energy_column
        or "",
        validation_precursor_mz_column=args.validation_precursor_mz_column or "",
        **kwargs,
    )
    if plan is not None:
        preparation_config.update(datasets=plan["entries"], dataset_mode=args.dataset_mode)
    kwargs["preparation_config"] = preparation_config
    validation_kwargs = {
        **kwargs,
        "smiles_column": validation_mapping["smilesColumn"],
        "adduct_type_column": validation_mapping["adductTypeColumn"],
        "collision_energy_column": validation_mapping["collisionEnergyColumn"],
        "precursor_mz_column": validation_mapping["precursorMzColumn"],
    }
    if plan is not None:
        assignments = plan["assignments"]
        train, validation, test = (plan["splits"][name] for name in ("train", "validation", "test"))
        validation = validation if len(validation) else None
    else:
        automatic, assignments = split_dataset(
            dataset, args.smiles_column,
            0.0 if args.validation_input else (args.validation_ratio or 0.0),
            args.test_ratio, args.validation_seed,
        )
        train, test = automatic["train"], automatic["test"]
        if args.validation_input:
            # A model must not be scored on a molecule it trained on, so overlapping
            # SMILES are always dropped from validation rather than failing the run;
            # --validation-ratio (optional here) then caps validation to a size
            # proportional to training instead of using every remaining record.
            validation, overlap_report = dedupe_validation(
                dataset,
                validation_dataset,
                args.smiles_column,
                validation_mapping["smilesColumn"],
                ratio=args.validation_ratio or None,
                seed=args.validation_seed,
            )
            if overlap_report["removedOverlapRecords"]:
                print(
                    json.dumps(dict(event="validation_overlap_removed", **overlap_report)),
                    flush=True,
                )
            if (
                "targetSmiles" in overlap_report
                and overlap_report["remainingSmiles"] < overlap_report["targetSmiles"]
            ):
                print(
                    json.dumps(
                        dict(
                            event="validation_below_target",
                            **overlap_report,
                            note="Validation has fewer molecule-disjoint SMILES than --validation-ratio requests; using all available records instead of failing.",
                        )
                    ),
                    flush=True,
                )
            if not len(validation):
                print(
                    json.dumps(
                        dict(
                            event="validation_empty",
                            note="No validation records remain after removing molecules shared with training; continuing with training only.",
                        )
                    ),
                    flush=True,
                )
                validation = None
        else:
            validation = automatic["validation"] if args.validation_ratio else None
        if validation is not None:
            for key in canonical_keys(validation[validation_mapping["smilesColumn"] if args.validation_input else args.smiles_column].astype(str)):
                assignments[key] = "validation"
    if not args.overwrite and (
        list(output.rglob("*.preft.pt"))
        or any(
            list((output / name).rglob("*.preft.pt"))
            for name in ("train_structures", "validation_structures", "test_structures")
        )
    ):
        raise FileExistsError(
            "Output already contains training structures. Enable overwrite or choose another directory."
        )
    workbench_path = output / "fragment-tree.training-data.pft"
    try:
        if workbench_path.exists():
            previous = json.loads(workbench_path.read_text())
        elif (output / "fragment-tree.training-data-config.pft").exists():
            previous = json.loads((output / "fragment-tree.training-data-config.pft").read_text()).get(
                "workbench_config", {}
            )
        elif (output / "preparation_config.json").exists():
            previous = json.loads((output / "preparation_config.json").read_text()).get(
                "workbench_config", {}
            )
        elif (output / "train_structures/fragment-tree.training-data.pft").exists():
            previous = json.loads(
                (
                    output / "train_structures/fragment-tree.training-data.pft"
                ).read_text()
            )
        else:
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, json.JSONDecodeError):
        previous = {}
    with stage_pretrained_models(config) as staged:
        _reset_output(
            output,
            bool(args.overwrite),
            [
                value
                for value in ([entry["path"] for entry in plan["entries"]] + [args.params]
                              if plan is not None else (args.input, args.validation_input, args.params))
                if value
            ],
        )
        archived = staged.install(output)
    config.clear()
    config.update(archived)
    workbench_path.unlink(missing_ok=True)
    aliases = {
        "validation_input": "validationInput",
        "validation_ratio": "validationRatio",
        "test_ratio": "testRatio",
        "dataset_mode": "datasetMode",
        "validation_seed": "validationSeed",
        "output_dir": "outputDir",
        "smiles_column": "smilesColumn",
        "adduct_type_column": "adductTypeColumn",
        "collision_energy_column": "collisionEnergyColumn",
        "precursor_mz_column": "precursorMzColumn",
        "validation_smiles_column": "validationSmilesColumn",
        "validation_adduct_type_column": "validationAdductTypeColumn",
        "validation_collision_energy_column": "validationCollisionEnergyColumn",
        "validation_precursor_mz_column": "validationPrecursorMzColumn",
        "minimum_relative_intensity": "minimumRelativeIntensity",
        "max_unique_fragment_smiles": "maxUniqueFragmentSmiles",
        "max_reaction_combinations": "maxReactionCombinations",
        "save_all_edges": "saveAllEdges",
        "num_workers": "numWorkers",
        "chunk_size": "chunkSize",
    }
    restored = {**previous, "application": "fragment-tree-data-preparation"}
    for key, value in preparation_config.items():
        if key == "chemistry_config":
            continue
        if key == "validation_ratio" and value is None:
            if args.validation_input:
                continue
            value = 0
        restored[aliases.get(key, key)] = value
    restored.update(
        fragmenterParams=config["fragmenter_params"],
        symbols=config["symbols"],
        reactionModels=config.get("reaction_models", []),
    )
    restored.pop("modelConfig", None)
    restored.pop("params", None)
    if plan is not None:
        for key in ("input", "trainInput", "validationInput"):
            restored.pop(key, None)
    else:
        restored.pop("datasets", None)
        restored.pop("datasetMode", None)
    preparation_config["workbench_config"] = restored
    for name in ("fragment-tree.training-data-config.pft", "preparation_config.json"):
        (output / name).write_text(json.dumps(preparation_config, indent=2))
    (output / "split_manifest.json").write_text(json.dumps({
        "version": 1, "key": "rdkit-canonical-isomeric-smiles", "seed": args.validation_seed,
        "validationRatio": args.validation_ratio, "testRatio": args.test_ratio,
        "mode": args.dataset_mode if plan is not None else "legacy",
        "sources": plan["sources"] if plan is not None else [],
        "assignments": dict(sorted(assignments.items())),
    }, indent=2))
    splits = [("train", train)]
    if (plan is not None and len(test)) or (plan is None and args.test_ratio):
        splits.append(("test", test))
    if validation is not None:
        splits.append(("validation", validation))
    for name, prepared_split in splits:
        directory = output / (name + "_structures")
        directory.mkdir(parents=True, exist_ok=True)
        split_chemistry = relocate_sources(config, output, directory)
        split_config = {
            **restored,
            "reactionModels": split_chemistry.get("reaction_models", []),
            "split": name,
            "status": "running",
        }
        (directory / "fragment-tree.training-data.pft").write_text(
            json.dumps(split_config, indent=2)
        )
        # The split is newly empty after resetting Output Directory. Do not delete its configuration.
        create_reaction_training_data(
            dataset=prepared_split,
            output_dir=directory,
            split=name,
            **{
                **(validation_kwargs if name == "validation" and args.validation_input else kwargs),
                "chemistry_config": split_chemistry,
                "overwrite": False,
            },
            _check_existing=False,
        )
        split_config["status"] = "completed"
        (directory / "fragment-tree.training-data.pft").write_text(
            json.dumps(split_config, indent=2)
        )
    (output / "invalid_records.json").write_text(json.dumps(reports, indent=2))
    for name in ("fragment-tree.training-data-config.pft", "preparation_config.json"):
        (output / name).write_text(json.dumps(preparation_config, indent=2))


if __name__ == "__main__":
    main()
