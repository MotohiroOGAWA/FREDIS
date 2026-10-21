from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from fredis.libs.msentity.msentity import MSDataset, load_ms_dataset
from fredis.libs.mmkit.mmkit import Compound, Formula
from fredis.domain.mass.tolerance import MassTolerance
from fredis.utils.parallel_subprocess import run_parallel_subprocesses

from ._assignment import (
    assign_record_formulas,
    group_record_indexes_by_precursor_formula,
)
from ._candidate import estimate_subformula_candidate_count
from .utils import get_possible_sub_formulas
from ._io import (
    initialize_output_columns,
    is_finished_dataset,
    prepare_dataset_io,
    save_dataset,
    try_move_peak_column_to_front,
    validate_columns,
)
from ._report import (
    AssignFormulaReportRow,
    default_report_file,
    get_record_spec_id,
    write_report_tsv,
)
from ._parallel import (
    delete_arg,
    make_chunks_by_precursor_formula,
    merge_report_tsvs,
    replace_arg,
    try_add_arg,
)


def assign_formulas(
    *,
    input_file: str,
    output_file: Optional[str],
    mass_tolerance: MassTolerance,
    calc_formula_coverage_column: str = "CalcFormulaCov",
    smiles_column: str = "SMILES",
    adduct_type_column: str = "AdductType",
    precursor_mz_column: str = "PrecursorMZ",
    overwrite: bool = False,
    timeout_sec: float = float("inf"),
    max_formula_candidates: Optional[int] = None,
    hydrogen_delta: int = 1,
    save_interval_sec: float = float("inf"),
    add_finished_tag: bool = False,
    report_file: Optional[str] = None,
    enable_neutral_loss: bool = False,
) -> MSDataset:
    """Assign possible subformulas to MS/MS peaks.

    This function writes:
        - peak-level formula matches into CalcFormula
        - record-level coverage into CalcFormulaCov
    """

    dataset, output_file = prepare_dataset_io(
        input_file=input_file,
        output_file=output_file,
        overwrite=overwrite,
    )

    if report_file is None:
        report_file = default_report_file(output_file)

    report_rows: list[AssignFormulaReportRow] = []
    success_record_indexes: list[int] = []

    validate_columns(
        dataset,
        smiles_column=smiles_column,
        adduct_type_column=adduct_type_column,
        precursor_mz_column=precursor_mz_column,
    )

    calc_formula_column, calc_formula_coverage_column = initialize_output_columns(
        dataset,
        calc_formula_coverage_column=calc_formula_coverage_column,
        overwrite=overwrite,
    )

    precursor_formula_groups = group_record_indexes_by_precursor_formula(
        dataset=dataset,
        smiles_column=smiles_column,
        adduct_type_column=adduct_type_column,
        precursor_mz_column=precursor_mz_column,
        mass_tolerance=mass_tolerance,
        error_column=calc_formula_coverage_column,
    )

    total_records = sum(len(v) for v in precursor_formula_groups.values())
    pbar = tqdm(
        total=total_records,
        desc="Assigning formulas to peaks",
        mininterval=1.0,
    )

    last_save_time = time.time()
    success_count = 0
    progress_count = 0

    for precursor_formula_value, record_indexes in precursor_formula_groups.items():
        precursor_formula = Formula.parse(precursor_formula_value)

        estimated_candidate_count = estimate_subformula_candidate_count(
            precursor_formula,
            hydrogen_delta=hydrogen_delta,
        )

        if (
            max_formula_candidates is not None
            and estimated_candidate_count > max_formula_candidates
        ):
            message = (
                "TooManyFormulaCandidates: "
                f"formula={precursor_formula_value}, "
                f"estimated={estimated_candidate_count}, "
                f"max={max_formula_candidates}"
            )
            print(f"[WARN] {message}", file=sys.stderr)

            for record_index in record_indexes:
                record = dataset[record_index]

                report_rows.append(
                    AssignFormulaReportRow(
                        index=int(record_index),
                        spec_id=get_record_spec_id(record),
                        status="TooManyFormulaCandidates",
                        message=message,
                        formula=str(precursor_formula_value),
                        smiles=str(record[smiles_column]),
                        adduct_type=str(record[adduct_type_column]),
                        precursor_mz=str(record[precursor_mz_column]),
                        estimated_candidates=str(estimated_candidate_count),
                        max_candidates=str(max_formula_candidates),
                    )
                )

                pbar.update(1)
                progress_count += 1

            continue

        try:
            possible_formulas = get_possible_sub_formulas(
                precursor_formula,
                hydrogen_delta=hydrogen_delta,
                timeout=timeout_sec,
            )
        except TimeoutError:
            message = (
                "TimeoutGeneratingFormulas: "
                f"formula={precursor_formula_value}, "
                f"timeout_sec={timeout_sec}"
            )
            print(f"[WARN] {message}", file=sys.stderr)

            for record_index in record_indexes:
                record = dataset[record_index]

                report_rows.append(
                    AssignFormulaReportRow(
                        index=int(record_index),
                        spec_id=get_record_spec_id(record),
                        status="TimeoutGeneratingFormulas",
                        message=message,
                        formula=str(precursor_formula_value),
                        smiles=str(record[smiles_column]),
                        adduct_type=str(record[adduct_type_column]),
                        precursor_mz=str(record[precursor_mz_column]),
                    )
                )

                pbar.update(1)
                progress_count += 1

            continue
        except Exception as e:
            message = (
                "ErrorGeneratingFormulas: "
                f"formula={precursor_formula_value}, error={e}"
            )
            print(f"[WARN] {message}", file=sys.stderr)

            for record_index in record_indexes:
                record = dataset[record_index]

                report_rows.append(
                    AssignFormulaReportRow(
                        index=int(record_index),
                        spec_id=get_record_spec_id(record),
                        status="ErrorGeneratingFormulas",
                        message=message,
                        formula=str(precursor_formula_value),
                        smiles=str(record[smiles_column]),
                        adduct_type=str(record[adduct_type_column]),
                        precursor_mz=str(record[precursor_mz_column]),
                    )
                )

                pbar.update(1)
                progress_count += 1

            continue

        if (
            max_formula_candidates is not None
            and len(possible_formulas) > max_formula_candidates
        ):
            message = (
                "TooManyGeneratedFormulaCandidates: "
                f"formula={precursor_formula_value}, "
                f"generated={len(possible_formulas)}, "
                f"max={max_formula_candidates}"
            )
            print(f"[WARN] {message}", file=sys.stderr)

            for record_index in record_indexes:
                record = dataset[record_index]

                report_rows.append(
                    AssignFormulaReportRow(
                        index=int(record_index),
                        spec_id=get_record_spec_id(record),
                        status="TooManyGeneratedFormulaCandidates",
                        message=message,
                        formula=str(precursor_formula_value),
                        smiles=str(record[smiles_column]),
                        adduct_type=str(record[adduct_type_column]),
                        precursor_mz=str(record[precursor_mz_column]),
                        generated_candidates=str(len(possible_formulas)),
                        max_candidates=str(max_formula_candidates),
                    )
                )

                pbar.update(1)
                progress_count += 1

            continue

        for record_index in record_indexes:
            record = dataset[record_index]

            try:
                if not overwrite and str(record[calc_formula_coverage_column]) != "":
                    progress_count += 1
                    pbar.update(1)
                    continue

                original_formula = (
                    Compound.from_smiles(str(record[smiles_column])).formula.plain
                    if enable_neutral_loss
                    else None
                )

                assign_record_formulas(
                    record=record,
                    formula_candidates=possible_formulas,
                    mass_tolerance=mass_tolerance,
                    calc_formula_column=calc_formula_column,
                    calc_formula_coverage_column=calc_formula_coverage_column,
                    original_formula=original_formula,
                )

                success_record_indexes.append(int(record_index))
                success_count += 1

            except Exception as e:
                message = (
                    "ErrorAssigningFormula: " f"record_index={record_index}, error={e}"
                )
                print(f"[WARN] {message}", file=sys.stderr)

                report_rows.append(
                    AssignFormulaReportRow(
                        index=int(record_index),
                        spec_id=get_record_spec_id(record),
                        status="ErrorAssigningFormula",
                        message=message,
                        formula=str(precursor_formula_value),
                        smiles=str(record[smiles_column]),
                        adduct_type=str(record[adduct_type_column]),
                        precursor_mz=str(record[precursor_mz_column]),
                    )
                )

            finally:
                progress_count += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "success": (
                            f"{success_count}/{progress_count}"
                            f"({success_count / progress_count * 100:.1f}%)"
                        )
                    }
                )

                current_time = time.time()
                if current_time - last_save_time > save_interval_sec:
                    save_dataset(dataset, output_file)
                    last_save_time = current_time

    pbar.close()

    success_dataset = dataset[success_record_indexes]

    try_move_peak_column_to_front(
        success_dataset,
        calc_formula_column,
    )

    save_dataset(
        success_dataset,
        output_file,
        add_finished_tag=add_finished_tag,
    )

    write_report_tsv(
        report_rows,
        report_file,
    )

    print(
        f"Saved successful records: {len(success_record_indexes)} " f"to {output_file}"
    )
    print(f"Saved skipped/error report: {len(report_rows)} " f"to {report_file}")

    return success_dataset


def parallel_assign_formulas(
    *,
    executable: str,
    argv: List[str],
    num_workers: int,
    chunk_size: int,
    input_file: str,
    output_file: Optional[str],
    mass_tolerance: MassTolerance,
    calc_formula_coverage_column: str = "CalcFormulaCov",
    smiles_column: str = "SMILES",
    adduct_type_column: str = "AdductType",
    precursor_mz_column: str = "PrecursorMZ",
    overwrite: bool = False,
    timeout_sec: float = float("inf"),
    max_formula_candidates: Optional[int] = None,
    hydrogen_delta: int = 1,
    report_file: Optional[str] = None,
    enable_neutral_loss: bool = False,
) -> None:
    """Assign formulas in parallel by precursor-formula chunks.

    Each subprocess saves:
        - successful records to part_*_done.msds
        - skipped/error records to part_*_report.tsv

    This function merges:
        - all part_*_done.msds files into output_file
        - all part_*_report.tsv files into report_file
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive for parallel processing.")

    if num_workers <= 1:
        raise ValueError("num_workers must be greater than 1 for parallel processing.")

    dataset, output_file = prepare_dataset_io(
        input_file=input_file,
        output_file=output_file,
        overwrite=overwrite,
    )

    if report_file is None:
        report_file = default_report_file(output_file)

    validate_columns(
        dataset,
        smiles_column=smiles_column,
        adduct_type_column=adduct_type_column,
        precursor_mz_column=precursor_mz_column,
    )

    _, calc_formula_coverage_column = initialize_output_columns(
        dataset,
        calc_formula_coverage_column=calc_formula_coverage_column,
        overwrite=overwrite,
    )

    precursor_formula_groups = group_record_indexes_by_precursor_formula(
        dataset=dataset,
        smiles_column=smiles_column,
        adduct_type_column=adduct_type_column,
        precursor_mz_column=precursor_mz_column,
        mass_tolerance=mass_tolerance,
        error_column=None,
    )

    chunks = make_chunks_by_precursor_formula(
        precursor_formula_groups,
        chunk_size=chunk_size,
    )

    if len(chunks) == 0:
        raise RuntimeError("No valid precursor-formula groups were found.")

    print(
        f"Split into {len(chunks)} chunks "
        f"(precursor-formula groups per chunk = {chunk_size})."
    )

    output_path = Path(output_file)
    temp_dir = output_path.parent / f"{output_path.stem}_temp_parallel_assign_formula"
    temp_dir.mkdir(parents=True, exist_ok=True)

    parallel_infos: List[Dict[str, object]] = []

    for chunk_index, record_indexes in enumerate(
        tqdm(chunks, desc="Preparing parallel tasks", mininterval=1.0)
    ):
        temp_input = temp_dir / f"part_{chunk_index}.msds"
        temp_output = temp_dir / f"part_{chunk_index}_done.msds"
        temp_report = temp_dir / f"part_{chunk_index}_report.tsv"

        subset = dataset[record_indexes]

        cmd_args = argv.copy()

        if input_file in cmd_args:
            cmd_args[cmd_args.index(input_file)] = str(temp_input)
        else:
            raise ValueError(
                "input_file was not found in argv. "
                f"input_file={input_file}, argv={argv}"
            )

        replace_arg(cmd_args, "-o", str(temp_output))
        replace_arg(cmd_args, "--output_file", str(temp_output))

        replace_arg(cmd_args, "--num_workers", "1")
        replace_arg(cmd_args, "-n_workers", "1")
        delete_arg(cmd_args, "--chunk_size")

        replace_arg(cmd_args, "--report_file", str(temp_report))
        try_add_arg(cmd_args, "--report_file", str(temp_report))

        try_add_arg(cmd_args, "--overwrite")
        try_add_arg(cmd_args, "--add_finished_tag")

        if max_formula_candidates is not None:
            try_add_arg(
                cmd_args,
                "--max_formula_candidates",
                str(max_formula_candidates),
            )

        try_add_arg(
            cmd_args,
            "--hydrogen_delta",
            str(hydrogen_delta),
        )

        if enable_neutral_loss:
            try_add_arg(cmd_args, "--enable-neutral-loss")

        if timeout_sec != float("inf"):
            try_add_arg(
                cmd_args,
                "--timeout_sec",
                str(timeout_sec),
            )

        command = [
            executable,
            "-m",
            "fredis.cli.main",
            *cmd_args[1:],
        ]

        enabled = True

        if temp_output.exists() and not overwrite:
            enabled = not is_finished_dataset(str(temp_output))

        if enabled:
            save_dataset(
                subset,
                str(temp_input),
                add_finished_tag=False,
            )

        parallel_infos.append(
            {
                "command": command,
                "temp_input": str(temp_input),
                "temp_output": str(temp_output),
                "temp_report": str(temp_report),
                "enabled": enabled,
            }
        )

    commands_list = [
        info["command"] for info in parallel_infos if bool(info["enabled"])
    ]

    if len(commands_list) == 0:
        print("No parallel tasks to run. All chunks are already finished.")
    else:
        run_parallel_subprocesses(
            commands_list=commands_list,
            max_workers=num_workers,
        )

    output_datasets: List[MSDataset] = []

    for info in tqdm(
        parallel_infos,
        desc="Merging chunk outputs",
        mininterval=1.0,
    ):
        temp_output = str(info["temp_output"])

        if not os.path.exists(temp_output):
            print(
                f"[WARN] Chunk output file was not created: {temp_output}",
                file=sys.stderr,
            )
            continue

        chunk_dataset = load_ms_dataset(temp_output)

        if len(chunk_dataset) == 0:
            continue

        output_datasets.append(chunk_dataset)

    if len(output_datasets) == 0:
        merge_report_tsvs(
            [str(info["temp_report"]) for info in parallel_infos],
            report_file,
        )

        raise RuntimeError(
            "No successful chunk outputs were produced. "
            f"See report file: {report_file}"
        )

    if hasattr(MSDataset, "concat"):
        merged_dataset = MSDataset.concat(output_datasets)
    else:
        raise AttributeError("MSDataset.concat() is required for merging chunks.")

    save_dataset(
        merged_dataset,
        output_file,
        add_finished_tag=False,
    )

    merge_report_tsvs(
        [str(info["temp_report"]) for info in parallel_infos],
        report_file,
    )

    print(f"Saved merged successful records to: {output_file}")
    print(f"Saved merged skipped/error report to: {report_file}")
