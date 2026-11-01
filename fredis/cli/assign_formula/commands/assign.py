from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ...base import CLICommand
from fredis.domain.mass.tolerance import parse_mass_tolerance
from fredis.domain.formula.assign_formula import (
    assign_formulas,
    parallel_assign_formulas,
)


class AssignFormulaCommand(CLICommand):
    name = "run"
    help = "Assign possible subformulas to MS/MS peaks."
    description = (
        "Assign possible subformulas to MS/MS peaks based on precursor formula."
    )
    order = 10

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input_file", type=str)
        output = parser.add_mutually_exclusive_group()
        output.add_argument("-o", "--output-dir", "--output_dir", type=str, default=None,
                            help="Directory for the assigned .msds dataset and automatic TSV report.")
        output.add_argument("--output_file", type=str, default=None,
                            help="Legacy explicit output file (prefer --output-dir).")
        parser.add_argument("--keep-temp", action="store_true",
                            help="Keep parallel chunk datasets and reports after execution (default: remove).")

        parser.add_argument(
            "-col_cov",
            "--calc_formula_coverage_column",
            type=str,
            default="CalcFormulaCov",
        )
        parser.add_argument(
            "-col_smiles",
            "--smiles_column",
            type=str,
            default="SMILES",
        )
        parser.add_argument(
            "-col_adduct",
            "--adduct_type_column",
            type=str,
            default="AdductType",
        )
        parser.add_argument(
            "-col_precursor_mz",
            "--precursor_mz_column",
            type=str,
            default="PrecursorMZ",
        )

        parser.add_argument(
            "-tol",
            "--tolerance",
            type=str,
            default="0.01Da,10ppm",
            help="Mass tolerance string, e.g. 0.01Da,10ppm.",
        )

        parser.add_argument(
            "-max_formula",
            "--max_formula_candidates",
            type=int,
            default=None,
            help=(
                "Maximum estimated number of subformula candidates. "
                "If exceeded, the spectrum group is skipped."
            ),
        )
        parser.add_argument(
            "-timeout",
            "--timeout_sec",
            type=float,
            default=float("inf"),
        )

        parser.add_argument("--overwrite", "-ow", action="store_true")
        parser.add_argument("--add_finished_tag", action="store_true")

        parser.add_argument("--num_workers", "-n_workers", type=int, default=1)
        parser.add_argument("--chunk_size", type=int, default=-1)

        parser.add_argument(
            "--report_file",
            type=str,
            default=None,
            help="TSV file path for skipped/error records.",
        )
        parser.add_argument(
            "--hydrogen_delta",
            type=int,
            default=1,
            help="Additional hydrogen count allowed during subformula generation.",
        )
        parser.add_argument(
            "--enable-neutral-loss",
            action="store_true",
            help="Output neutral-loss labels such as C5H3(-CH3) when possible.",
        )

    def run(self, args: argparse.Namespace) -> None:
        mass_tolerance = parse_mass_tolerance(args.tolerance)

        input_path = Path(args.input_file).resolve()
        if args.output_file is not None:
            output_file = Path(args.output_file).resolve()
        else:
            directory = Path(args.output_dir).resolve() if args.output_dir else input_path.with_name(input_path.stem + "_assign_formula")
            directory.mkdir(parents=True, exist_ok=True)
            output_file = directory / (input_path.stem + "_assigned_formula.msds")
            if args.report_file:
                raise ValueError("--report_file is only supported with legacy --output_file; directory output generates its report automatically.")
        from fredis.domain.formula._report import default_report_file
        report_file = Path(args.report_file).resolve() if args.report_file else Path(default_report_file(str(output_file)))
        if len({input_path, output_file.resolve(), report_file.resolve()}) != 3:
            raise ValueError("Input dataset, output dataset and report must have different paths.")
        if not args.overwrite:
            for target in (output_file, report_file):
                if target.exists():
                    raise FileExistsError(f"Output already exists: {target}. Use --overwrite to replace it.")
        start_time = time.time()

        if args.num_workers > 1:
            if args.chunk_size <= 0:
                print(
                    "Error: --chunk_size must be positive when --num_workers > 1.",
                    file=sys.stderr,
                )
                sys.exit(1)

            parallel_assign_formulas(
                executable=sys.executable,
                argv=sys.argv,
                num_workers=args.num_workers,
                chunk_size=args.chunk_size,
                input_file=args.input_file,
                output_file=str(output_file),
                mass_tolerance=mass_tolerance,
                calc_formula_coverage_column=args.calc_formula_coverage_column,
                smiles_column=args.smiles_column,
                adduct_type_column=args.adduct_type_column,
                precursor_mz_column=args.precursor_mz_column,
                overwrite=args.overwrite,
                timeout_sec=args.timeout_sec,
                max_formula_candidates=args.max_formula_candidates,
                hydrogen_delta=args.hydrogen_delta,
                report_file=str(report_file),
                enable_neutral_loss=args.enable_neutral_loss,
                keep_temp=args.keep_temp,
            )
        else:
            assign_formulas(
                input_file=args.input_file,
                output_file=str(output_file),
                mass_tolerance=mass_tolerance,
                calc_formula_coverage_column=args.calc_formula_coverage_column,
                smiles_column=args.smiles_column,
                adduct_type_column=args.adduct_type_column,
                precursor_mz_column=args.precursor_mz_column,
                overwrite=args.overwrite,
                timeout_sec=args.timeout_sec,
                max_formula_candidates=args.max_formula_candidates,
                hydrogen_delta=args.hydrogen_delta,
                add_finished_tag=args.add_finished_tag,
                report_file=str(report_file),
                enable_neutral_loss=args.enable_neutral_loss,
            )

        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds.")
