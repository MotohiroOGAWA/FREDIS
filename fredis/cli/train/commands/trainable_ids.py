from __future__ import annotations
import argparse
import json
from pathlib import Path
from ...base import CLICommand, output_directory
from ...chemistry_options import configure_chemistry_options, resolve_chemistry_options
from fredis.analysis.training_eligibility.native import extract


class TrainableIdsCommand(CLICommand):
    name = "trainable-ids"
    help = "Export SpecIDs verified by FREDIS preparation, forward, loss and backward."
    order = 40

    def configure(self, parser):
        parser.add_argument("--input", required=True)
        parser.add_argument("--output-dir", required=True, type=output_directory)
        configure_chemistry_options(parser)
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--model-architecture")
        group.add_argument("--model-architecture-json")
        for flag, default in [
            ("spec-id-column", "SpecID"),
            ("smiles-column", "SMILES"),
            ("adduct-type-column", "AdductType"),
            ("collision-energy-column", "CollisionEnergy"),
            ("precursor-mz-column", "PrecursorMZ"),
        ]:
            parser.add_argument("--" + flag, default=default)
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--minimum-relative-intensity", type=float, default=0.0)
        parser.add_argument("--max-unique-fragment-smiles", type=int, default=10000)
        parser.add_argument("--max-reaction-combinations", type=int, default=10000)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--keep-export", action="store_true")

    def run(self, args):
        default = (
            Path(__file__).resolve().parents[3]
            / "presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.fragmenter.pft"
        )
        fragmenter = json.loads(default.read_text())
        fragmenter["fragment_ion_tree_builder"]["max_reaction_count"] = 1
        chemistry = resolve_chemistry_options(
            args, base=dict(fragmenter_params=fragmenter)
        )
        architecture = (
            json.loads(Path(args.model_architecture).read_text())
            if args.model_architecture
            else (
                json.loads(args.model_architecture_json)
                if args.model_architecture_json
                else None
            )
        )
        if architecture:
            architecture = architecture.get("model_config", architecture)

        def progress(event):
            print(json.dumps(dict(event="trainable_progress", **event)), flush=True)

        result = extract(
            args.input,
            args.output_dir,
            chemistry,
            model_config=architecture,
            mapping=dict(
                smilesColumn=args.smiles_column,
                adductTypeColumn=args.adduct_type_column,
                collisionEnergyColumn=args.collision_energy_column,
                precursorMzColumn=args.precursor_mz_column,
            ),
            id_column=args.spec_id_column,
            workers=args.workers,
            batch_size=args.batch_size,
            seed=args.seed,
            resume=args.resume,
            keep_export=args.keep_export,
            minimum_relative_intensity=args.minimum_relative_intensity,
            max_unique_fragment_smiles=args.max_unique_fragment_smiles,
            max_reaction_combinations=args.max_reaction_combinations,
            progress=progress,
        )
        print(
            f'FREDIS trainable SpecIDs: {result["trainable"]} / {result["total_records"]}; excluded: {result["excluded"]}'
        )
