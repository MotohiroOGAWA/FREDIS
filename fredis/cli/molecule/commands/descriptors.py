from __future__ import annotations

import argparse
import json

from rdkit import Chem, rdBase

from ...base import CLICommand
from fredis.domain.molecule.descriptors import (
    DEFAULT_DESCRIPTOR_NAMES,
    SUPPORTED_DESCRIPTOR_NAMES,
    compute_descriptors,
)


class MolecularDescriptorsCommand(CLICommand):
    name = "descriptors"
    aliases = ("descriptor", "desc")
    help = "Calculate molecular descriptors from a SMILES string."
    description = (
        "Parse a SMILES string with RDKit and calculate FREDIS molecular "
        "descriptors. Output is JSON by default."
    )
    order = 10

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("smiles", help="Molecule represented as a SMILES string.")
        parser.add_argument(
            "-d",
            "--descriptor",
            dest="descriptors",
            action="append",
            choices=SUPPORTED_DESCRIPTOR_NAMES,
            help=(
                "Descriptor to calculate. Repeat this option to select multiple "
                "descriptors. By default all FREDIS descriptors are calculated."
            ),
        )
        parser.add_argument(
            "--format",
            choices=("json", "tsv"),
            default="json",
            help="Output format (default: json).",
        )

    def run(self, args: argparse.Namespace) -> None:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(args.smiles)
        if mol is None:
            raise SystemExit(f"Invalid SMILES: {args.smiles}")

        names = tuple(args.descriptors or DEFAULT_DESCRIPTOR_NAMES)
        values = compute_descriptors(mol, names)
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        if args.format == "tsv":
            print("descriptor\tvalue")
            for name, value in values.items():
                print(f"{name}\t{value:.12g}")
            return

        print(
            json.dumps(
                {
                    "smiles": args.smiles,
                    "canonical_smiles": canonical_smiles,
                    "descriptors": values,
                },
                indent=2,
                allow_nan=False,
            )
        )
