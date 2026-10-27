from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest

from fredis.cli.molecule.group import MoleculeGroup


class TestMoleculeDescriptorsCLI(unittest.TestCase):
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="fredis")
        subparsers = parser.add_subparsers(dest="command", required=True)
        MoleculeGroup().register(subparsers)
        return parser

    def run_command(self, argv: list[str]) -> str:
        parser = self.build_parser()
        args = parser.parse_args(argv)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            args.func(args)
        return output.getvalue()

    def test_calculates_default_descriptors_as_json(self) -> None:
        payload = json.loads(self.run_command(["molecule", "descriptors", "CCO"]))
        self.assertEqual(payload["canonical_smiles"], "CCO")
        self.assertAlmostEqual(payload["descriptors"]["ExactMolWt"], 46.041864812)
        self.assertEqual(payload["descriptors"]["HeavyAtomCount"], 3.0)

    def test_can_select_descriptors_and_use_group_alias(self) -> None:
        payload = json.loads(
            self.run_command(
                ["mol", "desc", "CCO", "-d", "MolWt", "-d", "FormalCharge"]
            )
        )
        self.assertEqual(tuple(payload["descriptors"]), ("MolWt", "FormalCharge"))
        self.assertAlmostEqual(payload["descriptors"]["MolWt"], 46.069, places=3)

    def test_tsv_output(self) -> None:
        output = self.run_command(
            ["molecule", "descriptors", "[NH4+]", "-d", "FormalCharge", "--format", "tsv"]
        )
        self.assertEqual(output, "descriptor\tvalue\nFormalCharge\t1\n")

    def test_rejects_invalid_smiles(self) -> None:
        parser = self.build_parser()
        args = parser.parse_args(["molecule", "descriptors", "not-a-smiles"])
        with self.assertRaisesRegex(SystemExit, "Invalid SMILES"):
            args.func(args)


if __name__ == "__main__":
    unittest.main()
