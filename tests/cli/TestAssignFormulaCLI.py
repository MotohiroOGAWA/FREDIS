from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fredis.cli.assign_formula.commands.assign import AssignFormulaCommand


class TestAssignFormulaCLI(unittest.TestCase):
    def run_command(self, argv):
        command = AssignFormulaCommand()
        parser = argparse.ArgumentParser()
        command.configure(parser)
        args = parser.parse_args(argv)
        with patch('fredis.cli.assign_formula.commands.assign.assign_formulas') as single, \
             patch('fredis.cli.assign_formula.commands.assign.parallel_assign_formulas') as parallel:
            command.run(args)
        return single, parallel

    def test_directory_outputs_and_default_report(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'input.msds'
            for flag in ('--output-dir', '--output_dir', '-o'):
                directory = Path(root) / flag.replace('-', '')
                single, parallel = self.run_command([str(source), flag, str(directory)])
                self.assertFalse(parallel.called)
                self.assertEqual(single.call_args.kwargs['output_file'], str(directory / 'input_assigned_formula.msds'))
                self.assertEqual(single.call_args.kwargs['report_file'], str(directory / 'input_assigned_formula_assign_formula_report.tsv'))

    def test_parallel_keep_temp_is_opt_in(self):
        with tempfile.TemporaryDirectory() as root:
            for keep in (False, True):
                single, parallel = self.run_command([str(Path(root) / 'input.msds'), '--num_workers', '2', '--chunk_size', '1', *(['--keep-temp'] if keep else [])])
                self.assertFalse(single.called)
                self.assertEqual(parallel.call_args.kwargs['keep_temp'], keep)
                self.assertEqual(Path(parallel.call_args.kwargs['output_file']).parent, Path(root) / 'input_assign_formula')

    def test_existing_report_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            report = directory / 'input_assigned_formula_assign_formula_report.tsv'
            report.write_text('do not replace')
            with self.assertRaises(FileExistsError):
                self.run_command([str(directory / 'input.msds'), '-o', root])
            self.assertEqual(report.read_text(), 'do not replace')
            single, _ = self.run_command([str(directory / 'input.msds'), '-o', root, '--overwrite'])
            self.assertTrue(single.called)

    def test_legacy_explicit_filename(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'legacy.msds'
            single, _ = self.run_command([str(Path(root) / 'input.msds'), '--output_file', str(output)])
            self.assertEqual(single.call_args.kwargs['output_file'], str(output))
