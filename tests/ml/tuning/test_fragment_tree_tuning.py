import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fredis.ml.tuning.fragment_tree import RESULT_NAME, expand_search, run, summarize_trial
from fredis.cli.train.commands.tune_fragment_tree import TuneFragmentTreeCommand


class TuningTests(unittest.TestCase):
    def spec(self, output):
        return {'outputDir': str(output),
                'baseArgs': ['--dataset-dir', 'prepared', '--lr', '0.001'],
                'parameters': {'--lr': [0.001, 0.002]}}

    def test_workbench_configuration_preserves_candidates_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'search'
            config = {'datasetDir': str(Path(tmp) / 'prepared'),
                      'outputDir': str(output), 'modelConfig': {},
                      'architectureMode': 'shared',
                      'sharedArchitecture': {'hidden_dim': 512},
                      'tuning': {'enabled': True, 'parameters': {
                          'sharedArchitecture.hidden_dim': [512, 256, 128]}}}
            plan = expand_search(self.spec(output))
            plan['configuration'] = config
            with patch('fredis.ml.tuning.fragment_tree.subprocess.call', return_value=0):
                self.assertEqual(run(plan), 0)
            plan_document = json.loads((output / 'fragment-tree.tuning-plan.pft').read_text())
            self.assertEqual(plan_document['schema'], 'fredis.tuning-plan')
            saved = plan_document['configuration']
            self.assertEqual(saved['tuning'], config['tuning'])
            self.assertEqual(saved['datasetDir'], config['datasetDir'])
            self.assertEqual(saved['outputDir'], str(output))
            self.assertEqual(saved['resume'], '')
            self.assertFalse((output / 'tuning-plan.json').exists())
            self.assertFalse((output / 'fragment-tree.tuning-config.pft').exists())
            self.assertEqual(config['outputDir'], str(output))

    def test_cli_and_subprocess_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.spec(root)
            plan_path = root / 'search.json'
            plan_path.write_text(json.dumps(spec))
            parser = argparse.ArgumentParser()
            command = TuneFragmentTreeCommand()
            command.configure(parser)
            calls = []

            def train(argv):
                self.assertEqual(argv[1:5], ['-m', 'fredis.cli', 'train', 'fragment-tree'])
                calls.append(argv)
                directory = Path(argv[argv.index('--output-dir') + 1])
                self.assertFalse(directory.exists())
                directory.mkdir()
                index = len(calls)
                (directory / 'metrics.tsv').write_text(
                    'event\tepoch\tglobal_step\tvalidation_loss\tvalidation_cosine\n'
                    f'train_step\t0\t0\t999\t999\n'
                    f'epoch_end\t1\t10\t{index}\t0.1\n'
                    f'epoch_end\t2\t20\t{3-index}\t{index / 10}\n')
                (directory / 'metric_distributions.tsv').write_text(
                    'global_step\tepoch\tsplit\tmetric\tvalue\n'
                    '10\t1\tvalidation\tcosine_similarity_q25\t0.8\n'
                    '20\t2\tvalidation\tcosine_similarity_q25\t0.2\n'
                    '20\t2\tvalidation\tcosine_similarity_q75\t0.9\n'
                    '20\t2\tvalidation\tcosine_similarity_median\t0.5\n'
                    '20\t2\tvalidation\tcosine_similarity_mean\t0.6\n'
                    '20\t2\tintermediate_validation\tcosine_similarity_q25\t0.99\n'
                    '30\t3\tvalidation\tcosine_similarity_q25\t1.0\n')
                return 0

            with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=train):
                with self.assertRaises(SystemExit) as result:
                    command.run(parser.parse_args(['--plan', str(plan_path)]))
                self.assertEqual(result.exception.code, 0)
            report = json.loads((root / RESULT_NAME).read_text())
            self.assertEqual(report['schema'], 'fredis.tuning-result')
            self.assertNotIn('evaluation', report)
            self.assertNotIn('bestTrial', report)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(len(report['trials'][0]['validationHistory']), 2)
            summaries = report['trials'][0]['scoreSummaries']
            self.assertEqual(summaries['validation_loss']['min'], 1)
            self.assertEqual(summaries['validation_loss']['max'], 2)
            self.assertEqual(summaries['validation_loss']['last'], 2)
            self.assertEqual(summaries['validation_cosine_similarity_q25'],
                             {'max': 0.8, 'maxEpoch': 1, 'last': 0.2, 'lastEpoch': 2})
            self.assertNotIn('min', summaries['validation_cosine_similarity_mean'])
            self.assertEqual(summaries['validation_cosine_similarity_q75']['last'], 0.9)
            with (root / 'tuning-scores.tsv').open() as stream:
                import csv
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['validation_cosine_similarity_q25.max'], '0.8')
            self.assertEqual(rows[0]['validation_cosine_similarity_q25.last'], '0.2')
            with patch('fredis.ml.tuning.fragment_tree.subprocess.call') as train_again:
                self.assertEqual(run(spec), 0)
                train_again.assert_not_called()

    def test_failures_and_missing_metrics_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('fredis.ml.tuning.fragment_tree.subprocess.call', side_effect=[1, 0]):
                self.assertEqual(run(self.spec(tmp)), 1)
            report = json.loads((Path(tmp) / RESULT_NAME).read_text())
            self.assertEqual(report['trials'][0]['status'], 'failed')
            self.assertEqual(report['trials'][0]['scoreSummaries'], {})
            self.assertEqual(len(report['trials']), 2)

    def test_last_does_not_reuse_an_earlier_missing_statistic(self):
        summary = summarize_trial({'validationHistory': [{'epoch': 1, 'validation_cosine_similarity_q25': 0.5}, {'epoch': 2}]})
        self.assertIsNone(summary['validation_cosine_similarity_q25']['last'])

    def test_invalid_plans_cannot_call_other_commands_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('command', 'output', 'resume'):
                plan = expand_search(self.spec(tmp))
                if kind == 'command':
                    plan['trials'][0]['args'] = ['-c', 'unexpected']
                elif kind == 'output':
                    plan['trials'][0]['config']['outputDir'] = '/tmp/outside'
                else:
                    plan['trials'][0]['args'].extend(['--resume', 'checkpoint'])
                with self.assertRaises(ValueError):
                    run(plan)
            grid = expand_search({**self.spec(tmp), 'parameters': {'--lr': list(range(129)), '--batch-size': [2, 4]}})
            self.assertEqual(len(grid['trials']), 258)
            self.assertEqual({(t['config']['parameters']['--lr'], t['config']['parameters']['--batch-size']) for t in grid['trials']}, {(lr, batch) for lr in range(129) for batch in [2, 4]})


if __name__ == '__main__':
    unittest.main()
