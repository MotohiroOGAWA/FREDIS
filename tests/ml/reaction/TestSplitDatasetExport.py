"""Checked split export uses the checked records and CLI reports real progress."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fredis.ml.data_preparation.fragment_tree.split_datasets import check_split, export_checked, restore_checked, write_snapshot
from fredis.ml.data_preparation.fragment_tree.datasets import canonical_keys, load_spectrum_dataset
from .TestDatasetSplits import dataset
from .TestReactionPipeline import chemistry


class TestSplitDatasetExport(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=chemistry()

    def configuration(self,mode='automatic',test=0.25,validation=0.25):
        entries=[]
        for index,(smiles,role) in enumerate([(['CCO','OCC','CCN'],'train'),(['CCC'],'validation'),(['COC'],'test')]):
            file=self.root/f'input-{index}.msds';dataset(smiles).save(str(file))
            entries.append(dict(id=f'dataset-{index}',path=str(file),split=role))
        return dict(datasets=entries,datasetMode=mode,fragmenterParams=self.config['fragmenter_params'],
                    symbols=self.config['symbols'],validationRatio=validation,testRatio=test,validationSeed=7)

    def check(self,config,progress=None):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return check_split(config,progress)

    def capped_configuration(self):
        config = self.configuration('assigned', test=0, validation=0)
        for entry, suffix in zip(config['datasets'], ('', 'O', 'N')):
            dataset(['C' * n + suffix for n in range(1, 11)] + ['CC' + suffix]).save(entry['path'])
        config.update(validationMaxTrainRatio=0.3, testMaxTrainRatio=0.2)
        return config

    def test_assigned_caps_use_unique_train_molecules_and_keep_all_selected_spectra(self):
        config = self.capped_configuration()
        plan, snapshot = self.check(config)
        self.assertEqual([snapshot['splits'][name]['molecules']
                          for name in ('train', 'validation', 'test')], [10, 3, 2])
        self.assertEqual(snapshot['assignments'], self.check(config)[1]['assignments'])
        self.assertEqual(sum(split == 'excluded' for split in snapshot['assignments'].values()), 15)
        for name in ('train', 'validation', 'test'):
            original = [key for key in canonical_keys(plan['combined']['SMILES'])
                        if snapshot['assignments'][key] == name]
            self.assertEqual(canonical_keys(plan['splits'][name]['SMILES']), original)
        exported = export_checked(snapshot, self.root / 'capped-export')
        for name, path in exported['files'].items():
            self.assertEqual(len(load_spectrum_dataset(path)), snapshot['splits'][name]['records'])
        self.assertEqual(snapshot['validationMaxTrainRatio'], 0.3)
        self.assertEqual(snapshot['testMaxTrainRatio'], 0.2)

    def test_assigned_caps_blank_zero_and_round_down(self):
        config = self.capped_configuration()
        config.update(validationMaxTrainRatio='', testMaxTrainRatio=None)
        self.assertEqual(self.check(config)[1]['splits']['validation']['molecules'], 10)
        config.update(validationMaxTrainRatio=0, testMaxTrainRatio=0.09)
        plan, snapshot = self.check(config)
        self.assertEqual(len(plan['splits']['validation']), 0)
        self.assertEqual(len(plan['splits']['test']), 0)
        exported = export_checked(snapshot, self.root / 'zero-cap')
        self.assertEqual(list(exported['files']), ['train'])

    def test_assigned_caps_reject_invalid_values_before_loading(self):
        from fredis.ml.data_preparation.fragment_tree.dataset_plan import validate_list_config
        for value in (-1, float('nan'), float('inf'), True, 'bad'):
            for key in ('validation_max_train_ratio', 'test_max_train_ratio'):
                with self.subTest(value=value, key=key), self.assertRaisesRegex(ValueError, 'limits'):
                    validate_list_config([dict(path='missing.msds', split='train')], 'assigned', **{key: value})

    def test_assigned_caps_remove_train_overlap_before_sampling_and_export(self):
        config = self.capped_configuration()
        train = ['C' * n for n in range(1, 11)]
        validation = train + ['C' * n + 'O' for n in range(1, 11)]
        dataset(validation).save(config['datasets'][1]['path'])
        # Train may appear after validation in the input list.
        config['datasets'] = list(reversed(config['datasets']))
        plan, snapshot = self.check(config)
        validation_entry = next(e for e in config['datasets'] if e['split'] == 'validation')
        self.assertEqual(snapshot['validIndexes'][validation_entry['id']], list(range(10, 20)))
        self.assertEqual(snapshot['splits']['validation']['molecules'], 3)
        exported = export_checked(snapshot, self.root / 'no-train-overlap')
        self.assertEqual(len(load_spectrum_dataset(exported['files']['train'])), 11)
        train_keys = set(canonical_keys(plan['splits']['train']['SMILES']))
        for name in ('validation', 'test'):
            self.assertFalse(train_keys & set(canonical_keys(plan['splits'][name]['SMILES'])))
        # Disabling the limit preserves assigned mode's existing overlap error.
        config['validationMaxTrainRatio'] = None
        with self.assertRaisesRegex(ValueError, 'appears in both'):
            self.check(config)

    def test_assigned_cap_all_candidates_overlap_train(self):
        config = self.capped_configuration()
        dataset(['C', 'CC']).save(config['datasets'][1]['path'])
        plan, snapshot = self.check(config)
        self.assertEqual(len(plan['splits']['validation']), 0)
        exported = export_checked(snapshot, self.root / 'all-overlap')
        self.assertNotIn('validation', exported['files'])

    def test_filtered_input_export_preserves_original_schema_metadata_and_peaks(self):
        import numpy as np
        import pandas as pd
        from fredis.msdata import with_metadata

        config = self.configuration('assigned', test=0, validation=0)
        config['datasets'] = config['datasets'][:2]
        config['validationMaxTrainRatio'] = 0.5
        originals = []
        for entry, smiles in zip(config['datasets'], (['CCO', 'CCN', 'CCO'], ['CCO', 'CCC', 'COC', 'COC'])):
            data = dataset(smiles)
            metadata = data.to_pandas().copy()
            metadata['OriginalStructure'] = metadata['SMILES']
            metadata['SMILES'] = ['original metadata, not the mapped structure'] * len(metadata)
            metadata['LabNote'] = [''] + [f'note-{i}' for i in range(1, len(metadata))]
            metadata['OriginalRow'] = np.arange(len(metadata))
            with_metadata(metadata, data, columns=list(metadata.columns), description='original description',
                          attributes={'source_lab': 'test'}, tags=['original']).save(entry['path'])
            entry['mapping'] = {'smilesColumn': 'OriginalStructure'}
            originals.append(load_spectrum_dataset(entry['path']))
        _, snapshot = self.check(config)
        before = dict(snapshot['fingerprints'])
        result = export_checked(snapshot, self.root / 'original-inputs', preserve_input_metadata=True)
        self.assertEqual(len(result['files']), 2)
        for index, (entry, original) in enumerate(zip(config['datasets'], originals)):
            name = entry['split']
            filename = f"{name}/{index + 1:03d}_{Path(entry['path']).stem}.msds"
            exported = load_spectrum_dataset(result['files'][filename])
            rows = [i for i in snapshot['validIndexes'][entry['id']]
                    if snapshot['assignments'][canonical_keys([original['OriginalStructure'][i]])[0]] == name]
            expected = original[rows]
            pd.testing.assert_frame_equal(exported.to_pandas(), expected.to_pandas())
            np.testing.assert_array_equal(exported.mz, expected.mz)
            np.testing.assert_array_equal(exported.intensity, expected.intensity)
            np.testing.assert_array_equal(exported.offsets, expected.offsets)
            self.assertEqual(exported.description, original.description)
            self.assertEqual(exported.attributes, original.attributes)
            self.assertEqual(exported.tags, original.tags)
            self.assertNotIn('_fredis_dataset_path', exported.to_pandas().columns)
        from fredis.ml.data_preparation.fragment_tree.split_datasets import fingerprint
        self.assertEqual(before, {entry['id']: fingerprint(entry['path']) for entry in config['datasets']})
        checked = self.root / 'original-checked.json'
        write_snapshot(snapshot, checked)
        cli = subprocess.run([sys.executable, '-m', 'fredis.cli', 'train', 'split-datasets',
                              '--checked-plan', str(checked), '--preserve-input-metadata',
                              '--output-dir', str(self.root / 'original-cli')], capture_output=True, text=True)
        self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
        self.assertEqual(set(json.loads(cli.stdout)['result']['files']), set(result['files']))
        with self.assertRaises(FileExistsError):
            export_checked(snapshot, self.root / 'original-inputs', preserve_input_metadata=True)

    def test_assigned_caps_real_preparation_and_cli_export(self):
        config = self.configuration('assigned', test=0, validation=0)
        for entry, smiles in zip(config['datasets'], (
                ['CCO', 'CCN', 'CCC', 'COC'], ['CCO', 'CCCCO', 'CCCCN', 'CCCC'], ['CCN', 'CCCOC', 'CCNCC', 'CCCCC'])):
            dataset(smiles).save(entry['path'])
        output = self.root / 'capped-prepared'
        from fredis.ml.data_preparation.fragment_tree.create_training_data import main
        args = ['--datasets-json', json.dumps(config['datasets']), '--dataset-mode', 'assigned',
                '--params-json', json.dumps(self.config), '--validation-max-train-ratio', '0.25',
                '--test-max-train-ratio', '0.5']
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(args + ['--output-dir', str(output), '--validation-seed', '7', '--num-workers', '1'])
        manifest = json.loads((output / 'split_manifest.json').read_text())
        self.assertEqual([list(manifest['assignments'].values()).count(name)
                          for name in ('train', 'validation', 'test')], [4, 1, 2])
        restored = json.loads((output / 'fragment-tree.training-data-config.pft').read_text())['workbench_config']
        self.assertEqual(restored['validationMaxTrainRatio'], 0.25)
        self.assertEqual(restored['testMaxTrainRatio'], 0.5)
        result = subprocess.run([sys.executable, '-m', 'fredis.cli', 'train', 'split-datasets',
                                 *args, '--seed', '7', '--output-dir', str(self.root / 'capped-cli')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        exported_manifest = json.loads((self.root / 'capped-cli/dataset_split_manifest.json').read_text())
        self.assertEqual(exported_manifest['assignments'], manifest['assignments'])

    def per_dataset_configuration(self):
        config = self.configuration('per-dataset', test=0, validation=0.1)
        config['datasets'] = config['datasets'][:2]
        for entry, suffix in zip(config['datasets'], ('', 'O')):
            dataset(['C' * n + suffix for n in range(1, 11)]).save(entry['path'])
        return config

    def test_per_dataset_ratios_and_checked_export(self):
        config = self.per_dataset_configuration()
        plan, snapshot = self.check(config)
        self.assertEqual(snapshot['mode'], 'per-dataset')
        for source in plan['sources']:
            self.assertEqual(source['splits']['train'], dict(molecules=9, records=9))
            self.assertEqual(source['splits']['validation'], dict(molecules=1, records=1))
        self.assertEqual(snapshot['assignments'], self.check(config)[1]['assignments'])
        exported = export_checked(snapshot, self.root / 'per-dataset-export')
        self.assertEqual(len(load_spectrum_dataset(exported['files']['train'])), 18)
        self.assertEqual(len(load_spectrum_dataset(exported['files']['validation'])), 2)
        from fredis.ml.data_preparation.fragment_tree.create_training_data import build_arg_parser as preparation_parser
        from fredis.ml.data_preparation.fragment_tree.split_datasets import build_arg_parser as split_parser
        for parser in (preparation_parser(), split_parser()):
            args = parser.parse_args(['--datasets-json', json.dumps(config['datasets']),
                                     '--dataset-mode', 'per-dataset', '--output-dir', str(self.root / 'out')])
            self.assertEqual(args.dataset_mode, 'per-dataset')

    def test_per_dataset_shared_molecules_remain_in_one_split(self):
        config = self.per_dataset_configuration()
        values = ['C' * n for n in range(1, 11)]
        dataset(values + ['CC']).save(config['datasets'][0]['path'])
        dataset(list(reversed(values))).save(config['datasets'][1]['path'])
        plan, snapshot = self.check(config)
        self.assertEqual(len(snapshot['assignments']), 10)
        for source in plan['sources']:
            self.assertEqual(source['splits']['train']['molecules'], 9)
            self.assertEqual(source['splits']['validation']['molecules'], 1)
        train = set(canonical_keys(plan['splits']['train']['SMILES']))
        validation = set(canonical_keys(plan['splits']['validation']['SMILES']))
        self.assertFalse(train & validation)
        self.assertEqual(sum(len(data) for data in plan['splits'].values()), 21)

    def test_per_dataset_overlap_prioritizes_no_leakage_over_exact_ratio(self):
        config = self.per_dataset_configuration()
        first = self.check(config)[1]
        shared_train = [key for key, split in first['assignments'].items()
                        if split == 'train' and 'O' not in key][:2]
        dataset(shared_train).save(config['datasets'][1]['path'])
        plan, _ = self.check(config)
        self.assertEqual(plan['sources'][1]['splits']['validation']['molecules'], 0)
        self.assertEqual(plan['sources'][1]['splits']['train']['molecules'], 2)

    def test_per_dataset_real_preparation_and_cli_export(self):
        config = self.per_dataset_configuration()
        config['validationRatio'], config['testRatio'] = 0.25, 0.25
        for entry, smiles in zip(config['datasets'], (
                ['CCO', 'CCN', 'CCC', 'COC'], ['CCCC', 'CCCCO', 'CCCCN', 'CCCOC'])):
            dataset(smiles).save(entry['path'])
        plan, snapshot = self.check(config)
        for source in plan['sources']:
            self.assertEqual([source['splits'][name]['molecules']
                              for name in ('train', 'validation', 'test')], [2, 1, 1])
        output = self.root / 'prepared-per-dataset'
        from fredis.ml.data_preparation.fragment_tree.create_training_data import main
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(['--datasets-json', json.dumps(config['datasets']), '--dataset-mode', 'per-dataset',
                  '--params-json', json.dumps(self.config), '--output-dir', str(output),
                  '--validation-ratio', '0.25', '--test-ratio', '0.25', '--validation-seed', '7',
                  '--num-workers', '1'])
        for split in ('train', 'validation', 'test'):
            self.assertTrue(list((output / f'{split}_structures').rglob('*.preft.pt')), split)
        command = [sys.executable, '-m', 'fredis.cli', 'train', 'split-datasets',
                   '--datasets-json', json.dumps(config['datasets']), '--dataset-mode', 'per-dataset',
                   '--params-json', json.dumps(self.config), '--validation-ratio', '0.25',
                   '--test-ratio', '0.25', '--seed', '7', '--output-dir', str(self.root / 'cli-per-dataset')]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)['result']['splits'], snapshot['splits'])

    def test_progress_and_exported_counts_match_checked_summary_in_both_modes(self):
        for mode in ('automatic','assigned'):
            with self.subTest(mode=mode):
                config=self.configuration(mode);events=[]
                plan,snapshot=self.check(config,events.append)
                self.assertIn('validating',{event['stage'] for event in events})
                self.assertTrue(any(event.get('current',0)>0 for event in events if event['stage']=='validating'))
                self.assertEqual(events[-1]['percent'],100)
                self.assertEqual(sorted(event['percent'] for event in events),[event['percent'] for event in events])
                output=self.root/mode
                exported=export_checked(snapshot,output,events.append)
                self.assertEqual(set(exported['files']),{'train','validation','test'})
                groups={}
                for name,file in exported['files'].items():
                    data=load_spectrum_dataset(file)
                    groups[name]=set(canonical_keys(data['SMILES']))
                    self.assertEqual(len(data),snapshot['splits'][name]['records'])
                    self.assertEqual(len(groups[name]),snapshot['splits'][name]['molecules'])
                self.assertFalse(groups['train']&groups['validation'])
                self.assertFalse(groups['train']&groups['test'])
                self.assertEqual(len(list(output.glob('*.msds'))),3)
                self.assertTrue((output/'dataset_split_manifest.json').exists())

    def test_zero_record_splits_are_not_written_and_original_data_is_preserved(self):
        plan,snapshot=self.check(self.configuration(test=0,validation=0))
        result=export_checked(snapshot,self.root/'out')
        self.assertEqual(list(result['files']),['train'])
        self.assertFalse((self.root/'out/validation.msds').exists())
        self.assertFalse((self.root/'out/test.msds').exists())
        exported=load_spectrum_dataset(result['files']['train'])
        self.assertEqual(list(exported['SMILES']),list(plan['combined']['SMILES']))
        self.assertEqual(exported.mz.tolist(),plan['combined'].mz.tolist())

    def test_changed_source_requires_rechecking_without_publishing_outputs(self):
        config=self.configuration();_,snapshot=self.check(config)
        dataset(['CCCO']).save(config['datasets'][0]['path'])
        with self.assertRaisesRegex(ValueError,'changed since checking'):
            export_checked(snapshot,self.root/'out')
        self.assertFalse((self.root/'out/train.msds').exists())

    def test_existing_output_and_stale_zero_split_are_never_overwritten(self):
        _,snapshot=self.check(self.configuration(test=0,validation=0));output=self.root/'out';output.mkdir()
        existing=output/'test.msds';existing.write_text('previous export')
        with self.assertRaises(FileExistsError):
            export_checked(snapshot,output)
        self.assertEqual(existing.read_text(),'previous export')
        self.assertFalse((output/'train.msds').exists())

    def test_failed_save_does_not_publish_partial_datasets(self):
        _,snapshot=self.check(self.configuration());output=self.root/'out'
        with patch('msentity.Dataset.save',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):
                export_checked(snapshot,output)
        self.assertEqual(list(output.iterdir()),[])

    def test_restore_does_not_repeat_metadata_validation(self):
        _,snapshot=self.check(self.configuration())
        with patch('fredis.ml.data_preparation.fragment_tree.dataset_plan.inspect_records',side_effect=AssertionError('must reuse checked rows')):
            splits=restore_checked(snapshot)
        self.assertEqual(sum(len(data) for data in splits.values()),5)

    def test_real_cli_check_then_export_and_direct_list_export(self):
        config=self.configuration();checked=self.root/'checked.json';events=[]
        command=[sys.executable,'-m','fredis.cli','train','split-datasets']
        check=subprocess.run(command+['--config-stdin','--plan-output',str(checked)],input=json.dumps(config),text=True,capture_output=True)
        self.assertEqual(check.returncode,0,check.stdout+check.stderr)
        result=json.loads(check.stdout.strip().splitlines()[-1]);self.assertTrue(result['ok'])
        self.assertEqual(result['result']['checkedPlan'],str(checked))
        for line in check.stderr.splitlines():
            if line.startswith('FREDIS_PROGRESS '):events.append(json.loads(line[len('FREDIS_PROGRESS '):]))
        self.assertGreater(len(events),5);self.assertEqual(events[-1]['percent'],100)
        export=subprocess.run(command+['--checked-plan',str(checked),'--output-dir',str(self.root/'export')],capture_output=True,text=True)
        self.assertEqual(export.returncode,0,export.stdout+export.stderr)
        self.assertEqual(json.loads(export.stdout)['result']['splits'],result['result']['splits'])
        direct=subprocess.run(command+['--datasets-json',json.dumps(config['datasets']),'--params-json',json.dumps(self.config),
            '--validation-ratio','0','--test-ratio','0','--output-dir',str(self.root/'direct')],capture_output=True,text=True)
        self.assertEqual(direct.returncode,0,direct.stdout+direct.stderr)
        self.assertEqual(list(json.loads(direct.stdout)['result']['files']),['train'])
