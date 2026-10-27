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
        self.assertEqual(exported['SMILES'].tolist(),plan['combined']['SMILES'].tolist())
        self.assertEqual(exported.peaks.mz.tolist(),plan['combined'].peaks.mz.tolist())

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
        with patch('fredis.data.MSDataset.save',side_effect=OSError('disk full')):
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
