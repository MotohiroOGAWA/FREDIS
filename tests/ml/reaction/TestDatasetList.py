"""Multiple input datasets: assigned roles and canonical union splitting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from fredis.ml.data_preparation.fragment_tree.dataset_plan import plan_datasets, validate_list_config
from fredis.ml.data_preparation.fragment_tree.datasets import canonical_keys
from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
from fredis.ml.data_preparation.fragment_tree.create_training_data import main
from .TestDatasetSplits import dataset
from .TestReactionPipeline import chemistry


class TestDatasetList(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=chemistry()
        self.config['fragmenter_params']['fragment_ion_tree_builder']['max_reaction_count']=1
        self.context=create_preparation_context(self.config)

    def source(self,name,smiles,split='train',mapping=None):
        file=self.root/(name+'.msds');data=dataset(smiles)
        if mapping:
            for key,column in mapping.items():
                original={'smilesColumn':'SMILES','adductTypeColumn':'AdductType','collisionEnergyColumn':'CollisionEnergy','precursorMzColumn':'PrecursorMZ'}[key]
                data[column]=data[original]
        data.save(str(file))
        return dict(id='dataset-'+name,path=str(file),split=split,mapping=mapping or {})

    def plan(self,entries,mode,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return plan_datasets(entries,mode,self.context.fragmenter,self.config['symbols'],**kwargs)

    def test_automatic_split_combines_files_before_assigning_canonical_molecules(self):
        entries=[self.source('a',['CCO','CCN']),self.source('b',['OCC','CCC','COC'],'test')]
        plan=self.plan(entries,'automatic',validation_ratio=0.25,test_ratio=0.25,seed=2)
        self.assertEqual(len(plan['assignments']),4)
        self.assertEqual(sum(item['records'] for item in plan['summary'].values()),5)
        keys=canonical_keys(plan['combined']['SMILES'])
        self.assertEqual(keys.count('CCO'),2)
        self.assertEqual(plan['combined'].peaks.lengths.tolist(),[1]*5)
        memberships={name:set(canonical_keys(data['SMILES'])) for name,data in plan['splits'].items()}
        self.assertEqual(sum('CCO' in group for group in memberships.values()),1)
        reordered=self.plan(list(reversed(entries)),'automatic',validation_ratio=0.25,test_ratio=0.25,seed=2)
        self.assertEqual(plan['assignments'],reordered['assignments'])

    def test_assigned_mode_combines_same_role_and_retains_all_validation_records(self):
        entries=[self.source('a',['CCO']),self.source('b',['OCC','CCN']),
                 self.source('v',['CCC','COC'],'validation'),self.source('t',['CCCO'],'test')]
        plan=self.plan(entries,'assigned',validation_ratio=0.9,test_ratio=0.9)
        self.assertEqual(plan['summary'],{'train':{'molecules':2,'records':3},'validation':{'molecules':2,'records':2},'test':{'molecules':1,'records':1}})
        self.assertEqual(set(plan['assignments'].values()),{'train','validation','test'})

    def test_assigned_mode_rejects_canonical_overlap_across_roles(self):
        for role in ('validation','test'):
            with self.subTest(role=role),self.assertRaisesRegex(ValueError,'Canonical molecule CCO'):
                self.plan([self.source('a',['CCO']),self.source('b',['OCC'],role)],'assigned')

    def test_independent_column_mappings_and_source_provenance(self):
        entries=[self.source('a',['CCO']),self.source('b',['CCN'],mapping={'smilesColumn':'Structure','adductTypeColumn':'Adduct'})]
        plan=self.plan(entries,'automatic')
        self.assertEqual(plan['splits']['train']['SMILES'].tolist(),['CCO','CCN'])
        self.assertEqual(plan['combined']['_fredis_source_record'].tolist(),[0,0])
        self.assertEqual(plan['combined']['_fredis_dataset_path'].tolist(),[entry['path'] for entry in entries])

    def test_invalid_records_filtered_before_union_and_kept_peaks_match(self):
        source=self.source('a',['CCO','CCC','CCN'])
        from fredis.ml.data_preparation.fragment_tree.datasets import load_spectrum_dataset
        data=load_spectrum_dataset(source['path']);data['SMILES']=['CCO','invalid','CCN'];data.save(source['path'])
        plan=self.plan([source,self.source('b',['COC'])],'automatic')
        self.assertEqual(plan['sources'][0]['excludedRecords'],1)
        self.assertEqual(plan['combined']['_fredis_source_record'].tolist(),[0,2,0])
        self.assertEqual(len(plan['combined']),3)
        self.assertEqual(len(plan['combined'].peaks.mz),3)
        self.assertEqual(plan['combined'].peaks.mz[:2].tolist(),data[[0,2]].peaks.mz.tolist())


    def test_assigned_train_only_has_no_other_outputs(self):
        plan=self.plan([self.source('a',['CCO'])],'assigned')
        self.assertEqual(len(plan['splits']['validation']),0)
        self.assertEqual(len(plan['splits']['test']),0)

    def test_invalid_lists_and_missing_train_are_rejected_before_loading(self):
        for entries,mode in (([],'automatic'),([{}],'automatic'),([{'path':'x','split':'test'}],'assigned'),([{'path':'x','split':'wrong'}],'assigned'),([{'path':'x'}],'wrong')):
            with self.subTest(entries=entries,mode=mode),self.assertRaises(ValueError):
                validate_list_config(entries,mode)
