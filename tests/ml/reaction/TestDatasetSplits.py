"""Canonical molecule partitioning and CLI agreement."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fredis.data import MSDataset
from fredis.data.core.PeakSeries import PeakSeries
from fredis.libs.mmkit.mmkit import Compound
from fredis.ml.data_preparation.fragment_tree.datasets import (
    canonical_keys, canonical_smiles, dedupe_validation, split_assignments, split_dataset,
)
from fredis.ml.data_preparation.fragment_tree.create_training_data import main
from .TestReactionPipeline import chemistry


def dataset(smiles):
    mzs=[Compound.from_smiles(value).formula.exact_mass+1.007276466 for value in smiles]
    metadata=pd.DataFrame({'SMILES':smiles,'AdductType':['[M+H]+']*len(smiles),
                           'CollisionEnergy':[20]*len(smiles),'PrecursorMZ':mzs})
    return MSDataset(metadata,PeakSeries(np.array([[mz,1] for mz in mzs]),np.arange(len(smiles)+1,dtype=np.int64)))


class TestDatasetSplits(unittest.TestCase):
    def test_canonical_identity_and_three_disjoint_partitions(self):
        data=dataset(['CCO','OCC','CCN','NCC','CCC','COC','CCCO'])
        splits, assignments=split_dataset(data,'SMILES',0.2,0.2,7)
        groups={name:set(canonical_keys(part['SMILES'])) for name,part in splits.items()}
        self.assertEqual([len(groups[name]) for name in ('train','validation','test')],[3,1,1])
        self.assertEqual(sum(len(part) for part in splits.values()),len(data))
        for name,keys in groups.items():
            self.assertTrue(all(assignments[key]==name for key in keys))
            for other,other_keys in groups.items():
                if name!=other: self.assertFalse(keys & other_keys)
        _, reordered=split_assignments(list(reversed(data['SMILES'].tolist())),0.2,0.2,7)
        self.assertEqual(assignments,reordered)
        self.assertEqual(canonical_smiles('[CH3:4][CH2:5][OH:6]'),canonical_smiles('OCC'))
        self.assertNotEqual(canonical_smiles('F[C@H](Cl)Br'),canonical_smiles('F[C@@H](Cl)Br'))

    def test_disabled_splits_rounding_and_invalid_fractions(self):
        values=['C','CC','CCC']
        _, assignments=split_assignments(values,0,0)
        self.assertEqual(set(assignments.values()),{'train'})
        _, assignments=split_assignments(values,0.49,0.49)
        self.assertEqual(set(assignments.values()),{'train','validation','test'})
        for ratios in ((-0.1,0),(0,float('nan')),(float('inf'),0),(0.5,0.5),(0,1)):
            with self.subTest(ratios=ratios),self.assertRaises(ValueError):
                split_assignments(values,*ratios)
        with self.assertRaisesRegex(ValueError,'Not enough canonical molecules'):
            split_assignments(['CCO','OCC'],0.1,0.1)
        with self.assertRaisesRegex(ValueError,'Invalid SMILES'):
            canonical_smiles('invalid')

    def test_external_validation_overlap_uses_canonical_identity(self):
        validation, report=dedupe_validation(dataset(['CCO']),dataset(['OCC','CCC']),'SMILES','SMILES')
        self.assertEqual(validation['SMILES'].tolist(),['CCC'])
        self.assertEqual(report['removedOverlapRecords'],1)


    def test_test_only_split_and_saved_train_only_ratio(self):
        for test_ratio in (0, 0.25):
            with self.subTest(test_ratio=test_ratio), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'input.msds';output=root/'out'
                dataset(['CCO','CCN','CCC','COC']).save(str(source))
                config=chemistry()
                config['fragmenter_params']['fragment_ion_tree_builder']['max_reaction_count']=1
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    main(['--input',str(source),'--output-dir',str(output),'--params-json',json.dumps(config),
                          '--test-ratio',str(test_ratio)])
                self.assertFalse((output/'validation_structures').exists())
                self.assertEqual((output/'test_structures').exists(),bool(test_ratio))
                saved=json.loads((output/'train_structures/fragment-tree.training-data.pft').read_text())
                self.assertEqual(saved['validationRatio'],0)
                self.assertEqual(saved['testRatio'],test_ratio)
