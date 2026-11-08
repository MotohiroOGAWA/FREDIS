from copy import deepcopy
import csv
import json
from unittest.mock import patch

import pytest
from rdkit import Chem

from fredis.domain.fragment.reaction import ReactionPatternSet
from fredis.ml.data_preparation.fragment_tree.pattern_filter import select_trainable_pattern_records
from fredis.ml.data_preparation.fragment_tree.create_training_data import create_reaction_training_data
from .TestReactionPipeline import chemistry


def definitions():
    return ReactionPatternSet.from_dict({'name': 'test', 'patterns': [
        {'name': 'base', 'reactant_smarts': '[C:1]-[C:2]',
         'products': [{'name': 'cut', 'smarts': '[C:1].[C:2]'}]},
        {'name': 'new oxygen', 'reactant_smarts': '[C:1]-[O:2]',
         'products': [{'name': 'cut', 'smarts': '[C:1].[O:2]'}]},
    ]})


def ownership(patterns):
    definition = deepcopy(patterns.patterns[0].to_dict())
    definition['pattern_id'] = 97  # ID in imported model differs from dataset.
    return [{'pattern': definition, 'source': 'pretrained.pt', 'source_pattern_id': 97}]


def test_selection_ignores_fixed_matches_and_preserves_order_and_repeated_conditions():
    patterns = definitions()
    with patch('fredis.ml.data_preparation.fragment_tree.pattern_filter.Chem.MolFromSmiles',
               wraps=Chem.MolFromSmiles) as parse:
        rows, report = select_trainable_pattern_records(['CCC', 'CCO', 'CCN', 'CCO', 'N'], patterns, ownership(patterns))
        assert parse.call_count == 4
    assert rows == [1, 3]
    assert report['excluded_records'] == 3
    assert report['retained_compounds'] == 1
    assert report['trainable_pattern_ids'] == [patterns.patterns[1].pattern_id]
    rows, _ = select_trainable_pattern_records(['CCC', 'N'], patterns)
    assert rows == [0]


def test_all_imported_and_zero_matches():
    patterns = definitions()
    with pytest.raises(ValueError, match='No trainable reaction patterns'):
        select_trainable_pattern_records(['CCO'], patterns,
            [{'pattern': p.to_dict(), 'source': 'model.pt'} for p in patterns])
    rows, report = select_trainable_pattern_records(['CCC'], patterns, ownership(patterns))
    assert rows == [] and report['retained_records'] == 0


@pytest.mark.parametrize('workers', [1, 2])
def test_preparation_filters_before_building_and_keeps_original_record_indices(tmp_path, workers):
    patterns = definitions()
    config = chemistry()
    config['fragmenter_params']['fragment_ion_tree_builder']['reaction_pattern_set'] = patterns.to_dict()
    config['reaction_models'] = ownership(patterns)
    from .TestPreparationWorkflow import TestPreparationWorkflow
    dataset = TestPreparationWorkflow().dataset()  # CCO, CCO, CCN, CCC, COC
    files = create_reaction_training_data(dataset=dataset, chemistry_config=config,
        output_dir=tmp_path, preparation_config={}, num_workers=workers,
        source_record_indexes=[10, 11, 12, 13, 14])
    assert files
    report = json.loads((tmp_path / 'pattern_filter_summary.json').read_text())
    assert report['input_records'] == 5
    assert report['retained_records'] == 3
    assert report['excluded_records'] == 2
    manifest = list(csv.DictReader((tmp_path / 'manifest.tsv').open(), delimiter='\t'))
    assert {row['smiles'] for row in manifest} == {'CCO', 'COC'}
    scores = list(csv.DictReader((tmp_path / 'assignment_scores.tsv').open(), delimiter='\t'))
    assert {int(row['record_index']) for row in scores} == {10, 11, 14}
    assert json.loads((tmp_path / 'reaction_statistics.json').read_text())['pattern_selection'] == report


def test_validation_cap_is_computed_after_trainable_pattern_selection(tmp_path):
    from .TestDatasetSplits import dataset
    from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
    from fredis.ml.data_preparation.fragment_tree.dataset_plan import plan_datasets
    patterns = definitions()
    config = chemistry()
    config['fragmenter_params']['fragment_ion_tree_builder']['reaction_pattern_set'] = patterns.to_dict()
    fragmenter = create_preparation_context(config).fragmenter
    entries = []
    for split, smiles in [('train', ['CCO', 'COC', 'CCC', 'CCN']),
                           ('validation', ['CCCO', 'CC(C)O', 'OCCO', 'CCCC'])]:
        file = tmp_path / f'{split}.msds'
        dataset(smiles).save(str(file))
        entries.append(dict(id=f'dataset-{split}', path=str(file), split=split, mapping={}))
    plan = plan_datasets(entries, 'assigned', fragmenter, config['symbols'],
                         validation_max_train_ratio=0.5, reaction_models=ownership(patterns))
    assert plan['summary']['train']['molecules'] == 2
    assert plan['summary']['validation']['molecules'] == 1
    assert all('O' in smiles for split in ('train', 'validation') for smiles in plan['splits'][split]['SMILES'])
    assert plan['sources'][0]['pattern_selection']['excluded_records'] == 2
    assert plan['valid_indexes']['dataset-train'] == [0, 1]


def test_all_pattern_selection_retains_frozen_only_matches():
    patterns = definitions()
    rows, report = select_trainable_pattern_records(
        ['CCC', 'CCO', 'N'], patterns, ownership(patterns), mode='all')
    assert rows == [0, 1]
    assert report['mode'] == 'all'
    assert report['excluded_records'] == 1
    rows, _ = select_trainable_pattern_records(['CCC'], patterns,
        [{'pattern': p.to_dict(), 'source': 'pretrained.pt'} for p in patterns], mode='all')
    assert rows == [0]
    with pytest.raises(ValueError, match='Pattern selection'):
        select_trainable_pattern_records([], patterns, mode='invalid')
