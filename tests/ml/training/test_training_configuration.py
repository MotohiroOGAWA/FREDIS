import json
from fredis.ml.training.fragment_tree_training.configuration import CONFIG_NAME, write_training_configuration
from fredis.ml.training.fragment_tree_training.report import REPORT_NAME, write_training_report


def test_clickable_training_configuration_and_result(tmp_path):
    model = {'dropout': 0.5, 'mol_encoder_params': {'node_dim': 16}, 'reaction_model_params': {'pattern_model_params': {'0': {'hidden_dim': 64}}}, 'fine_tuning': {'frozen_pattern_ids': [2]}}
    settings = {'train_dir': str(tmp_path/'prepared'/'train_structures'), 'val_dir': str(tmp_path/'prepared'/'validation_structures'), 'epochs': 30, 'batch_size': 4, 'lr': 0.0001, 'validation_interval_steps': 1000, 'resume': None, 'cache_dataset_in_memory': True}
    output=tmp_path/'run'
    target=write_training_configuration(output,model,settings)
    assert target.name == CONFIG_NAME == 'fragment-tree.training-config.pft'
    doc=json.loads(target.read_text())
    assert doc['application'] == 'fragment-tree-training'
    assert doc['datasetDir'] == str(tmp_path/'prepared')
    assert doc['outputDir'] == str(output)
    assert doc['modelConfig'] == model
    assert doc['batchSize'] == 4 and doc['epochs'] == 30
    assert doc['dropout'] == 0.5 and doc['cacheDatasetInMemory']
    assert doc['validationIntervalSteps'] == 1000
    assert doc['patternArchitecture']['mode'] == 'individual'
    assert doc['patternArchitecture']['frozenPatternIds'] == [2]
    result=write_training_report(output,status='running')
    assert result.name == REPORT_NAME == 'result.training.pft'
    assert json.loads(result.read_text())['artifacts']['trainingConfiguration'] == CONFIG_NAME


def test_training_setup_writes_only_pft_configuration(tmp_path, monkeypatch):
    from copy import deepcopy
    from unittest.mock import MagicMock
    import pytest
    from fredis.ml.training.fragment_tree_training import training, sources
    from ..reaction.TestReactionPipeline import config

    class StopBeforeModelBuild(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopBeforeModelBuild()

    monkeypatch.setattr(training.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(training, 'progress_bar', lambda **kwargs: MagicMock())
    monkeypatch.setattr(training, '_load_prepared_datasets', lambda *args, **kwargs: ([[], []], {}))
    monkeypatch.setattr(sources, 'inherit_model_config', lambda value, *args, **kwargs: deepcopy(value))
    monkeypatch.setattr(training, 'create_spectrum_generator', stop)
    output=tmp_path/'run'
    with pytest.raises(StopBeforeModelBuild):
        training.train_reactions(model_config=config(), train_dir=tmp_path/'prepared'/'train_structures', val_dir=tmp_path/'prepared'/'validation_structures', output_dir=output)
    assert (output/CONFIG_NAME).is_file()
    assert not (output/'training_args.json').exists()
    assert not (output/'training_config.json').exists()
    document=json.loads((output/CONFIG_NAME).read_text())
    assert document['modelConfig']['reaction_model_params']['pattern_model_params']
    assert document['datasetDir']==str(tmp_path/'prepared')


def test_cli_does_not_write_separate_argument_json(tmp_path, monkeypatch):
    from fredis.ml.training.fragment_tree_training import training
    dataset=tmp_path/'prepared'
    (dataset/'train_structures').mkdir(parents=True)
    (dataset/'validation_structures').mkdir()
    (dataset/'fragment-tree.training-data-config.pft').write_text('{}')
    output=tmp_path/'run'
    calls=[]
    monkeypatch.setattr(training, 'train_reactions', lambda **kwargs: calls.append(kwargs))
    training.main(['--dataset-dir', str(dataset), '--output-dir', str(output), '--epochs', '3'])
    assert calls[0]['epochs']==3
    assert not (output/'training_args.json').exists()
    assert not (output/'training_config.json').exists()
