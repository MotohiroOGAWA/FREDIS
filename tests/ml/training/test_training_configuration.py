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
