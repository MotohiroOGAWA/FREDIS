"""Check inherited inputs independently of model training and dataset tensor loading."""
import importlib.util
import json
import sys
from pathlib import Path
import tempfile

app_root = Path(__file__).resolve().parents[2]
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

source = app_root / 'fredis/ml/training/fragment_tree_training/sources.py'
spec = importlib.util.spec_from_file_location('training_sources', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, validation = root / 'train', root / 'validation'
    model = {'fragmenter_params': {'mass_tolerance': '0.01Da'},
             'adduct_type_strs': ['[M+H]+', '[M+Na]+'], 'symbols': ['C', 'O']}
    for split in (train, validation):
        (split / 'data').mkdir(parents=True)
        (split / 'reaction_statistics.json').write_text(json.dumps({'chemistry_config': model}))
    result = module.inherit_model_config({'fragmenter_params': {'ignored': True},
        'mol_encoder_params': {'node_dim': 999}, 'adduct_type_strs': ['ignored'],
        'reaction_model_params': {'hidden_dim': 128}}, train / 'data', validation, prepared_max_roles=None)
    assert result['fragmenter_params'] == model['fragmenter_params']
    assert result['adduct_type_strs'] == model['adduct_type_strs']
    assert result['mol_encoder_params'] == {'node_dim': 999, 'symbols': ['C', 'O']}
    assert result['reaction_model_params'] == {'hidden_dim': 128}
    assert module.inherit_model_config(result, train, validation, prepared_max_roles=None, saved_model=result) == result
    def rejects(operation):
        try:
            operation()
        except ValueError:
            return
        raise AssertionError('Incompatible source settings were accepted')
    fresh = module.inherit_model_config({'mol_encoder_params': {'node_dim': 48, 'graph_dim': 192}}, train, validation, prepared_max_roles=None)
    assert fresh['mol_encoder_params']['symbols'] == ['C', 'O']
    assert fresh['mol_encoder_params']['node_dim'] == 48
    rejects(lambda: module.inherit_model_config({}, train, validation, prepared_max_roles=None))
    model['fragmenter_params']['mass_tolerance'] = '0.1Da'
    (validation / 'reaction_statistics.json').write_text(json.dumps({'chemistry_config': model}))
    rejects(lambda: module.dataset_sources(train, validation))
    model['fragmenter_params']['mass_tolerance'] = '0.01Da'
    model['adduct_type_strs'].reverse()
    (validation / 'reaction_statistics.json').write_text(json.dumps({'chemistry_config': model}))
    rejects(lambda: module.dataset_sources(train, validation))
    model['adduct_type_strs'].reverse()
    (validation / 'reaction_statistics.json').write_text(json.dumps({'chemistry_config': model}))
    mismatched = {**result, 'mol_encoder_params': {**result['mol_encoder_params'], 'symbols': ['N', 'C']}}
    rejects(lambda: module.inherit_model_config(result, train, validation, prepared_max_roles=None, saved_model=mismatched))
    metadata = root / 'metadata'
    metadata.mkdir()
    (metadata / 'preparation_config.json').write_text(json.dumps({'chemistry_config': model}))
    (metadata / 'fragment-tree.pft').write_text(json.dumps({
        'fragmenterParams': model['fragmenter_params'], 'symbols': ['O', 'C']}))
    (metadata / 'reaction_statistics.json').write_text(json.dumps({'chemistry_config': model}))
    loaded = module.dataset_chemistry_config(metadata)
    assert loaded == {'fragmenter_params': model['fragmenter_params'], 'symbols': ['C', 'O'],
                      'adduct_type_strs': model['adduct_type_strs']}
    stale = json.loads(json.dumps(model))
    stale['fragmenter_params']['mass_tolerance'] = '9Da'
    (metadata / 'reaction_statistics.json').write_text(json.dumps({'chemistry_config': stale}))
    rejects(lambda: module.dataset_chemistry_config(metadata))
print('Dataset inheritance, resume compatibility and mismatch checks passed.')
