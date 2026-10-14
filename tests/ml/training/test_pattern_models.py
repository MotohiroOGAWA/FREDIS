from copy import deepcopy
import pytest
import torch
from fredis.domain.fragment import Fragmenter
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel
from fredis.ml.training.fragment_tree_training.pattern_models import (
    assemble_config, export_pattern, inspect_model, initialize_imports, pattern_config)
from .test_pattern_integration import checkpoints


def test_export_import_complete_pattern_and_freeze(checkpoints):
    source, _, output = checkpoints
    before = torch.load(source, weights_only=False)
    pid = inspect_model(source)['patterns'][0]['pattern']['pattern_id']
    asset = output.with_name('single.reaction-model.pt')
    exported = export_pattern(source, pid, asset)
    assert len(exported['patterns']) == 1
    params = assemble_config(before['model_config'], [dict(path=str(asset))])
    generator = create_spectrum_generator(params)
    model = ReactionFragmentTreeTrainingModel(generator.feature_model, spectrum_model=generator.spectrum_model)
    initialize_imports(model, params)
    for name, tensor in torch.load(asset, weights_only=False)['model_state_dict'].items():
        assert torch.equal(tensor, model.state_dict()[name])
    reaction = generator.feature_model.reaction_instance_encoder
    generator.train()
    assert not reaction.reactant_encoders[str(pid)].training
    assert all(not p.requires_grad for p in reaction.pattern_heads[str(pid)].parameters())
    assert any(p.requires_grad for p in generator.spectrum_model.parameters())
    for other, encoder in reaction.reactant_encoders.items():
        if other != str(pid):
            assert any(p.requires_grad for p in encoder.parameters())
    with pytest.raises(ValueError, match='Duplicate pretrained'):
        assemble_config(before['model_config'], [dict(path=str(asset)),dict(path=str(asset))])


def test_individual_sizes_and_chemical_inputs(checkpoints):
    source, _, _ = checkpoints
    params = deepcopy(torch.load(source, weights_only=False)['model_config'])
    patterns = Fragmenter.from_dict(params['fragmenter_params']).reaction_pattern_set.patterns
    pid = str(patterns[0].pattern_id)
    params.setdefault('reaction_model_params', {})['pattern_model_params'] = {
        pid: dict(node_dim=12, graph_dim=19, num_layers=3, num_heads=3, dropout=0, descriptors=['MolLogP','TPSA'])}
    generator = create_spectrum_generator(params)
    reaction = generator.feature_model.reaction_instance_encoder
    assert reaction.reactant_encoders[pid].graph_dim == 19
    assert reaction.pattern_heads[pid].descriptor_indices
    for other, config in reaction.pattern_model_params.items():
        if other != pid:
            assert config['descriptors'] == []
            assert config['num_layers'] == 2


def test_conflicting_shared_inputs_rejected(checkpoints):
    source, other, _ = checkpoints
    saved = torch.load(other, weights_only=False)
    key = 'feature_model.reaction_instance_encoder.atom_projection.weight'
    saved['model_state_dict'][key] += 1
    torch.save(saved, other)
    with pytest.raises(ValueError, match='conflicting shared'):
        assemble_config(saved['model_config'], [dict(path=str(source)),dict(path=str(other))])


def test_preparation_owns_definitions_and_pretrained_selections(checkpoints, tmp_path):
    import json
    from fredis.ml.training.fragment_tree_training.pattern_models import prepare_pattern_models
    from fredis.ml.training.fragment_tree_training.sources import inherit_model_config
    source, _, _ = checkpoints
    saved=torch.load(source, weights_only=False)
    params=saved['model_config']
    item=inspect_model(source)['patterns'][0]
    chemistry=prepare_pattern_models({'fragmenter_params':params['fragmenter_params'],
        'symbols':params['mol_encoder_params']['symbols']},[dict(pattern=item['pattern'],source=str(source),source_pattern_id=item['pattern']['pattern_id'],architecture={'hidden_dim':999})])
    assert all('architecture' not in record for record in chemistry['reaction_models'])
    directories=[]
    for name in ('train_structures','validation_structures'):
        directory=tmp_path/name;directory.mkdir();directories.append(directory)
        (directory/'preparation_config.json').write_text(json.dumps({'chemistry_config':chemistry}))
    inherited=inherit_model_config(params,*directories,prepared_max_roles=None)
    assert inherited['fine_tuning']['frozen_pattern_ids']==[item['pattern']['pattern_id']]
    assert inherited['reaction_model_params']['pattern_model_params'][str(item['pattern']['pattern_id'])]==item['architecture']
    # Resume consumes its saved architecture without reopening original imports.
    source.unlink()
    resumed=inherit_model_config(inherited,*directories,saved_model=inherited,prepared_max_roles=None)
    assert resumed==inherited


def test_raw_pattern_position_capacity_is_independent_of_legacy_roles(checkpoints):
    from fredis.ml.training.fragment_tree_training.pattern_models import prepare_pattern_models, apply_dataset_pattern_models
    source,_,_=checkpoints
    saved=torch.load(source,weights_only=False);params=saved['model_config'];first=inspect_model(source)['patterns'][0]
    large=dict(name='large',reactant_smarts='-'.join(f'[#6:{i}]' for i in range(1,71)),products=[dict(name='left',smarts='[*:1]')])
    chemistry=prepare_pattern_models(dict(fragmenter_params=params['fragmenter_params'],symbols=params['mol_encoder_params']['symbols']),[
        dict(pattern=first['pattern'],source=str(source),source_pattern_id=first['pattern']['pattern_id']),dict(pattern=large)])
    config={**params,'fragmenter_params':chemistry['fragmenter_params'],'reaction_model_params':{**params.get('reaction_model_params',{}),'pattern_input_version':2}}
    config=apply_dataset_pattern_models(config,chemistry['reaction_models'])
    generator=create_spectrum_generator(config)
    model=ReactionFragmentTreeTrainingModel(generator.feature_model,spectrum_model=generator.spectrum_model)
    initialize_imports(model,config)
    reaction=generator.feature_model.reaction_instance_encoder
    assert reaction.roles.num_embeddings==64
    assert any(encoder.position_count==70 and encoder.position_encoding=='one_hot' for encoder in reaction.reactant_encoders.values())


def test_global_memory_is_exported_imported_and_frozen(checkpoints):
    source, _, output = checkpoints
    saved=torch.load(source,weights_only=False)
    params=deepcopy(saved['model_config'])
    pid=str(Fragmenter.from_dict(params['fragmenter_params']).reaction_pattern_set.patterns[0].pattern_id)
    reaction_params=params.setdefault('reaction_model_params',{})
    reaction_params['pattern_input_version']=2
    reaction_params['pattern_model_params']={pid:dict(global_num_layers=3,global_hidden_dim=8,global_num_heads=2)}
    generator=create_spectrum_generator(params)
    model=ReactionFragmentTreeTrainingModel(generator.feature_model,spectrum_model=generator.spectrum_model)
    saved.update(model_config=params,model_state_dict=model.state_dict());torch.save(saved,source)
    asset=output.with_name('global.reaction-model.pt')
    exported=export_pattern(source,int(pid),asset)
    assert exported['patterns'][0]['architecture']['global_num_layers']==3
    config=assemble_config(params,[dict(path=str(asset))])
    restored=create_spectrum_generator(config)
    target=ReactionFragmentTreeTrainingModel(restored.feature_model,spectrum_model=restored.spectrum_model)
    initialize_imports(target,config);restored.train()
    original=generator.feature_model.reaction_instance_encoder.global_encoders[pid]
    imported=restored.feature_model.reaction_instance_encoder.global_encoders[pid]
    assert not imported.training
    assert all(not p.requires_grad for p in imported.parameters())
    for key,tensor in original.state_dict().items():
        assert torch.equal(tensor,imported.state_dict()[key])
