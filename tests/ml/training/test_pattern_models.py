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
        pid: dict(hidden_dim=12, graph_dim=19, num_layers=3, num_heads=3, dropout=0)}
    generator = create_spectrum_generator(params)
    reaction = generator.feature_model.reaction_instance_encoder
    assert reaction.reactant_encoders[pid].graph_dim == 19
    for other, config in reaction.pattern_model_params.items():
        if other != pid:
            assert config['descriptors'] == []
            assert config['num_layers'] == 2


def test_conflicting_shared_inputs_rejected(checkpoints):
    source, other, _ = checkpoints
    saved = torch.load(other, weights_only=False)
    key = next(k for k in saved['model_state_dict'] if k.startswith('feature_model.mol_encoder.') and saved['model_state_dict'][k].is_floating_point())
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
    inherited=inherit_model_config(params,*directories)
    assert inherited['fine_tuning']['frozen_pattern_ids']==[item['pattern']['pattern_id']]
    assert inherited['reaction_model_params']['pattern_model_params'][str(item['pattern']['pattern_id'])]==item['architecture']
    # Resume consumes its saved architecture without reopening original imports.
    source.unlink()
    resumed=inherit_model_config(inherited,*directories,saved_model=inherited)
    assert resumed==inherited


def test_raw_pattern_position_capacity_is_independent_of_legacy_roles(checkpoints):
    from fredis.ml.training.fragment_tree_training.pattern_models import prepare_pattern_models, apply_dataset_pattern_models
    source,_,_=checkpoints
    saved=torch.load(source,weights_only=False);params=saved['model_config'];first=inspect_model(source)['patterns'][0]
    large=dict(name='large',reactant_smarts='-'.join(f'[#6:{i}]' for i in range(1,71)),products=[dict(name='left',smarts='[*:1]')])
    chemistry=prepare_pattern_models(dict(fragmenter_params=params['fragmenter_params'],symbols=params['mol_encoder_params']['symbols']),[
        dict(pattern=first['pattern'],source=str(source),source_pattern_id=first['pattern']['pattern_id']),dict(pattern=large)])
    config={**params,'fragmenter_params':chemistry['fragmenter_params'],'reaction_model_params':{**params.get('reaction_model_params',{})}}
    config=apply_dataset_pattern_models(config,chemistry['reaction_models'])
    generator=create_spectrum_generator(config)
    model=ReactionFragmentTreeTrainingModel(generator.feature_model,spectrum_model=generator.spectrum_model)
    initialize_imports(model,config)
    reaction=generator.feature_model.reaction_instance_encoder
    assert not hasattr(reaction,'roles')
    assert any(encoder.position_count==70 and encoder.node_proj.in_features==encoder.atom_dim+70 for encoder in reaction.reactant_encoders.values())


def test_shared_dropout_is_authoritative(checkpoints):
    from fredis.ml.training.fragment_tree_training.pattern_models import use_shared_dropout

    source, _, _ = checkpoints
    original = torch.load(source, weights_only=False)["model_config"]
    params = deepcopy(original)
    reaction = params.setdefault("reaction_model_params", {})
    params["dropout"] = 0.35
    reaction["reactant_dropout"] = 0.1
    pid = str(Fragmenter.from_dict(params["fragmenter_params"]).reaction_pattern_set.patterns[0].pattern_id)
    reaction.setdefault("pattern_model_params", {}).setdefault(pid, {})["dropout"] = 0.8
    shared = use_shared_dropout(params)
    assert reaction["pattern_model_params"][pid]["dropout"] == 0.8
    assert "dropout" not in shared["reaction_model_params"]["pattern_model_params"][pid]
    generator = create_spectrum_generator(shared)
    encoders = generator.feature_model.reaction_instance_encoder
    for config in encoders.pattern_model_params.values():
        assert config["dropout"] == 0.35
    for encoder in encoders.reactant_encoders.values():
        drops = [module for module in encoder.modules() if isinstance(module, torch.nn.Dropout)]
        assert drops and all(module.p == 0.35 for module in drops)
    edge_drops = [module for module in generator.feature_model.edge_head.modules() if isinstance(module, torch.nn.Dropout)]
    assert edge_drops and all(module.p == 0.35 for module in edge_drops)
    for invalid in (-0.1, 1, float("nan")):
        params["dropout"] = invalid
        with pytest.raises(ValueError, match="Shared dropout"):
            use_shared_dropout(params)


def test_global_dropout_covers_all_new_networks_and_eval(checkpoints):
    from fredis.ml.training.fragment_tree_training.pattern_models import use_shared_dropout

    source, _, _ = checkpoints
    params = deepcopy(torch.load(source, weights_only=False)["model_config"])
    params["dropout"] = 0.4
    params.pop("fine_tuning", None)
    params["spectrum_model_params"].update(fragment_fc_dims=[], state_fc_dims=[])
    shared = use_shared_dropout(params)
    generator = create_spectrum_generator(shared)
    features = generator.feature_model
    groups = [
        generator.mol_encoder,
        features.edge_head,
        *features.reaction_instance_encoder.reactant_encoders.values(),
        *features.reaction_instance_encoder.neighborhood_encoders.values(),
        *features.reaction_instance_encoder.pattern_heads.values(),
        *generator.spectrum_model.fragment_heads.values(),
        *generator.spectrum_model.state_heads.values(),
    ]
    for network in groups:
        drops = [m for m in network.modules() if isinstance(m, torch.nn.Dropout)]
        assert drops and all(m.p == 0.4 for m in drops), type(network)
    for module in generator.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            assert module.dropout == 0.4
    head = next(iter(generator.spectrum_model.fragment_heads.values()))
    with torch.no_grad():
        head.layers[0].weight.fill_(1)
        head.layers[0].bias.zero_()
    values = torch.ones(128, head.layers[0].weight.shape[-1])
    condition = torch.zeros(128, dtype=torch.long)
    generator.train()
    assert not torch.equal(head(values, condition), head(values, condition))
    generator.eval()
    torch.testing.assert_close(head(values, condition), head(values, condition), rtol=0, atol=0)
    ce = features.collision_energy_encoder
    energy = torch.linspace(1, 100, 128)
    generator.train()
    assert not any(isinstance(m, torch.nn.Dropout) for m in ce.modules())
    assert not any(isinstance(m, torch.nn.Dropout) for m in features.adduct_encoder.modules())
    assert not any(isinstance(m, torch.nn.Dropout) for m in features.main_adduct_projection.modules())
    torch.testing.assert_close(ce(energy), ce(energy), rtol=0, atol=0)
    adduct_index = torch.zeros(128, dtype=torch.long)
    torch.testing.assert_close(features.adduct_encoder(adduct_index), features.adduct_encoder(adduct_index), rtol=0, atol=0)
    generator.eval()
    torch.testing.assert_close(ce(energy), ce(energy), rtol=0, atol=0)
    # Reconstructing a saved architecture preserves keys and rates.
    restored = create_spectrum_generator(shared)
    restored.load_state_dict(generator.state_dict(), strict=True)
    restored.eval()
    torch.testing.assert_close(ce(energy), restored.feature_model.collision_energy_encoder(energy))


def test_global_dropout_preserves_imported_features(checkpoints):
    from fredis.ml.training.fragment_tree_training.pattern_models import use_shared_dropout

    source, _, _ = checkpoints
    original = torch.load(source, weights_only=False)["model_config"]
    imported = assemble_config(original, [dict(path=str(source))])
    imported["dropout"] = 0.45
    shared = use_shared_dropout(imported)
    assert shared["mol_encoder_params"] == imported["mol_encoder_params"]
    before = create_spectrum_generator(imported)
    generator = create_spectrum_generator(shared)
    initialize_imports(generator, shared)
    generator.train()
    reaction = generator.feature_model.reaction_instance_encoder
    old_reaction = before.feature_model.reaction_instance_encoder
    protected = [(generator.mol_encoder, before.mol_encoder)]
    for pid in reaction.frozen_pattern_ids:
        for new_group, old_group in zip(
            (reaction.reactant_encoders, reaction.pattern_heads, reaction.neighborhood_encoders),
            (old_reaction.reactant_encoders, old_reaction.pattern_heads, old_reaction.neighborhood_encoders),
        ):
            if pid in new_group:
                protected.append((new_group[pid], old_group[pid]))
    for network, old in protected:
        assert not network.training
        assert not any(p.requires_grad for p in network.parameters())
        assert [m.p for m in network.modules() if isinstance(m, torch.nn.Dropout)] == [
            m.p for m in old.modules() if isinstance(m, torch.nn.Dropout)
        ]
    assert not any(isinstance(m, torch.nn.Dropout) for m in generator.feature_model.collision_energy_encoder.modules())
    assert generator.feature_model.collision_energy_encoder.training
    wrapper = ReactionFragmentTreeTrainingModel(generator.feature_model, spectrum_model=generator.spectrum_model)
    wrapper.train()
    assert not generator.mol_encoder.training
    assert all(not reaction.reactant_encoders[pid].training for pid in reaction.frozen_pattern_ids)


def test_leaf_networks_receive_dropout_at_construction(checkpoints):
    from fredis.ml.mol.mol_encoder import MolEncoder
    from fredis.ml.reaction.external_neighborhood import ExternalNeighborhoodEncoder
    from fredis.ml.specgen.components.condition.condition_linear import ConditionHead
    from fredis.ml.specgen.components.reaction.reaction_instance_encoder import PatternReactionReadout

    source, _, _ = checkpoints
    mol_params = torch.load(source, weights_only=False)["model_config"]["mol_encoder_params"]
    networks = [
        MolEncoder(**{**mol_params, "dropout": 0.3, "external_feature_dim": 4}),
        ExternalNeighborhoodEncoder(4, 8, 2, dropout=0.3),
        ConditionHead(2, 8, (), dropout=0.3),
        PatternReactionReadout(8, 8, 8, 2, 2, dropout=0.3),
    ]
    for network in networks:
        drops = [m for m in network.modules() if isinstance(m, torch.nn.Dropout)]
        assert drops and all(m.p == 0.3 for m in drops), type(network)


def test_model_dropout_defaults_to_zero(checkpoints):
    source, _, _ = checkpoints
    params = deepcopy(torch.load(source, weights_only=False)["model_config"])
    params.pop("dropout", None)
    generator = create_spectrum_generator(params)
    drops = [m for m in generator.modules() if isinstance(m, torch.nn.Dropout)]
    assert drops and all(m.p == 0.0 for m in drops)
    assert all(m.dropout == 0.0 for m in generator.modules() if isinstance(m, torch.nn.MultiheadAttention))


def test_individual_cli_pattern_parameters(checkpoints):
    from fredis.cli.base import namespace_argv
    from fredis.ml.training.fragment_tree_training.training import build_arg_parser
    from fredis.ml.training.fragment_tree_training.pattern_models import parse_pattern_parameters, apply_pattern_parameters

    source, _, _ = checkpoints
    params = torch.load(source, weights_only=False)["model_config"]
    pid = str(Fragmenter.from_dict(params["fragmenter_params"]).reaction_pattern_set.patterns[0].pattern_id)
    parser = build_arg_parser()
    flags = ["--dataset-dir", "dataset", "--output-dir", "output"]
    values = dict(hidden_dim=12, graph_dim=19, num_layers=3, num_heads=3, neighborhood_hops=0, neighborhood_hidden_dim=7)
    for key, value in values.items():
        flags += ["--pattern-" + key.replace("_", "-"), f"{pid}={value}"]
    args = parser.parse_args(flags)
    forwarded = parser.parse_args(namespace_argv(parser, args))
    overrides = parse_pattern_parameters(forwarded)
    assert overrides == {pid: values}
    resolved = apply_pattern_parameters(params, overrides)
    assert all(pattern_config(resolved, pid)[key] == value for key, value in values.items())
    generator = create_spectrum_generator(resolved)
    assert generator.feature_model.reaction_instance_encoder.reactant_encoders[pid].graph_dim == 19
    assert pid not in generator.feature_model.reaction_instance_encoder.neighborhood_encoders
    assert params != resolved
    with pytest.raises(ValueError, match="Unknown reaction"):
        apply_pattern_parameters(params, {"99999": values})
    with pytest.raises(ValueError, match="Pretrained"):
        apply_pattern_parameters({**params, "fine_tuning": {"frozen_pattern_ids": [pid]}}, overrides)
    with pytest.raises(ValueError, match="divisible"):
        apply_pattern_parameters(params, {pid: {"hidden_dim": 13, "num_heads": 3}})
    for invalid in ("x=12", "0=1.5", "-1=12", "0=0"):
        with pytest.raises(ValueError):
            parse_pattern_parameters(parser.parse_args(["--dataset-dir", "d", "--output-dir", "o", "--pattern-hidden-dim=" + invalid]))
    with pytest.raises(ValueError, match="Duplicate"):
        parse_pattern_parameters(parser.parse_args(flags + ["--pattern-hidden-dim", f"{pid}=12"]))
