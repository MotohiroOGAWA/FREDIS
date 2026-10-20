import pytest
import torch
from fredis.libs.mmkit.mmkit import Compound,Adduct
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.competitive_spectrum_model import propagate_probability
from fredis.ml.common.torch_utils.segment_ops import segment_softmax
from fredis.ml.specgen.components.condition.collision_energy_feature import CollisionEnergyFeatureLayer
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.input.reaction_batching import select_samples
from .TestReactionPipeline import config,chemistry


def competitive_config():
    value=config()
    value['spectrum_model_params'].update(ce_fc_dims=[12,8],collision_energy_dim=8,ce_scale=100.,ce_center=0.,
        transition_fc_dims=[16,8],transition_dropout=0.,propagation_steps=3)
    return value


def test_flow_competes_and_adds_diamond_paths():
    # 0 -> 1/2 -> 3. Self transitions compete too; mass must remain one.
    source=torch.tensor([0,0,0,1,1,2,2,3]);target=torch.tensor([0,1,2,1,3,2,3,3])
    logits=torch.zeros(8,requires_grad=True)
    probs=segment_softmax(logits,source,4)
    torch.testing.assert_close(probs[:3],torch.full((3,),1/3))
    final=propagate_probability(torch.tensor([1.,0.,0.,0.]),source,target,probs,2)
    torch.testing.assert_close(final.sum(),torch.tensor(1.))
    torch.testing.assert_close(final[3],torch.tensor(1/3))
    (-final[3].log()).backward();assert logits.grad.abs().sum()>0


def test_ce_is_normalized_scalar_mlp_and_roundtrips():
    layer=CollisionEnergyFeatureLayer(3,fc_dims=(),center=10.,scale=100.)
    with torch.no_grad():layer.proj[0].weight.fill_(1);layer.proj[0].bias.zero_()
    torch.testing.assert_close(layer(torch.tensor([20.,21.])),torch.tensor([[.1]*3,[.11]*3]))
    assert 'frequencies' not in layer.state_dict()
    with pytest.raises(ValueError):CollisionEnergyFeatureLayer(8,scale=0.)


def test_competitive_training_prediction_and_checkpoint():
    torch.manual_seed(9)
    params=competitive_config();generator=create_spectrum_generator(params).eval()
    source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    data,_=ReactionStructureBuilder(generator).build(source,[adduct,adduct],[20.,40.],[[18.033826,44.049476]]*2,[[.5,1.]]*2)
    assert data.spectrum_batch.decoded.reaction_state_graph
    features=generator.feature_model(data,training_pool=True)
    output=generator.spectrum_model(data.spectrum_batch,reaction_h=features.reaction_h,
        collision_energy_h=features.collision_energy_h,source_embeddings=features.source_embeddings)
    assert torch.isfinite(output.loss) and output.metrics['probability_mass_error']<1e-5
    output.loss.backward()
    assert next(iter(generator.spectrum_model.transition_heads.values())).layers[0].weight.grad.abs().sum()>0
    assert generator.feature_model.collision_energy_encoder.proj[0].weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in generator.feature_model.reaction_instance_encoder.parameters())
    restored=create_spectrum_generator(params).eval();restored.load_state_dict(generator.state_dict(),strict=True)
    with torch.no_grad():
        pred=generator.predict([source],[adduct],[20.])
        again=restored.predict([source],[adduct],[20.])
    torch.testing.assert_close(pred.spectra.intensity,again.spectra.intensity)
    assert pred.spectra.metrics['probability_mass_error']<1e-5
    from fredis.ml.specgen.competitive_spectrum_model import merge_candidate_peaks
    merged=merge_candidate_peaks(pred.spectra)
    valid = (pred.spectra.formula_tensor[:, :-1] >= 0).all(1) & (pred.spectra.mz > 0)
    torch.testing.assert_close(merged.intensity.sum(),pred.spectra.intensity[valid].sum())
    torch.testing.assert_close(merged.intensity.max(),torch.tensor(1.))
    torch.testing.assert_close(merged.ion_probability.sum(), pred.spectra.ion_probability[valid].sum())
    probabilities=pred.spectra.transition_probability
    sources=pred.spectra.transition_source_index
    totals=probabilities.new_zeros(pred.spectra.intensity.numel()).scatter_add_(0,sources,probabilities)
    torch.testing.assert_close(totals,torch.ones_like(totals))
    assert torch.any(pred.spectra.transition_explanation_index>=0)

    assert not any('tree_encoder' in k for k in generator.spectrum_model.state_dict())
    subset=select_samples(data,[1]);assert subset.spectrum_batch.precursor_adducts==('[M+H]+',)
    from fredis.ml.input.source_reaction_structure import SourceReactionStructure
    batched=SourceReactionStructure.from_structures([subset,subset])
    f=generator.feature_model(batched,training_pool=True)
    b=generator.spectrum_model(batched.spectrum_batch,reaction_h=f.reaction_h,
        collision_energy_h=f.collision_energy_h,source_embeddings=f.source_embeddings)
    assert b.metrics['probability_mass_error']<1e-5
    assert len(batched.spectrum_batch.precursor_adducts)==2



def test_preparation_context_keeps_state_graph():
    from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context
    data,_=ReactionStructureBuilder(create_preparation_context(chemistry())).build(Compound.from_smiles('CC(O)N'),[Adduct.parse('[M+H]+')],[20.],[[44.049476]],[[1.]])
    assert data.spectrum_batch.decoded.reaction_state_graph


def test_mlp_cli_forwarding_and_configuration():
    from fredis.ml.training.fragment_tree_training.training import build_arg_parser,training_model_config
    from fredis.cli.base import namespace_argv
    parser=build_arg_parser()
    args=parser.parse_args(['--train-dir','train','--val-dir','val','--output-dir','out',
        '--ce-fc-dims','24,12','--ce-scale','50','--collision-energy-feature-dim','7',
        '--transition-fc-dims','48,24','--propagation-steps','4'])
    forwarded=parser.parse_args(namespace_argv(parser,args))
    cfg=training_model_config(forwarded)['spectrum_model_params']
    assert cfg['ce_fc_dims']==[24,12] and cfg['collision_energy_dim']==7 and cfg['ce_scale']==50
    assert cfg['transition_fc_dims']==[48,24] and cfg['propagation_steps']==4
    assert parser.parse_args(namespace_argv(parser,parser.parse_args(['--train-dir','t','--val-dir','v','--output-dir','o','--transition-fc-dims','']))).transition_fc_dims==[]




def test_all_parents_survive_reaction_set_deduplication():
    from fredis.ml.specgen.components.reaction.reaction_decoder import ReactionPool,ReactionDecoderOutput,with_all_transitions
    from fredis.ml.specgen.components.reaction.reaction_compatibility import ReactionCompatibilityEngine
    masks=torch.zeros((1,2,2),dtype=torch.bool)
    pool=ReactionPool(torch.tensor([[0,1]]),torch.ones((1,2),dtype=torch.bool),torch.randn(1,2,4),masks,masks,masks,torch.ones((1,2,1),dtype=torch.bool),torch.ones((1,1),dtype=torch.bool))
    decoded=ReactionDecoderOutput(torch.zeros(4,dtype=torch.long),torch.tensor([[-1,-1],[0,-1],[1,-1],[0,1]]),torch.zeros(4),torch.tensor([False,False,False,True]),torch.tensor([-1,0,0,1]),torch.tensor([-1,0,1,1]),pool)
    result=with_all_transitions(decoded,ReactionCompatibilityEngine(2))
    assert set(result.transition_parent_index[result.transition_child_index==3].tolist())=={1,2}


def test_neutral_loss_precursor_initialization():
    generator=create_spectrum_generator(competitive_config()).eval()
    output=generator.predict([Compound.from_smiles('CCO')],[Adduct.parse('[M+H-H2O]+')],[20.])
    assert output.spectra.metrics['probability_mass_error']<1e-5


def test_configurable_adduct_embedding_and_dropout_free_ce():
    params = competitive_config()
    params['spectrum_model_params']['adduct_embedding_dim'] = 7
    generator = create_spectrum_generator(params)
    features = generator.feature_model
    assert features.adduct_encoder.embedding.embedding_dim == 7
    assert features.main_adduct_projection[0].in_features == 7
    ce = features.collision_energy_encoder
    assert not any(isinstance(module, torch.nn.Dropout) for module in ce.modules())
    energy = torch.tensor([10., 20.])
    torch.testing.assert_close(ce(energy), ce(energy), rtol=0, atol=0)
    for width in (0, -1, True):
        params['spectrum_model_params']['adduct_embedding_dim'] = width
        with pytest.raises(ValueError, match='adduct_embedding_dim'):
            create_spectrum_generator(params)


def test_main_adduct_condition_banks_have_independent_weights():
    params = config()
    rules = params['fragmenter_params']['fragment_ion_tree_builder']['fragment_ion_adduct_rule_set']
    rules['hydrogen_shift_min'], rules['hydrogen_shift_max'] = -5, 0
    sodium = next(rule for rule in rules['adduct_rules'] if rule['adduct_type'] == '[M+Na]+')
    sodium['fragment_ion_adducts'] = ['[M+Na]+', '[M+H]+']
    generator = create_spectrum_generator(params).eval()
    model = generator.spectrum_model
    na = model.transition_heads[model.head_index['[M+Na]+']]
    proton = model.transition_heads[model.head_index['[M+H]+']]
    assert na.layers[0].weight.shape == (12, 16, 64)
    assert na.layers[0].bias.shape == (12, 16)
    assert na.layers[-1].weight.shape == (12, 1, 16)
    assert isinstance(na.activation, torch.nn.GELU)
    x = torch.randn(2, 64)
    ids = torch.tensor([0, 1])
    before = proton(x, ids).detach().clone()
    with torch.no_grad():
        na.layers[-1].bias[0].add_(5)
    torch.testing.assert_close(proton(x, ids), before, rtol=0, atol=0)
    loss = na(x, ids).sum()
    loss.backward()
    assert na.layers[0].weight.grad[:2].abs().sum() > 0
    assert na.layers[0].weight.grad[2:].abs().sum() == 0
    assert proton.layers[0].weight.grad is None


def test_formula_and_mz_do_not_control_transitions_or_probability_flow():
    from dataclasses import replace
    generator = create_spectrum_generator(config()).eval()
    source, adduct = Compound.from_smiles('CC(O)N'), Adduct.parse('[M+H]+')
    data, _ = ReactionStructureBuilder(generator).build(source, [adduct], [20.], [[44.049476]], [[1.]])
    features = generator.feature_model(data)
    kwargs = dict(reaction_h=features.reaction_h, collision_energy_h=features.collision_energy_h,
                  source_embeddings=features.source_embeddings)
    original = generator.spectrum_model(data.spectrum_batch, **kwargs)
    modified = replace(data.spectrum_batch,
        physical_candidate_formula_tensor=data.spectrum_batch.physical_candidate_formula_tensor + 100,
        physical_candidate_mz=data.spectrum_batch.physical_candidate_mz + 1000,
        physical_candidate_charge=-data.spectrum_batch.physical_candidate_charge,
    )
    changed = generator.spectrum_model(modified, **kwargs)
    for field in ('transition_source_index', 'transition_target_index', 'transition_probability', 'ion_probability'):
        torch.testing.assert_close(getattr(original, field), getattr(changed, field), rtol=0, atol=0)


def test_inference_annotates_full_condition_grid_after_scoring(monkeypatch):
    import fredis.ml.specgen.fragment_spectrum_model as batches
    generator = create_spectrum_generator(config()).eval()
    model = generator.spectrum_model
    scored = []
    hook = next(iter(model.transition_heads.values())).register_forward_hook(lambda *args: scored.append(True))
    original = batches.annotate_ion_conditions
    def annotate(data, tensorizer):
        assert scored, 'Formula annotations must follow neural scoring'
        return original(data, tensorizer)
    monkeypatch.setattr(batches, 'annotate_ion_conditions', annotate)
    result = generator.predict([Compound.from_smiles('CCO')], [Adduct.parse('[M+H]+')], [20.])
    hook.remove()
    data = result.spectrum_batch
    per_node = len(model.main_adduct_conditions['[M+H]+'])
    assert data.physical_candidate_node_index.numel() == len(result.fragments.compounds) * per_node
    assert (data.physical_candidate_explanation_ptr.diff() == 1).all()
    index = 0
    node = int(data.physical_candidate_node_index[index])
    ion = Adduct.parse(data.ion_type_strs[int(data.explanation_adduct_index[index])])
    shift = Adduct.from_dict({'H': int(data.explanation_hydrogen_shift[index])})
    expected = ion.apply_to_formula(shift.apply_to_formula(result.fragments.compounds[node].formula)).normalized
    torch.testing.assert_close(result.spectra.formula_tensor[index], generator.tensorizer.formula_to_tensor(expected))
    assert float(result.spectra.mz[index]) == pytest.approx(expected.exact_mass)


def test_mixed_main_adduct_batch_routes_to_separate_banks():
    params = config()
    rules = params['fragmenter_params']['fragment_ion_tree_builder']['fragment_ion_adduct_rule_set']
    sodium = next(rule for rule in rules['adduct_rules'] if rule['adduct_type'] == '[M+Na]+')
    sodium['fragment_ion_adducts'] = ['[M+Na]+', '[M+H]+']
    generator = create_spectrum_generator(params).eval()
    called = []
    hooks = [generator.spectrum_model.transition_heads[generator.spectrum_model.head_index[main]].register_forward_hook(
        lambda module, args, output, main=main: called.append(main))
        for main in ('[M+H]+', '[M+Na]+')]
    source = Compound.from_smiles('CCO')
    result = generator.predict([source, source], [Adduct.parse('[M+H]+'), Adduct.parse('[M+Na]+')], [20., 40.])
    for hook in hooks:
        hook.remove()
    assert set(called) == {'[M+H]+', '[M+Na]+'}
    assert len(result.fragments.trees) == 2
    probabilities = result.spectra.ion_probability
    totals = probabilities.new_zeros(2).scatter_add_(0, result.spectra.sample_index, probabilities)
    torch.testing.assert_close(totals, torch.ones(2))
    invalid = (result.spectra.formula_tensor[:, :-1] < 0).any(1)
    assert invalid.any(), 'The neural grid must not be pruned by output formula validity'
    from fredis.ml.specgen.competitive_spectrum_model import merge_candidate_peaks
    merged = merge_candidate_peaks(result.spectra)
    assert (merged.formula_tensor[:, :-1] >= 0).all()
    assert (merged.mz > 0).all()
