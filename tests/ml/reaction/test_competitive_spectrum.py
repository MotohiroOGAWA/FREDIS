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
        state_fc_dims=[12],state_hidden_dim=8,transition_fc_dims=[16,8],transition_dropout=0.,propagation_steps=3)
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
        main_adduct_h=features.sample_main_adduct_h,collision_energy_h=features.collision_energy_h,source_embeddings=features.source_embeddings)
    assert torch.isfinite(output.loss) and output.metrics['probability_mass_error']<1e-5
    output.loss.backward()
    assert generator.spectrum_model.transition_score[0].weight.grad.abs().sum()>0
    assert generator.spectrum_model.state_encoder[0].weight.grad.abs().sum()>0
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
    torch.testing.assert_close(merged.intensity.sum(),pred.spectra.intensity.sum())
    torch.testing.assert_close(merged.intensity.max(),torch.tensor(1.))
    torch.testing.assert_close(merged.ion_probability.sum(),torch.tensor(1.))
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
    b=generator.spectrum_model(batched.spectrum_batch,reaction_h=f.reaction_h,main_adduct_h=f.sample_main_adduct_h,
        collision_energy_h=f.collision_energy_h,source_embeddings=f.source_embeddings)
    assert b.metrics['probability_mass_error']<1e-5
    assert batched.spectrum_batch.precursor_formula_tensor.shape[0]==2



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
        '--ion-state-fc-dims','32,16','--transition-fc-dims','48,24','--propagation-steps','4'])
    forwarded=parser.parse_args(namespace_argv(parser,args))
    cfg=training_model_config(forwarded)['spectrum_model_params']
    assert cfg['ce_fc_dims']==[24,12] and cfg['collision_energy_dim']==7 and cfg['ce_scale']==50
    assert cfg['state_fc_dims']==[32,16] and cfg['transition_fc_dims']==[48,24] and cfg['propagation_steps']==4
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
    assert features.intensity_main_adduct_projection[0].in_features == 7
    assert not any(isinstance(module, torch.nn.Dropout) for module in generator.spectrum_model.state_encoder.modules())
    ce = features.collision_energy_encoder
    assert not any(isinstance(module, torch.nn.Dropout) for module in ce.modules())
    energy = torch.tensor([10., 20.])
    torch.testing.assert_close(ce(energy), ce(energy), rtol=0, atol=0)
    for width in (0, -1, True):
        params['spectrum_model_params']['adduct_embedding_dim'] = width
        with pytest.raises(ValueError, match='adduct_embedding_dim'):
            create_spectrum_generator(params)
