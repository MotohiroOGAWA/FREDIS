from copy import deepcopy
import pytest
import torch
from fredis.libs.mmkit.mmkit import Compound,Adduct
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from .TestReactionPipeline import config


def test_descriptors_only_affect_post_branch_intensity():
    torch.manual_seed(23)
    params=config();params['mol_encoder_params']['external_feature_dim']=0
    params['spectrum_model_params']['descriptors']=['TPSA','MolLogP']
    generator=create_spectrum_generator(params).eval()
    assert generator.mol_encoder.external_feature_projection is None
    reaction=generator.feature_model.reaction_instance_encoder
    assert all(encoder.bond_dim>0 for encoder in reaction.neighborhood_encoders.values())
    source=Compound.from_smiles('CC(O)N');adduct=Adduct.parse('[M+H]+')
    data,_=ReactionStructureBuilder(generator).build(source,[adduct],[20.],[[18.033826,44.049476]],[[.5,1.]])
    def output(value):
        features=generator.feature_model(value,training_pool=True)
        spectrum=generator.spectrum_model(value.spectrum_batch,reaction_h=features.reaction_h,
            main_adduct_h=features.sample_main_adduct_h,collision_energy_h=features.collision_energy_h,
            source_embeddings=features.source_embeddings)
        return features,spectrum
    with torch.no_grad():
        before_features,before=output(data)
        modified=deepcopy(data)
        modified.source_graph.descriptors+=100
        modified.spectrum_batch.decoded.node_graph.descriptors+=100
        after_features,after=output(modified)
    torch.testing.assert_close(before_features.reaction_h,after_features.reaction_h,rtol=0,atol=0)
    torch.testing.assert_close(before_features.source_embeddings,after_features.source_embeddings,rtol=0,atol=0)
    assert not torch.equal(before.intensity,after.intensity)
    _,train=output(data)
    train.intensity.square().sum().backward()
    grad=generator.spectrum_model.transition_score[0].weight.grad
    assert grad.abs().sum()>0


def test_new_patterns_reject_descriptor_inputs():
    params=config()
    params['mol_encoder_params']['external_feature_dim']=0
    pid=params['fragmenter_params']['fragment_ion_tree_builder']['reaction_pattern_set']['patterns'][0].get('pattern_id',0)
    params['reaction_model_params']['pattern_model_params']={str(pid):dict(descriptors=['TPSA'])}
    with pytest.raises(ValueError,match='intensity'):create_spectrum_generator(params)
