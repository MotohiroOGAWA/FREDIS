from copy import deepcopy
import json

import pytest
import torch

from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel
from fredis.ml.training.fragment_tree_training.integration import integrate_checkpoints
from ..reaction.TestReactionPipeline import config


@pytest.fixture
def checkpoints(tmp_path):
    params = config()
    generator = create_spectrum_generator(params)
    wrapper = ReactionFragmentTreeTrainingModel(generator.feature_model,spectrum_model=generator.spectrum_model)
    checkpoint = {'architecture':generator.architecture,'model_config':params,'model_state_dict':wrapper.state_dict()}
    a,b = tmp_path/'a.pt', tmp_path/'b.pt'
    torch.save(checkpoint,a);torch.save(checkpoint,b)
    return a,b,tmp_path/'combined.pt'


def test_integration_preserves_patterns_and_exports_chemistry(checkpoints):
    a,b,output = checkpoints
    report = integrate_checkpoints([a,b],a,output)
    before = torch.load(a,weights_only=False)
    merged = torch.load(output,weights_only=False)
    assert report['patterns']
    assert 'version' not in merged['model_config']['fine_tuning']
    assert all(torch.equal(value,merged['model_state_dict'][key]) for key,value in before['model_state_dict'].items())
    chemistry = json.loads(output.with_suffix('.fragmenter.pft').read_text())
    assert chemistry['fragmenter_params'] == merged['model_config']['fragmenter_params']
    assert not any('.parametrizations.' in key for key in merged['model_state_dict'])
    with pytest.raises(ValueError,match='new file'):
        integrate_checkpoints([a,b],a,output)


@pytest.mark.parametrize('scope,message', [('shared','shared feature'),('pattern','Conflicting weights'),('mode','feature setting')])
def test_integration_rejects_conflicting_features(checkpoints,scope,message):
    a,b,output = checkpoints
    checkpoint = torch.load(b,weights_only=False)
    if scope=='mode':
        checkpoint['model_config'].setdefault('reaction_model_params',{})['hidden_dim']=32
    else:
        key = next(key for key in checkpoint['model_state_dict'] if
                   (key.startswith('feature_model.mol_encoder.') and checkpoint['model_state_dict'][key].is_floating_point() if scope=='shared'
                    else '.pattern_heads.' in key))
        checkpoint['model_state_dict'][key] = checkpoint['model_state_dict'][key]+0.1
    torch.save(checkpoint,b)
    with pytest.raises(ValueError,match=message):
        integrate_checkpoints([a,b],a,output)
    assert not output.exists()


def test_head_only_rejects_new_untrained_pattern_features():
    from fredis.domain.fragment.reaction import ReactionPatternSet
    from fredis.ml.training.fragment_tree_training.fine_tuning import pattern_mapping
    first = dict(name='carbon',reactant_smarts='[#6:1]-[#6:2]',products=[dict(name='left',smarts='[*:1]')])
    second = dict(name='oxygen',reactant_smarts='[#6:1]-[#8:2]',products=[dict(name='left',smarts='[*:1]')])
    old = ReactionPatternSet.from_dict(dict(name='old',patterns=[first]))
    new = ReactionPatternSet.from_dict(dict(name='new',patterns=[first,second]))
    assert pattern_mapping(old,old,allow_same=True)==[[0,0]]
    assert len(pattern_mapping(old,new))==1
    with pytest.raises(ValueError,match='same reaction pattern'):
        pattern_mapping(old,new,allow_same=True)
