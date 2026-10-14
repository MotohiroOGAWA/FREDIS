"""Combine immutable per-pattern reaction features without averaging weights."""
from copy import deepcopy
from pathlib import Path
import re
import torch

from fredis.domain.fragment import Fragmenter
from fredis.domain.fragment.reaction import ReactionPatternSet
from fredis.ml.specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
from .model import ReactionFragmentTreeTrainingModel

PATTERN_KEY = re.compile(r'\.(?:reactant_encoders|pattern_heads|neighborhood_encoders|global_encoders)\.(\d+)\.')


def integrate_checkpoints(checkpoints, head_checkpoint, output):
    paths = [Path(path).resolve() for path in checkpoints]
    head = Path(head_checkpoint).resolve()
    if head not in paths:
        raise ValueError('Head checkpoint must be one of the input checkpoints')
    destination = Path(output).resolve()
    if destination.exists() or destination.with_suffix('.fragmenter.pft').exists() or destination in paths:
        raise ValueError('Integration output must be a new file')
    saved = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    for checkpoint in saved:
        if checkpoint.get('architecture') != SourceAnchoredFragmentSpectrumGenerator.architecture:
            raise ValueError('Only fragment-tree checkpoints can be integrated')
        if any('.parametrizations.' in key for key in checkpoint['model_state_dict']):
            raise ValueError('Adapter checkpoints cannot be integrated; retrain with immutable pattern features')
    selected = saved[paths.index(head)]
    config = deepcopy(selected['model_config'])
    config = config.get('params', config)
    selected_params = selected['model_config'].get('params', selected['model_config'])
    fragments = [Fragmenter.from_dict(item['model_config'].get('params', item['model_config'])['fragmenter_params']) for item in saved]
    common = {key: tensor for key, tensor in selected['model_state_dict'].items()
              if key.startswith(('feature_model.mol_encoder.', 'spectrum_model.mol_encoder.',
                                 'feature_model.reaction_instance_encoder.')) and not PATTERN_KEY.search(key)}
    definitions = {}; owners = {}; pattern_tensors = {}; architectures = {}
    reference_chemistry = fragments[0].to_dict()
    reference_chemistry['fragment_ion_tree_builder'].pop('reaction_pattern_set')
    for path, checkpoint, fragmenter in zip(paths, saved, fragments):
        chemistry = fragmenter.to_dict()
        chemistry['fragment_ion_tree_builder'].pop('reaction_pattern_set')
        if chemistry != reference_chemistry:
            raise ValueError(f'Incompatible chemistry configuration: {path}')
        params = checkpoint['model_config'].get('params', checkpoint['model_config'])
        if params['mol_encoder_params'] != selected_params['mol_encoder_params']:
            raise ValueError(f'Conflicting molecular feature configuration: {path}')
        for name in ('hidden_dim', 'reaction_neighborhood_mode', 'reaction_neighborhood_max_hop'):
            if params.get('reaction_model_params', {}).get(name) != selected_params.get('reaction_model_params', {}).get(name):
                raise ValueError(f'Conflicting shared reaction feature setting {name}: {path}')
        state = checkpoint['model_state_dict']
        for key, value in common.items():
            if key not in state or not torch.equal(value, state[key]):
                raise ValueError(f'Conflicting shared feature tensor {key} in {path}')
        for pattern in fragmenter.reaction_pattern_set.patterns:
            tensors = {}
            for key, value in state.items():
                match = PATTERN_KEY.search(key)
                if match and int(match.group(1)) == pattern.pattern_id:
                    normalized = key[:match.start(1)] + '{pattern_id}' + key[match.end(1):]
                    tensors[normalized] = value
            if not tensors or not any('.pattern_heads.' in key for key in tensors):
                raise ValueError('Checkpoint lacks complete pattern-owned feature readouts')
            identity = pattern.key
            from .pattern_models import pattern_config
            if identity in architectures and architectures[identity] != pattern_config(params, pattern.pattern_id):
                raise ValueError(f'Conflicting architecture for pattern {pattern.name}')
            if identity in pattern_tensors:
                prior = pattern_tensors[identity]
                if set(prior) != set(tensors) or any(not torch.equal(prior[key], tensor) for key, tensor in tensors.items()):
                    raise ValueError(f'Conflicting weights for pattern {pattern.name}; select compatible branches')
            else:
                from .pattern_models import pattern_config
                architectures[identity] = pattern_config(params, pattern.pattern_id)
                definitions[identity] = pattern.to_dict()
                definitions[identity].pop('pattern_id', None)
                owners[identity] = str(path)
                pattern_tensors[identity] = tensors
    union = ReactionPatternSet.from_dict({'name': 'integrated_patterns', 'patterns': list(definitions.values())})
    # PatternSet assigns IDs from chemical definitions, so transfer by key.
    config['fragmenter_params']['fragment_ion_tree_builder']['reaction_pattern_set'] = union.to_dict()
    role_tensor=selected['model_state_dict'].get('feature_model.reaction_instance_encoder.roles.weight')
    if role_tensor is not None:
        config.setdefault('reaction_model_params',{})['max_roles']=role_tensor.shape[0]
    config.setdefault('reaction_model_params', {})['pattern_model_params'] = {str(p.pattern_id): architectures[p.key] for p in union.patterns}
    config['fine_tuning'] = {'version': 2, 'frozen_pattern_ids': [p.pattern_id for p in union.patterns],
                             'pattern_mapping': [], 'source_checkpoint': str(head)}
    from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
    generator = create_spectrum_generator(config)
    model = ReactionFragmentTreeTrainingModel(generator.feature_model, spectrum_model=generator.spectrum_model)
    target = model.state_dict()
    by_id = {p.pattern_id: p.key for p in union.patterns}
    for key, value in target.items():
        match = PATTERN_KEY.search(key)
        if match:
            normalized = key[:match.start(1)] + '{pattern_id}' + key[match.end(1):]
            source = pattern_tensors[by_id[int(match.group(1))]][normalized]
        else:
            source = selected['model_state_dict'].get(key)
        if source is None or source.shape != value.shape:
            raise ValueError(f'Incompatible architecture tensor: {key}')
        value.copy_(source)
    model.load_state_dict(target, strict=True)
    report = {'inputs': list(map(str, paths)), 'head_checkpoint': str(head),
              'patterns': [{'id': p.pattern_id, 'name': p.name, 'source': owners[p.key]} for p in union.patterns],
              'next_step': 'Prepare union-pattern data with --pretrained-pattern-model OUTPUT, then train using those datasets'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'architecture': generator.architecture, 'model_config': config,
                'model_state_dict': model.state_dict(), 'epoch': 0, 'integration': report}, destination)
    import json
    destination.with_suffix('.fragmenter.pft').write_text(json.dumps({
        'fragmenter_params':config['fragmenter_params'],
        'symbols':config['mol_encoder_params']['symbols']},indent=2))
    return report
