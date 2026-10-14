"""Prepare immutable-feature pattern expansion or downstream-only fine-tuning."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re

import torch

from fredis.domain.fragment import Fragmenter
from fredis.domain.fragment.reaction import ReactionPatternSet


def pattern_mapping(old_set, new_set, *, allow_same=False):
    """Match immutable feature modules by chemical definition, not numeric ID."""
    def definitions(pattern_set):
        result = {pattern.key: pattern.pattern_id for pattern in pattern_set.patterns}
        ids = {pattern.pattern_id for pattern in pattern_set.patterns}
        if len(result) != len(pattern_set.patterns) or ids != set(range(len(result))):
            raise ValueError('Pattern IDs and chemical definitions must be unique and contiguous')
        return result
    old, new = definitions(old_set), definitions(new_set)
    if not old or not old.keys() <= new.keys():
        raise ValueError('The new set removes or changes an existing reaction definition')
    if allow_same and old.keys() != new.keys():
        raise ValueError('--head-only requires the same reaction pattern set as the checkpoint')
    if not allow_same and old.keys() == new.keys():
        raise ValueError('Add at least one reaction pattern, or use --head-only')
    return [[pid, new[key]] for key, pid in old.items()]


def prepare_model_config(*, checkpoint_path, pattern_set_path, new_params_path, head_only=False):
    """Build a fine-tuning model_config from a base reaction checkpoint.

    new_params_path must already carry the complete, expanded fragmenter_params
    (produced the same way as any other reaction model config); the training
    data itself must be regenerated with that same config before fine-tuning,
    since resolve_precursor_reactions and the reaction universe both depend on it.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model_state_dict'), dict):
        raise ValueError('A reaction training checkpoint with model_state_dict is required.')
    from fredis.ml.specgen.source_anchored_spectrum_predictor import SourceAnchoredFragmentSpectrumGenerator
    if checkpoint.get('architecture') != SourceAnchoredFragmentSpectrumGenerator.architecture:
        raise ValueError('This command fine-tunes a Source-anchored reaction checkpoint.')
    base_config = deepcopy(checkpoint.get('model_config', {}))
    base_config = base_config.get('params', base_config)
    if 'fragmenter_params' not in base_config:
        raise ValueError('Checkpoint is missing fragmenter_params.')
    new_config = json.loads(Path(new_params_path).read_text())
    new_config = new_config.get('params', new_config)
    if 'fragmenter_params' not in new_config:
        raise ValueError('new_params_path is missing fragmenter_params.')
    old_fragmenter = Fragmenter.from_dict(base_config['fragmenter_params'])
    new_fragmenter = Fragmenter.from_dict(new_config['fragmenter_params'])
    requested = json.loads(Path(pattern_set_path).read_text())
    requested = ReactionPatternSet.from_dict(requested.get('reaction_pattern_set', requested))
    if sorted(p.key for p in requested.patterns) != sorted(p.key for p in new_fragmenter.reaction_pattern_set.patterns):
        raise ValueError('Selected reaction pattern set does not match new_params_path. Regenerate the training data with the new set first.')
    old_params, new_params = old_fragmenter.to_dict(), new_fragmenter.to_dict()
    for params in (old_params, new_params):
        params['fragment_ion_tree_builder'].pop('reaction_pattern_set')
    if old_params != new_params:
        raise ValueError('Only the reaction pattern set may change; keep adducts, depth and other Fragmenter settings unchanged.')
    mapping = pattern_mapping(old_fragmenter.reaction_pattern_set, new_fragmenter.reaction_pattern_set, allow_same=head_only)
    config = deepcopy(new_config)
    from .pattern_models import pattern_config
    config.setdefault('reaction_model_params',{}).setdefault('pattern_input_version',2)
    role_tensor=checkpoint['model_state_dict'].get('feature_model.reaction_instance_encoder.roles.weight')
    if role_tensor is not None:
        config['reaction_model_params']['max_roles']=role_tensor.shape[0]
    overrides = config.setdefault('reaction_model_params', {}).setdefault('pattern_model_params', {})
    for old_id, new_id in mapping:
        overrides[str(new_id)] = pattern_config(base_config, old_id)
    config['fragmenter_params'] = deepcopy(new_config['fragmenter_params'])
    config['fine_tuning'] = dict(version=2, source_checkpoint=str(checkpoint_path), pattern_set_path=str(pattern_set_path),
                                 pattern_mapping=mapping, frozen_pattern_ids=([p.pattern_id for p in new_fragmenter.reaction_pattern_set.patterns] if head_only else [pair[1] for pair in mapping]), head_only=head_only)
    return config


def initialize_from_base(model, model_config):
    """Copy shared tensors and complete existing pattern features by chemical identity."""
    config = model_config['fine_tuning']
    if config.get('pattern_imports'):
        from .pattern_models import initialize_imports
        return initialize_imports(model, model_config)
    source = torch.load(config['source_checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
    target = model.state_dict()
    inverse = {int(new): int(old) for old, new in config['pattern_mapping']}
    used = set()
    for key, value in target.items():
        original_key = key
        match = re.search(r'\.(?:reactant_encoders|pattern_heads|neighborhood_encoders)\.(\d+)\.', key)
        if match:
            pid = int(match.group(1))
            if pid not in inverse:
                continue
            original_key = key[:match.start(1)] + str(inverse[pid]) + key[match.end(1):]
        if original_key not in source:
            raise ValueError(f'Base checkpoint is missing {original_key}; legacy architectures require retraining')
        old = source[original_key]
        used.add(original_key)
        if old.shape != value.shape:
            raise ValueError(f'Base checkpoint shape mismatch for {original_key}')
        value.copy_(old)
    if set(source) - used:
        raise ValueError(f'Unexpected base checkpoint tensors: {sorted(set(source)-used)[:5]}')
    model.load_state_dict(target, strict=True)


def parameter_report(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[name] = parameter.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(groups.values())
    return dict(total_parameters=total, frozen_parameters=total-trainable,
                trainable_parameters=trainable, trainable_tensors=groups)
