"""Prepare and validate frozen-base fine-tuning for an expanded reaction pattern set."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re

import torch

from fredis.domain.fragment import Fragmenter
from fredis.domain.fragment.reaction import ReactionPatternSet


def category_mapping(old_set, new_set):
    """Match chemical definitions, not positional IDs that may be sorted on import."""
    def categories(pattern_set):
        patterns, rules, products = {}, {}, {}
        ids = set()
        for pattern in pattern_set.patterns:
            if pattern.pattern_id < 0 or pattern.pattern_id in ids or pattern.key in patterns:
                raise ValueError('Pattern IDs and chemical definitions must be unique.')
            ids.add(pattern.pattern_id)
            patterns[pattern.key] = int(pattern.pattern_id)
            for rule in pattern.reaction_rules:
                key = (pattern.key, int(rule.id))
                rules[key] = len(rules)
                for product_id, _ in enumerate(rule.prod_idx_to_maps):
                    products[(*key, product_id)] = len(products)
        if ids != set(range(len(ids))):
            raise ValueError('Pattern IDs must be contiguous from zero.')
        return patterns, rules, products

    old = categories(old_set)
    new = categories(new_set)
    if not old[0] or len(new[0]) <= len(old[0]):
        raise ValueError('Select a set containing every old pattern and at least one new pattern.')
    mapping = {}
    for name, before, after in zip(('pattern_embedding', 'reaction_rule_embedding', 'product_embedding'), old, new):
        if not before.keys() <= after.keys():
            raise ValueError('The new set removes or changes an existing reaction definition.')
        mapping[name] = [[index, after[key]] for key, index in before.items()]
    return mapping


def _local_category_sizes(pattern_set):
    """Match SourceAnchoredFragmentSpectrumGenerator's category_sizes exactly."""
    reaction_size = max((len(pattern.reaction_rules) for pattern in pattern_set.patterns), default=0)
    product_size = max((len(rule.prod_temp) for pattern in pattern_set.patterns
                        for rule in pattern.reaction_rules), default=1)
    return reaction_size, product_size


def _reaction_category_mapping(old_set, new_set):
    """Map old_set's rows into new_set's, for ReactionInstanceEncoder.categories.

    categories[0] (pattern) is keyed by the globally unique pattern identity,
    so it reuses category_mapping()'s chemistry-identity matching directly.
    categories[1]/[2] (reaction/product) are instead *local* position indices
    shared additively by every pattern (row k always means "the k-th reaction
    of whichever pattern produced this reaction"), so old positions never move:
    expanding the pattern set can only ever append new local rows at the end.
    """
    pattern_mapping = category_mapping(old_set, new_set)['pattern_embedding']
    old_reaction_size, old_product_size = _local_category_sizes(old_set)
    new_reaction_size, new_product_size = _local_category_sizes(new_set)
    if new_reaction_size < old_reaction_size or new_product_size < old_product_size:
        raise ValueError('The new pattern set has fewer reactions/products per pattern than the base; cannot fine-tune.')
    return {
        'categories.0': pattern_mapping,
        'categories.1': [[index, index] for index in range(old_reaction_size)],
        'categories.2': [[index, index] for index in range(old_product_size)],
    }


def prepare_model_config(*, checkpoint_path, pattern_set_path, new_params_path, width=8):
    """Build a fine-tuning model_config from a base reaction checkpoint.

    new_params_path must already carry the complete, expanded fragmenter_params
    (produced the same way as any other reaction model config); the training
    data itself must be regenerated with that same config before fine-tuning,
    since resolve_precursor_reactions and the reaction universe both depend on it.
    """
    if width < 1:
        raise ValueError('Adapter width must be positive.')
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
    if base_config.get('fine_tuning'):
        raise ValueError('This command starts from a base checkpoint. Resume an existing fine-tuning run with --resume.')
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
    mapping = _reaction_category_mapping(old_fragmenter.reaction_pattern_set, new_fragmenter.reaction_pattern_set)
    config = deepcopy(new_config)
    config['fragmenter_params'] = new_fragmenter.to_dict()
    config['fine_tuning'] = dict(version=1, width=int(width), source_checkpoint=str(checkpoint_path), pattern_set_path=str(pattern_set_path), category_mapping=mapping)
    return config


def initialize_from_base(model, model_config):
    """Copy every old tensor; retain only new rows/adapter initialization and new lookup tables."""
    config = model_config['fine_tuning']
    source = torch.load(config['source_checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
    target = model.state_dict()
    used = set()
    for key, value in target.items():
        if re.search(r'\.parametrizations\.[^.]+\.0\.', key):
            continue  # adapter/row values and new-row index buffer
        original_key = re.sub(r'\.parametrizations\.([^.]+)\.original$', r'.\1', key)
        pattern_match = re.search(r'\.reactant_encoders\.(\d+)\.', original_key)
        if pattern_match:
            inverse = {int(new): int(old) for old, new in config['category_mapping']['categories.0']}
            new_id = int(pattern_match.group(1))
            if new_id not in inverse:
                continue  # entirely new pattern-specific encoder
            original_key = (original_key[:pattern_match.start(1)] + str(inverse[new_id])
                            + original_key[pattern_match.end(1):])
        if original_key not in source:
            raise ValueError(f'Base checkpoint is missing {original_key}')
        old_value = source[original_key]
        used.add(original_key)
        if key.endswith(('.reaction_rule_lookup_table', '.product_lookup_table')):
            continue  # constructed using the new definitions
        category = next((name for name in config['category_mapping']
                         if key.endswith(f'.{name}.parametrizations.weight.original')), None)
        if category:
            pairs = config['category_mapping'][category]
            if old_value.ndim != 2 or old_value.shape[1] != value.shape[1] or len(pairs) != old_value.shape[0]:
                raise ValueError(f'Incompatible category weights: {key}')
            for old_row, new_row in pairs:
                value[new_row].copy_(old_value[old_row])
        else:
            if old_value.shape != value.shape:
                raise ValueError(f'Base checkpoint shape mismatch for {original_key}')
            value.copy_(old_value)
    unexpected = set(source) - used
    if unexpected:
        raise ValueError(f'Unexpected base checkpoint tensors: {sorted(unexpected)[:5]}')
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
