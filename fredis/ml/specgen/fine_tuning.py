"""Freeze complete reaction features; train new patterns and downstream predictors."""
from __future__ import annotations


def configure_pattern_training(generator, config):
    """Configure trainability without adding layers or altering weights."""
    if config.get('version') != 2:
        raise ValueError('Legacy adapter checkpoints are incompatible; use a version 2 pattern-feature checkpoint.')
    generator.requires_grad_(True)
    generator.mol_encoder.requires_grad_(False)
    reaction = generator.feature_model.reaction_instance_encoder
    reaction.requires_grad_(False)
    frozen = {str(pid) for pid in config['frozen_pattern_ids']}
    unknown = frozen - set(reaction.reactant_encoders)
    if unknown:
        raise ValueError(f'Unknown frozen reaction patterns: {sorted(unknown)}')
    reaction.frozen_pattern_ids = frozen
    for pid, encoder in reaction.reactant_encoders.items():
        if pid not in frozen:
            encoder.requires_grad_(True)
            reaction.pattern_heads[pid].requires_grad_(True)
            if pid in reaction.neighborhood_encoders:
                reaction.neighborhood_encoders[pid].requires_grad_(True)
        else:
            encoder.eval()
            reaction.pattern_heads[pid].eval()
            if pid in reaction.neighborhood_encoders:
                reaction.neighborhood_encoders[pid].eval()
    generator.feature_model.freeze_mol_encoder()
