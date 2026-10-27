"""Freeze complete reaction features; train new patterns and downstream predictors."""

from __future__ import annotations


def configure_pattern_training(generator, config):
    """Configure trainability without adding layers or altering weights."""
    generator.requires_grad_(True)
    generator.feature_model._freeze_mol_encoder = False
    generator.mol_encoder.train(generator.training)
    reaction = generator.feature_model.reaction_instance_encoder
    reaction.requires_grad_(False)
    frozen = {str(pid) for pid in config["frozen_pattern_ids"]}
    unknown = frozen - set(reaction.reactant_encoders)
    if unknown:
        raise ValueError(f"Unknown frozen reaction patterns: {sorted(unknown)}")
    reaction.frozen_pattern_ids = frozen
    for pid, encoder in reaction.reactant_encoders.items():
        if pid not in frozen:
            encoder.requires_grad_(True)
            reaction.pattern_heads[pid].requires_grad_(True)
            reaction.endpoint_projections[pid].requires_grad_(True)
        else:
            encoder.eval()
            reaction.pattern_heads[pid].eval()
            reaction.endpoint_projections[pid].eval()
