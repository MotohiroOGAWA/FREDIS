"""Frozen-base, low-rank expansion shared by training and inference."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankExpansion(nn.Module):
    """W_eff = W_frozen + up @ down; zero initial residual, width extra nodes."""

    def __init__(self, weight: torch.Tensor, width: int):
        super().__init__()
        self.down = nn.Parameter(weight.new_empty(width, weight.shape[1]))
        self.up = nn.Parameter(weight.new_zeros(weight.shape[0], width))
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def forward(self, original):
        return original + self.up @ self.down


class NewCategoryRows(nn.Module):
    """Only newly introduced category rows are trainable (including under AdamW)."""

    def __init__(self, weight: torch.Tensor, frozen_rows: list[int]):
        super().__init__()
        frozen = set(frozen_rows)
        rows = [index for index in range(weight.shape[0]) if index not in frozen]
        self.register_buffer('rows', torch.tensor(rows, dtype=torch.long, device=weight.device))
        self.new_rows = nn.Parameter(weight.detach()[rows].clone())
        nn.init.normal_(self.new_rows, std=0.02)

    def forward(self, original):
        return original.index_copy(0, self.rows, self.new_rows)


def install_expansion(generator, config):
    """Called by the generator constructor, before any optimizer is created.

    ReactionInstanceEncoder.categories is an nn.ModuleList indexed
    (pattern_id, reaction_rule_id, product_molecule_id); category_mapping() in
    training/fragment_tree_training/fine_tuning.py returns semantic names, so
    callers translate them to this positional attribute path before calling in.
    """
    if config.get('version') != 1 or int(config.get('width', 0)) < 1:
        raise ValueError('Unsupported fine-tuning version or nonpositive adapter width.')
    width = int(config['width'])
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    old_patterns = {str(pair[1]) for pair in config['category_mapping']['categories.0']}
    new_reactants = [encoder for pid, encoder in generator.feature_model.reaction_instance_encoder.reactant_encoders.items()
                    if pid not in old_patterns]
    new_reactant_modules = {id(module) for encoder in new_reactants for module in encoder.modules()}
    for encoder in new_reactants:
        encoder.requires_grad_(True)
    mol_modules = {id(module) for module in generator.mol_encoder.modules()}
    # Snapshot first: newly registered low-rank parameters must not be expanded again.
    for _, module in list(generator.named_modules()):
        if id(module) in mol_modules or id(module) in new_reactant_modules:
            continue
        names = ['weight'] if isinstance(module, nn.Linear) else []
        if isinstance(module, nn.MultiheadAttention):
            names = [name for name in ('in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight')
                     if getattr(module, name, None) is not None]
        for name in names:
            parametrize.register_parametrization(module, name, LowRankExpansion(getattr(module, name), width))
    categories = generator.feature_model.reaction_instance_encoder.categories
    for name, mapping in config['category_mapping'].items():
        embedding = categories[int(name.rsplit('.', 1)[1])]
        parametrize.register_parametrization(
            embedding, 'weight', NewCategoryRows(embedding.weight, [pair[1] for pair in mapping]))
    generator.feature_model.freeze_mol_encoder()
