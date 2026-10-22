"""Prepared-tensor branch MIL and physical-ion intensity training."""

from __future__ import annotations
from dataclasses import dataclass, replace
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ...specgen.components.reaction.reaction_decoder import ReactionPool
from ...specgen.fragment_spectrum_model import SpectrumLossSettings


def smooth_max_mil(path_scores: Tensor, peak_path_ptr: Tensor, temperature: float = 0.1) -> Tensor:
    """Normalized smooth maximum for alternative teacher paths of each peak."""
    if not 0 < temperature or not math.isfinite(temperature):
        raise ValueError("branch_mil_temperature must be positive and finite")
    counts = peak_path_ptr[1:] - peak_path_ptr[:-1]
    if torch.any(counts <= 0):
        raise ValueError("Every supervised peak needs at least one candidate path")
    owner = torch.repeat_interleave(torch.arange(counts.numel(), device=path_scores.device), counts)
    scaled = path_scores / temperature
    maximum = scaled.new_full((counts.numel(),), -torch.inf)
    maximum.scatter_reduce_(0, owner, scaled, reduce="amax", include_self=True)
    total = scaled.new_zeros(counts.numel()).scatter_add_(0, owner, (scaled - maximum[owner]).exp())
    return temperature * (maximum + total.log() - counts.to(path_scores.dtype).log())


def branch_mil_loss(
    logits: Tensor,
    data: object,
    inverse_reaction: Tensor,
    temperature: float = 0.1,
    negative_weight: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return total, positive MIL, weak-negative loss and path log scores."""
    step_state = data.teacher.path_step_state_index
    step_reaction = data.teacher.path_step_reaction_index
    if step_state.numel():
        owner = torch.repeat_interleave(
            torch.arange(data.teacher.path_step_ptr.numel() - 1, device=logits.device),
            data.teacher.path_step_ptr[1:] - data.teacher.path_step_ptr[:-1],
        )
        group = data.states.branch_group_index[step_state]
        local = inverse_reaction[group, step_reaction]
        if torch.any(local < 0):
            raise ValueError("Prepared positive reaction is absent from its branch group")
        step_log_probability = F.logsigmoid(logits[step_state, local])
        path_scores = logits.new_zeros(data.teacher.path_step_ptr.numel() - 1).scatter_add_(
            0, owner, step_log_probability
        )
        positive = -smooth_max_mil(path_scores, data.teacher.peak_path_ptr, temperature).mean()
    else:
        path_scores = logits.new_empty(0)
        positive = logits.sum() * 0
    negative_counts = (
        data.teacher.negative_reaction_ptr[1:] - data.teacher.negative_reaction_ptr[:-1]
    )
    negative_owner = torch.repeat_interleave(
        torch.arange(negative_counts.numel(), device=logits.device), negative_counts
    )
    if negative_owner.numel():
        group = data.states.branch_group_index[negative_owner]
        local = inverse_reaction[group, data.teacher.negative_reaction_index]
        valid = local >= 0
        negative = (
            F.softplus(logits[negative_owner[valid], local[valid]]).mean()
            if torch.any(valid)
            else logits.sum() * 0
        )
    else:
        negative = logits.sum() * 0
    return positive + negative_weight * negative, positive, negative, path_scores


@dataclass(frozen=True)
class ReactionTrainingOutput:
    loss: Tensor
    branch_loss: Tensor
    branch_logits: Tensor
    branch_positive_mask: Tensor
    pool: ReactionPool
    metrics: dict[str, Tensor]
    spectrum_output: object | None = None


class ReactionFragmentTreeTrainingModel(nn.Module):
    def __init__(
        self,
        feature_model: nn.Module,
        negative_weight: float = 0.5,
        spectrum_model: nn.Module | None = None,
        branch_mil_temperature: float = 0.1,
        branch_weight: float = 1.0,
        spectrum_loss: SpectrumLossSettings | None = None,
    ) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.spectrum_model = spectrum_model
        self.spectrum_loss = spectrum_loss or SpectrumLossSettings()
        self.negative_weight = negative_weight
        self.branch_mil_temperature = branch_mil_temperature
        self.branch_weight = branch_weight

    def set_checkpoint_model_config(self, config: dict) -> None:
        self._checkpoint_model_config = dict(config)

    def get_params(self) -> dict:
        return dict(getattr(self, "_checkpoint_model_config", {}))

    def forward(self, data: object) -> ReactionTrainingOutput:
        encoded = self.feature_model(data, training_pool=True)
        pool = encoded.pool
        groups = encoded.branch_main_adduct_h.shape[0]
        reactions = encoded.reaction_h.shape[0]
        inverse = torch.full(
            (groups, reactions + 1),
            -1,
            dtype=torch.long,
            device=encoded.reaction_h.device,
        )
        ids = pool.reaction_index.masked_fill(~pool.valid, reactions)
        inverse.scatter_(
            1, ids, torch.arange(pool.valid.shape[1], device=ids.device).expand_as(ids)
        )
        inverse[:, reactions] = -1
        state_count = data.states.branch_group_index.numel()
        owners = data.states.branch_group_index
        reaction_ptr, reaction_index = data.active_reactions()
        counts = reaction_ptr[1:] - reaction_ptr[:-1]
        row = torch.repeat_interleave(torch.arange(state_count, device=ids.device), counts)
        position = torch.arange(row.numel(), device=ids.device) - reaction_ptr[row]
        state = torch.full(
            (state_count, self.feature_model.max_reaction_count),
            -1,
            dtype=torch.long,
            device=ids.device,
        )
        # Independent set members reference the global concrete registry.
        local = inverse[owners[row], reaction_index]
        if torch.any(local < 0):
            raise ValueError("Prepared state reaction is absent from its branch group")
        state[row, position] = local
        sentinel = pool.valid.shape[1]
        state = state.masked_fill(state < 0, sentinel).sort(1).values
        state = state.masked_fill(state == sentinel, -1)
        logits, _ = self.feature_model.decoder.score_states(
            pool, encoded.branch_main_adduct_h, state, owners
        )
        branch_loss, positive_loss, negative_loss, path_scores = branch_mil_loss(
            logits, data, inverse, self.branch_mil_temperature, self.negative_weight
        )
        positive = torch.zeros_like(logits, dtype=torch.bool)
        next_counts = (
            data.teacher.positive_reaction_ptr[1:] - data.teacher.positive_reaction_ptr[:-1]
        )
        next_owner = torch.repeat_interleave(
            torch.arange(state_count, device=ids.device), next_counts
        )
        if next_owner.numel():
            next_local = inverse[owners[next_owner], data.teacher.positive_reaction_index]
            positive[next_owner, next_local] = True
        predicted = (logits.sigmoid() >= 0.5) & torch.isfinite(logits)
        metrics = {
            "branch_loss": branch_loss,
            "branch_positive_mil_loss": positive_loss,
            "branch_negative_loss": negative_loss,
            "num_branch_groups": logits.new_tensor(float(groups)),
            "num_teacher_paths": logits.new_tensor(float(path_scores.numel())),
        }
        max_depth = (
            int(data.state_reaction_count.max().item())
            if data.state_reaction_count.numel()
            else -1
        )
        for depth in range(max_depth + 1):
            mask = data.state_reaction_count == depth
            metrics[f"branch_recall_depth_{depth}"] = (
                predicted[mask] & positive[mask]
            ).sum() / positive[mask].sum().clamp_min(1)
            metrics[f"num_training_states_depth_{depth}"] = mask.sum().to(logits.dtype)
        spectrum_output = None
        loss = self.branch_weight * branch_loss
        if self.spectrum_model is not None:
            if data.spectrum_batch is None:
                raise ValueError("Intensity training requires prepared physical-ion tensors")
            spectrum_batch = data.spectrum_batch
            if data.annotations is not None:
                spectrum_batch = replace(
                    spectrum_batch, target_intensity=data.teacher.spectrum_targets(data.annotations)
                )
            spectrum_output = self.spectrum_model(
                spectrum_batch,
                reaction_h=encoded.reaction_h,
                collision_energy_h=encoded.collision_energy_h,
                source_embeddings=encoded.source_embeddings,
                loss_settings=self.spectrum_loss,
            )
            loss = loss + spectrum_output.loss
            metrics.update(spectrum_output.metrics)
        return ReactionTrainingOutput(
            loss, branch_loss, logits, positive, pool, metrics, spectrum_output
        )
