"""CE-independent edge classification and spectrum/formula intensity supervision."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ...specgen.fragment_spectrum_model import (
    SpectrumLossSettings,
    spectrum_batch_from_input,
)


def branch_edge_loss(logits, labels, negative_weight=0.5):
    if labels is None:
        raise ValueError("Branch training requires edge labels")
    if labels.shape != logits.shape or labels.dtype != torch.int8:
        raise ValueError("Branch logits and int8 edge labels must align")
    if not bool(torch.isin(labels, labels.new_tensor([-1, 0, 1])).all()):
        raise ValueError("Unknown branch label")
    if negative_weight < 0 or not torch.isfinite(torch.tensor(negative_weight)):
        raise ValueError("Invalid negative weight")
    positive = labels == 1
    negative = labels == 0
    zero = logits.sum() * 0
    pl = F.softplus(-logits[positive])
    nl = F.softplus(logits[negative])
    weight = positive.sum() + negative_weight * negative.sum()
    loss = (pl.sum() + negative_weight * nl.sum()) / weight.clamp_min(1e-12)
    return loss, pl.mean() if pl.numel() else zero, nl.mean() if nl.numel() else zero


@dataclass(frozen=True)
class ReactionTrainingOutput:
    loss: Tensor
    branch_loss: Tensor
    branch_logits: Tensor
    branch_positive_mask: Tensor
    pool: object
    metrics: dict[str, Tensor]
    spectrum_output: object | None = None
    edge_index: Tensor | None = None
    branch_group_index: Tensor | None = None


class ReactionFragmentTreeTrainingModel(nn.Module):
    def __init__(
        self,
        feature_model,
        negative_weight=0.5,
        spectrum_model=None,
        branch_weight=1.0,
        spectrum_loss=None,
    ):
        super().__init__()
        self.feature_model = feature_model
        self.spectrum_model = spectrum_model
        self.negative_weight = negative_weight
        self.branch_weight = branch_weight
        self.spectrum_loss = spectrum_loss or SpectrumLossSettings()

    def set_checkpoint_model_config(self, config):
        self._checkpoint_model_config = dict(config)

    def get_params(self):
        return dict(getattr(self, "_checkpoint_model_config", {}))

    def forward(self, data):
        if not hasattr(data, "targets") or data.targets is None:
            raise ValueError("Training requires annotated spectrum targets")
        encoded = self.feature_model(data)
        logits, molecular = self.feature_model.score_edges(data, encoded)
        labels = data.reaction_contexts.edge_label
        branch, positive, negative = branch_edge_loss(
            logits, labels, self.negative_weight
        )
        metrics = dict(
            branch_loss=branch,
            branch_positive_loss=positive,
            branch_negative_loss=negative,
            num_branch_groups=logits.new_tensor(float(data.num_branch_groups)),
            num_labelled_edges=(labels >= 0).sum().float(),
            num_positive_edges=(labels == 1).sum().float(),
            num_weak_negative_edges=(labels == 0).sum().float(),
        )
        spectrum = None
        loss = self.branch_weight * branch
        if self.spectrum_model is not None:
            batch = spectrum_batch_from_input(data, data.targets)
            spectrum = self.spectrum_model(
                batch,
                reaction_h=encoded.reaction_h,
                collision_energy_h=encoded.collision_energy_h,
                source_embeddings=encoded.source_embeddings,
                molecular_embeddings=molecular,
                loss_settings=self.spectrum_loss,
            )
            loss = loss + spectrum.loss
            metrics.update(spectrum.metrics)
        return ReactionTrainingOutput(
            loss,
            branch,
            logits,
            labels == 1,
            encoded.pool,
            metrics,
            spectrum,
            data.reaction_contexts.edge_index,
            torch.repeat_interleave(
                torch.arange(data.num_branch_groups, device=data.device),
                data.reaction_contexts.edge_ptr.diff(),
            ),
        )
