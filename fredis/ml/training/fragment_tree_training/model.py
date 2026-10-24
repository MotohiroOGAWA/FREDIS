"""CE-independent edge classification and spectrum/formula intensity supervision."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ...specgen.fragment_spectrum_model import (
    SpectrumLossSettings,
    spectrum_batch_from_input,
)


def branch_edge_loss(
    logits, labels, negative_weight=0.5, *, source_index, branch_group_index
):
    """Sum class-balanced losses over (branch group, source Compound) pairs.

    Diagnostic positive/negative losses are sums of the corresponding node means.
    Negative-only terminal nodes are supervised; ignored edges contribute no loss.
    """
    if labels is None:
        raise ValueError("Branch training requires edge labels")
    if logits.ndim != 1 or labels.shape != logits.shape or labels.dtype != torch.int8:
        raise ValueError("Branch logits and int8 edge labels must align")
    for index in (source_index, branch_group_index):
        if index.shape != logits.shape or index.dtype != torch.long:
            raise ValueError(
                "Branch source/group indices must be aligned int64 vectors"
            )
        if index.device != logits.device or bool((index < 0).any()):
            raise ValueError(
                "Branch source/group indices must be nonnegative and share the logits device"
            )
    if not bool(torch.isin(labels, labels.new_tensor([-1, 0, 1])).all()):
        raise ValueError("Unknown branch label")
    if negative_weight < 0 or not torch.isfinite(torch.tensor(negative_weight)):
        raise ValueError("Invalid negative weight")
    zero = logits.sum() * 0
    if not logits.numel():
        return zero, zero, zero
    pairs = torch.stack((branch_group_index, source_index), dim=1)
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    positive = labels == 1
    negative = labels == 0

    def node_means(mask, values):
        count = logits.new_zeros(len(unique)).scatter_add_(
            0, inverse[mask], torch.ones_like(values)
        )
        total = logits.new_zeros(len(unique)).scatter_add_(0, inverse[mask], values)
        return total / count.clamp_min(1), count

    positive_mean, positive_count = node_means(positive, F.softplus(-logits[positive]))
    negative_mean, negative_count = node_means(negative, F.softplus(logits[negative]))
    weight = (positive_count > 0).to(logits.dtype) + negative_weight * (
        negative_count > 0
    ).to(logits.dtype)
    supervised = weight > 0
    divisor = torch.where(supervised, weight, torch.ones_like(weight))
    node_loss = (positive_mean + negative_weight * negative_mean) / divisor
    return (
        node_loss[supervised].sum() + zero,
        positive_mean[supervised].sum() + zero,
        negative_mean[supervised].sum() + zero,
    )


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

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.feature_model, "_freeze_mol_encoder", False):
            self.feature_model.mol_encoder.eval()
        return self

    def forward(self, data):
        if not hasattr(data, "targets") or data.targets is None:
            raise ValueError("Training requires annotated spectrum targets")
        encoded = self.feature_model(data)
        logits, molecular = self.feature_model.score_edges(data, encoded)
        labels = data.reaction_contexts.edge_label
        branch_group_index = torch.repeat_interleave(
            torch.arange(data.num_branch_groups, device=data.device),
            data.reaction_contexts.edge_ptr.diff(),
        )
        source_index = data.compound_graph.source_compound_index[
            data.reaction_contexts.edge_index
        ]
        branch, positive, negative = branch_edge_loss(
            logits,
            labels,
            self.negative_weight,
            source_index=source_index,
            branch_group_index=branch_group_index,
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
        predicted = logits.detach() >= 0
        for name, mask in {
            "tp": predicted & (labels == 1),
            "tn": ~predicted & (labels == 0),
            "fp": predicted & (labels == 0),
            "fn": ~predicted & (labels == 1),
        }.items():
            metrics["branch_count_" + name] = mask.sum().float()
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
            branch_group_index,
        )
