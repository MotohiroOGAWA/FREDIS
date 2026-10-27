"""CE-independent edge classification and spectrum/formula intensity supervision."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .branch_mil import branch_path_mil_loss
from ...specgen.fragment_spectrum_model import SpectrumLossSettings


@dataclass(frozen=True)
class ReactionTrainingOutput:
    loss: Tensor
    branch_loss: Tensor
    branch_logits: Tensor
    branch_positive_mask: Tensor
    metrics: dict[str, Tensor]
    spectrum_output: object | None = None
    edge_index: Tensor | None = None
    branch_group_index: Tensor | None = None

    @property
    def branch_log_probs(self):
        return F.logsigmoid(self.branch_logits)


class ReactionFragmentTreeTrainingModel(nn.Module):
    def __init__(
        self,
        feature_model,
        negative_weight=0.5,
        spectrum_model=None,
        branch_weight=1.0,
        spectrum_loss=None,
        branch_mil=None,
    ):
        super().__init__()
        self.feature_model = feature_model
        self.spectrum_model = spectrum_model
        self.negative_weight = negative_weight
        self.branch_weight = branch_weight
        self.branch_mil = dict(branch_mil or {})
        self.negative_weight = self.branch_mil.get("negative_weight", negative_weight)
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
        """Return joint branch/spectrum loss, logits, predictions and training metrics."""
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
        branch, positive, negative = branch_path_mil_loss(
            self.feature_model.score_execution(data, encoded), data,
            self.negative_weight, self.branch_mil.get("temperature", .1),
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
            batch = data.spectrum_batch
            spectrum = self.spectrum_model(
                batch,
                reaction_h=encoded.local_reaction_h,
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
            metrics,
            spectrum,
            data.reaction_contexts.edge_index,
            branch_group_index,
        )
