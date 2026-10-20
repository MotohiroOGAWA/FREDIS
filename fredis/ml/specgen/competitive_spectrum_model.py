"""CE-conditioned competing ion transitions and differentiable probability flow.

The structural graph supplies allowed reactions. Every child adduct/shift
explanation competes with all other outgoing explanations and a self transition.
No tree-wide attention or independently regressed intensity is used.
"""

from __future__ import annotations
from dataclasses import replace
import torch
from torch import nn
from ..common.torch_utils.nn_utils import build_fc_layers
from ..common.torch_utils.segment_ops import segment_softmax
from .fragment_spectrum_model import SpectrumModelOutput, SpectrumLossSettings
from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES


def propagate_probability(initial, source, target, probability, steps):
    state = initial
    for _ in range(steps):
        state = state.new_zeros(state.shape).scatter_add_(
            0, target, state[source] * probability
        )
    return state


def merge_candidate_peaks(output):
    """Sum identical physical formulas before output filtering, across all states."""
    keys = torch.cat(
        (
            output.sample_index[:, None].to(output.formula_tensor.dtype),
            output.formula_tensor,
        ),
        1,
    )
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = unique.shape[0]
    first = output.sample_index.new_full((count,), output.sample_index.numel())
    first.scatter_reduce_(
        0,
        inverse,
        torch.arange(inverse.numel(), device=inverse.device),
        reduce="amin",
        include_self=True,
    )
    intensity = output.intensity.new_zeros(count).scatter_add_(
        0, inverse, output.intensity
    )
    probability = output.ion_probability.new_zeros(count).scatter_add_(
        0, inverse, output.ion_probability
    )
    return replace(
        output,
        sample_index=output.sample_index[first],
        formula_tensor=output.formula_tensor[first],
        mz=output.mz[first],
        intensity=intensity,
        confidence=torch.ones_like(intensity),
        ion_probability=probability,
    )


class CompetitiveSpectrumModel(nn.Module):
    """Compete outgoing ion events and propagate precursor probability mass.

    Structural nodes, physical ion candidates and adduct/shift explanations have
    separate index spaces. Helpers keep their conversions explicit while sharing
    the same trainable encoders and scoring MLP.
    """

    requires_reaction_features = True
    prediction_mode = "competitive"

    def __init__(
        self,
        mol_encoder,
        reaction_dim,
        *,
        ion_type_count=1,
        hydrogen_shift_min=0,
        hydrogen_shift_max=0,
        ion_embedding_dim=32,
        hydrogen_shift_embedding_dim=16,
        state_hidden_dim=128,
        main_adduct_dim=128,
        collision_energy_dim=16,
        state_fc_dims=(128,),
        transition_fc_dims=(128, 64),
        transition_dropout=0.0,
        propagation_steps=6,
        descriptors=(),
    ):
        super().__init__()
        if type(propagation_steps) is not int or propagation_steps < 1:
            raise ValueError("propagation_steps must be a positive integer")
        if hydrogen_shift_min > hydrogen_shift_max:
            raise ValueError("Invalid hydrogen shift range")
        self.mol_encoder = mol_encoder
        self.hydrogen_shift_min = hydrogen_shift_min
        self.propagation_steps = propagation_steps
        self.ion_embedding = nn.Embedding(max(1, ion_type_count), ion_embedding_dim)
        self.hydrogen_shift_embedding = nn.Embedding(
            hydrogen_shift_max - hydrogen_shift_min + 1, hydrogen_shift_embedding_dim
        )
        self.state_encoder = build_fc_layers(
            ion_embedding_dim + hydrogen_shift_embedding_dim,
            list(state_fc_dims),
            state_hidden_dim,
            0.0,
        )
        self.state_encoder = nn.Sequential(*(
            layer for layer in self.state_encoder if not isinstance(layer, nn.Dropout)
        ))
        if len(set(descriptors)) != len(descriptors) or set(descriptors) - set(
            SUPPORTED_DESCRIPTOR_NAMES
        ):
            raise ValueError("Invalid transition descriptor selection")
        self.descriptor_indices = tuple(
            SUPPORTED_DESCRIPTOR_NAMES.index(name) for name in descriptors
        )
        self.transition_score = build_fc_layers(
            mol_encoder.graph_dim * 2
            + 2 * len(descriptors)
            + reaction_dim
            + state_hidden_dim * 2
            + main_adduct_dim
            + collision_energy_dim,
            list(transition_fc_dims),
            1,
            transition_dropout,
        )
        self.reaction_dim = reaction_dim

    def forward(
        self,
        data,
        *,
        reaction_h,
        main_adduct_h,
        collision_energy_h,
        source_embeddings=None,
        molecular_embeddings=None,
        loss_settings=None,
    ):
        """Score ion transitions, propagate mass, then compare formula-level peaks."""
        if not getattr(data.decoded, "reaction_state_graph", False):
            raise ValueError(
                "Competitive prediction requires all reaction-state transitions; regenerate prepared data"
            )
        settings = loss_settings or SpectrumLossSettings()
        molecular = self._encode_molecules(
            data, source_embeddings, molecular_embeddings
        )
        device = molecular.device
        candidate_node_index = data.physical_candidate_node_index
        candidate_sample_index = data.physical_candidate_sample_index
        candidate_count = candidate_node_index.numel()
        sample_count = main_adduct_h.shape[0]
        if candidate_count == 0:
            raise ValueError("Competitive model requires physical ion candidates")

        explanation_features, ion_state_features = self._encode_ion_states(
            data, candidate_count, device
        )
        initial_probability = self._initial_probability(
            data, molecular, candidate_node_index, candidate_sample_index, sample_count
        )
        (
            reaction_source_index,
            reaction_target_index,
            explanation_index,
            reaction_index,
        ) = self._enumerate_transitions(
            data, candidate_node_index, candidate_sample_index, sample_count, device
        )
        source, target, probabilities, ids = self._score_transitions(
            molecular,
            candidate_node_index,
            candidate_sample_index,
            ion_state_features,
            explanation_features,
            reaction_h,
            main_adduct_h,
            collision_energy_h,
            reaction_source_index,
            reaction_target_index,
            explanation_index,
            reaction_index,
        )
        final_probability = propagate_probability(
            initial_probability, source, target, probabilities, self.propagation_steps
        )
        formulas = data.physical_candidate_formula_tensor
        normalized, loss, cross_entropy, cosine = self._spectrum_loss(
            data,
            final_probability,
            candidate_sample_index,
            formulas,
            sample_count,
            settings,
        )
        metrics = {
            "intensity_loss": loss,
            "transition_cross_entropy": cross_entropy,
            "full_spectrum_cosine": cosine,
            "num_ion_transitions": final_probability.new_tensor(
                float(reaction_source_index.numel())
            ),
            "num_physical_ion_candidates": final_probability.new_tensor(
                float(candidate_count)
            ),
            "num_ion_explanations": final_probability.new_tensor(
                float(explanation_features.shape[0])
            ),
            "probability_mass_error": (
                final_probability.new_zeros(sample_count).scatter_add_(
                    0, candidate_sample_index, final_probability
                )
                - 1
            )
            .abs()
            .max(),
        }
        # No independent presence classifier: confidence gating is inapplicable.
        return SpectrumModelOutput(
            normalized,
            candidate_sample_index,
            formulas,
            data.physical_candidate_mz,
            loss,
            torch.ones_like(final_probability),
            final_probability,
            metrics,
            transition_source_index=source,
            transition_target_index=target,
            transition_probability=probabilities,
            transition_explanation_index=torch.cat(
                (explanation_index, torch.full_like(ids, -1))
            ),
            transition_reaction_index=torch.cat(
                (reaction_index, torch.full_like(ids, -1))
            ),
        )

    def _encode_molecules(self, data, source_embeddings, molecular_embeddings):
        """Reuse source and duplicate graph embeddings before appending descriptors."""
        if molecular_embeddings is not None:
            molecular = molecular_embeddings
        elif (
            data.node_graph_inverse is not None and data.unique_source_index is not None
        ):
            if source_embeddings is None:
                raise ValueError("Source embeddings required")
            shared = data.unique_source_index >= 0
            unique = source_embeddings.new_zeros(
                (shared.numel(), source_embeddings.shape[1])
            )
            unique[shared] = source_embeddings[data.unique_source_index[shared]]
            if torch.any(~shared):
                unique[~shared] = self.mol_encoder(data.unique_node_graph)[
                    : int((~shared).sum())
                ]
            molecular = unique[data.node_graph_inverse]
        else:
            molecular = self.mol_encoder(data.decoded.node_graph)
        if self.descriptor_indices:
            values = data.decoded.node_graph.descriptors[:, self.descriptor_indices].to(
                molecular.dtype
            )
            molecular = torch.cat(
                (molecular, values.sign() * torch.log1p(values.abs())), 1
            )
        return molecular

    def _encode_ion_states(self, data, candidate_count, device):
        """Keep explanation features distinct and average them for each parent ion."""
        shifts = data.explanation_hydrogen_shift - self.hydrogen_shift_min
        explanation_features = self.state_encoder(
            torch.cat(
                (
                    self.ion_embedding(data.explanation_adduct_index),
                    self.hydrogen_shift_embedding(shifts),
                ),
                1,
            )
        )
        sizes = data.physical_candidate_explanation_ptr.diff()
        owner = torch.repeat_interleave(
            torch.arange(candidate_count, device=device), sizes
        )
        # Equivalent explanations share one physical parent state. Their individual
        # embeddings remain distinct in the outgoing competing transitions below.
        ion_state_features = explanation_features.new_zeros(
            (candidate_count, explanation_features.shape[1])
        ).index_add(0, owner, explanation_features) / sizes[:, None].clamp_min(1)
        return explanation_features, ion_state_features

    def _initial_probability(
        self,
        data,
        molecular,
        candidate_node_index,
        candidate_sample_index,
        sample_count,
    ):
        """Place precursor mass on matching roots, or matching neutral-loss states."""
        candidate_count = candidate_node_index.numel()
        root_node_by_group = {}
        node_group_indices = data.decoded.node_sample_index.detach().cpu().tolist()
        for node, group in enumerate(node_group_indices):
            root_node_by_group.setdefault(group, node)
        candidate_nodes = candidate_node_index.detach().cpu().tolist()
        precursor_formulas = getattr(data, "precursor_formula_tensor", None)
        if precursor_formulas is None or precursor_formulas.shape[0] != sample_count:
            raise ValueError(
                "Competitive model requires precursor ion metadata; regenerate prepared data"
            )
        initial_probability = molecular.new_zeros(candidate_count)
        for sample_index in range(sample_count):
            matches = (
                (
                    (
                        data.physical_candidate_formula_tensor
                        == precursor_formulas[sample_index]
                    ).all(1)
                    & (candidate_sample_index == sample_index)
                )
                .nonzero()
                .flatten()
                .tolist()
            )
            roots = [
                candidate_index
                for candidate_index in matches
                if candidate_nodes[candidate_index]
                == root_node_by_group[
                    node_group_indices[candidate_nodes[candidate_index]]
                ]
            ]
            seeds = roots or matches
            if not seeds:
                raise ValueError(
                    "Observed precursor ion is absent from the candidate graph; check adduct rules"
                )
            # Prefer the original structure. A neutral-loss precursor can map
            # to several structural states; start uniformly over those matches.
            initial_probability[seeds] = 1.0 / len(seeds)
        return initial_probability

    def _enumerate_transitions(
        self, data, candidate_node_index, candidate_sample_index, sample_count, device
    ):
        """Enumerate unique, charge-preserving transitions allowed by elemental counts."""
        candidate_nodes = candidate_node_index.detach().cpu().tolist()
        candidate_samples = candidate_sample_index.detach().cpu().tolist()
        candidates_by_node_and_sample = {}
        for candidate_index, (node_index, sample_index) in enumerate(
            zip(candidate_nodes, candidate_samples)
        ):
            candidates_by_node_and_sample.setdefault(
                (node_index, sample_index), []
            ).append(candidate_index)
        source = []
        target = []
        explanation = []
        reactions = []
        seen = set()
        explanation_ptr = (
            data.physical_candidate_explanation_ptr.detach().cpu().tolist()
        )
        formulas = data.physical_candidate_formula_tensor
        elemental_counts = formulas[:, :-1].detach().cpu().tolist()
        charges = data.physical_candidate_charge.detach().cpu().tolist()
        reaction_ids = data.decoded.edge_added_reaction_index.detach().cpu().tolist()
        # Formula tensors contain elemental counts followed by charge. Require
        # charge retention and a nonnegative neutral loss, in addition to the
        # already validated structural reaction and precursor-specific ion rules.
        for edge, (a, b) in enumerate(
            data.decoded.edge_index.detach().cpu().T.tolist()
        ):
            reaction_index = reaction_ids[edge]
            for sample_index in range(sample_count):
                for parent_index in candidates_by_node_and_sample.get(
                    (a, sample_index), ()
                ):
                    for child_index in candidates_by_node_and_sample.get(
                        (b, sample_index), ()
                    ):
                        if charges[parent_index] != charges[child_index]:
                            continue
                        if any(
                            child > parent
                            for child, parent in zip(
                                elemental_counts[child_index],
                                elemental_counts[parent_index],
                            )
                        ):
                            continue
                        for explanation_index in range(
                            explanation_ptr[child_index],
                            explanation_ptr[child_index + 1],
                        ):
                            event = (
                                parent_index,
                                child_index,
                                explanation_index,
                                reaction_index,
                            )
                            if event in seen:
                                continue
                            seen.add(event)
                            source.append(parent_index)
                            target.append(child_index)
                            explanation.append(explanation_index)
                            reactions.append(reaction_index)
        reaction_source_index = torch.tensor(source, device=device, dtype=torch.long)
        reaction_target_index = torch.tensor(target, device=device, dtype=torch.long)
        explanation_index = torch.tensor(explanation, device=device, dtype=torch.long)
        reaction_index = torch.tensor(reactions, device=device, dtype=torch.long)
        return (
            reaction_source_index,
            reaction_target_index,
            explanation_index,
            reaction_index,
        )

    def _score_transitions(
        self,
        molecular,
        candidate_node_index,
        candidate_sample_index,
        ion_state_features,
        explanation_features,
        reaction_h,
        main_adduct_h,
        collision_energy_h,
        reaction_source_index,
        reaction_target_index,
        explanation_index,
        reaction_index,
    ):
        """Normalize reaction events and persistence events together for each parent."""
        candidate_count = candidate_node_index.numel()
        device = molecular.device
        transition_reaction_features = reaction_h.new_zeros(
            (reaction_source_index.numel(), self.reaction_dim)
        )
        valid = reaction_index >= 0
        transition_reaction_features[valid] = reaction_h[reaction_index[valid]]
        inputs = torch.cat(
            (
                molecular[candidate_node_index[reaction_source_index]],
                molecular[candidate_node_index[reaction_target_index]],
                transition_reaction_features,
                ion_state_features[reaction_source_index],
                explanation_features[explanation_index],
                main_adduct_h[candidate_sample_index[reaction_source_index]],
                collision_energy_h[candidate_sample_index[reaction_source_index]],
            ),
            1,
        )
        break_logits = self.transition_score(inputs).squeeze(-1)
        self_inputs = torch.cat(
            (
                molecular[candidate_node_index],
                molecular[candidate_node_index],
                reaction_h.new_zeros((candidate_count, self.reaction_dim)),
                ion_state_features,
                ion_state_features,
                main_adduct_h[candidate_sample_index],
                collision_energy_h[candidate_sample_index],
            ),
            1,
        )
        self_logits = self.transition_score(self_inputs).squeeze(-1)
        ids = torch.arange(candidate_count, device=device)
        source = torch.cat((reaction_source_index, ids))
        target = torch.cat((reaction_target_index, ids))
        probabilities = segment_softmax(
            torch.cat((break_logits, self_logits)), source, candidate_count
        )
        return source, target, probabilities, ids

    def _spectrum_loss(
        self,
        data,
        final_probability,
        candidate_sample_index,
        formulas,
        sample_count,
        settings,
    ):
        """Merge formula-equivalent probabilities for normalization and supervision."""
        # Compare observed peaks only after summing all structures/paths that
        # yield the same formula in a sample. Keep candidate rows for viewers.
        keys = torch.cat(
            (candidate_sample_index[:, None].to(formulas.dtype), formulas), 1
        )
        _, peak_group_index = torch.unique(keys, dim=0, return_inverse=True)
        peak_count = int(peak_group_index.max()) + 1
        peak_probability = final_probability.new_zeros(peak_count).scatter_add_(
            0, peak_group_index, final_probability
        )
        peak_sample = candidate_sample_index.new_zeros(peak_count).scatter_reduce_(
            0,
            peak_group_index,
            candidate_sample_index,
            reduce="amax",
            include_self=True,
        )
        maxima = final_probability.new_zeros(sample_count).scatter_reduce_(
            0, peak_sample, peak_probability, reduce="amax", include_self=True
        )
        normalized = final_probability / maxima[candidate_sample_index].clamp_min(1e-12)
        loss = final_probability.sum() * 0
        cross_entropy = loss
        cosine = loss
        if data.target_intensity is not None:
            target_peak = final_probability.new_zeros(peak_count).scatter_reduce_(
                0,
                peak_group_index,
                data.target_intensity.clamp_min(0),
                reduce="amax",
                include_self=True,
            )
            target_total = final_probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, target_peak
            )
            has_observed_peaks = target_total > 0
            weights = target_peak / target_total[peak_sample].clamp_min(1e-12)
            cross_entropy = -(
                weights * peak_probability.clamp_min(1e-12).log()
            ).sum() / has_observed_peaks.sum().clamp_min(1)
            dot = final_probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, peak_probability * target_peak
            )
            prediction_squared_norm = final_probability.new_zeros(
                sample_count
            ).scatter_add_(0, peak_sample, peak_probability.square())
            target_squared_norm = final_probability.new_zeros(
                sample_count
            ).scatter_add_(0, peak_sample, target_peak.square())
            cosine = (
                (
                    dot
                    / (prediction_squared_norm * target_squared_norm)
                    .sqrt()
                    .clamp_min(1e-12)
                )[has_observed_peaks].mean()
                if bool(has_observed_peaks.any())
                else loss
            )
            loss = settings.intensity_loss_weight * (
                cross_entropy
                + settings.cosine_loss_weight
                * ((1 - cosine) if bool(has_observed_peaks.any()) else cosine)
            )
        return normalized, loss, cross_entropy, cosine
