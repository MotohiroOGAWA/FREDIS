"""CE-conditioned competing ion transitions and differentiable probability flow.

The structural graph supplies allowed reactions. Every child adduct/shift
explanation competes with all other outgoing explanations and a self transition.
No tree-wide attention or independently regressed intensity is used.
"""

from __future__ import annotations
from dataclasses import replace
import torch
from torch import nn
from .components.condition.condition_linear import ConditionTransitionHead
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
    valid = (output.formula_tensor[:, :-1] >= 0).all(1) & (output.mz > 0)
    output = replace(
        output,
        intensity=output.intensity[valid],
        sample_index=output.sample_index[valid],
        formula_tensor=output.formula_tensor[valid],
        mz=output.mz[valid],
        ion_probability=output.ion_probability[valid],
    )
    keys = keys[valid]
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
    """Compete structural ion-condition events with main-adduct-specific heads.

    State and edge construction is independent of formula/mz annotations.
    """

    requires_reaction_features = True
    prediction_mode = "competitive"

    def __init__(
        self,
        mol_encoder,
        reaction_dim,
        *,
        main_adduct_conditions,
        precursor_main_adducts,
        formula_tensorizer,
        collision_energy_dim=16,
        transition_fc_dims=(128, 64),
        transition_dropout=0.0,
        propagation_steps=6,
        descriptors=(),
    ):
        super().__init__()
        if type(propagation_steps) is not int or propagation_steps < 1:
            raise ValueError("propagation_steps must be a positive integer")
        self.formula_tensorizer = formula_tensorizer
        self.mol_encoder = mol_encoder
        self.propagation_steps = propagation_steps
        self.main_adduct_conditions = {
            key: tuple(values) for key, values in main_adduct_conditions.items()
        }
        self.precursor_main_adducts = dict(precursor_main_adducts)
        self.head_index = {
            key: str(i) for i, key in enumerate(self.main_adduct_conditions)
        }
        if len(set(descriptors)) != len(descriptors) or set(descriptors) - set(
            SUPPORTED_DESCRIPTOR_NAMES
        ):
            raise ValueError("Invalid transition descriptor selection")
        self.descriptor_indices = tuple(
            SUPPORTED_DESCRIPTOR_NAMES.index(name) for name in descriptors
        )
        input_dim = (
            2 * (mol_encoder.graph_dim + len(descriptors))
            + reaction_dim
            + collision_energy_dim
        )
        self.transition_heads = nn.ModuleDict(
            {
                self.head_index[main]: ConditionTransitionHead(
                    len(conditions),
                    input_dim,
                    tuple(transition_fc_dims),
                    transition_dropout,
                )
                for main, conditions in self.main_adduct_conditions.items()
            }
        )
        self.reaction_dim = reaction_dim

    def forward(
        self,
        data,
        *,
        reaction_h,
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
        sample_count = collision_energy_h.shape[0]
        if candidate_count == 0:
            raise ValueError("Competitive model requires physical ion candidates")

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
            data,
            molecular,
            candidate_node_index,
            candidate_sample_index,
            reaction_h,
            collision_energy_h,
            reaction_source_index,
            reaction_target_index,
            explanation_index,
            reaction_index,
        )
        final_probability = propagate_probability(
            initial_probability, source, target, probabilities, self.propagation_steps
        )
        if data.target_intensity is None:
            from .fragment_spectrum_model import annotate_ion_conditions

            data = annotate_ion_conditions(data, self.formula_tensorizer)
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
                float(data.explanation_adduct_index.numel())
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

    def _sample_main_adducts(self, data):
        return [self.precursor_main_adducts[value] for value in data.precursor_adducts]

    def _initial_probability(
        self,
        data,
        molecular,
        candidate_node_index,
        candidate_sample_index,
        sample_count,
    ):
        """Seed source structures by adduct/shift labels, never by formula or m/z."""
        result = molecular.new_zeros(candidate_node_index.numel())
        groups = data.decoded.node_sample_index.tolist()
        roots = {}
        for node, group in enumerate(groups):
            roots.setdefault(group, node)
        starts = data.physical_candidate_explanation_ptr[:-1]
        mains = self._sample_main_adducts(data)
        for sample in range(sample_count):
            rows = (candidate_sample_index == sample).nonzero().flatten()
            if not rows.numel():
                raise ValueError("Every sample requires ion-condition states")
            root = roots[groups[int(candidate_node_index[rows[0]])]]
            ion = data.ion_type_strs.index(mains[sample])
            seeds = rows[
                (candidate_node_index[rows] == root)
                & (data.explanation_adduct_index[starts[rows]] == ion)
                & (data.explanation_hydrogen_shift[starts[rows]] == 0)
            ]
            if not seeds.numel():
                raise ValueError(
                    "Ion rules must include main adduct with hydrogen shift 0 at Source"
                )
            result[seeds] = 1.0 / seeds.numel()
        return result

    def _enumerate_transitions(
        self, data, candidate_node_index, candidate_sample_index, sample_count, device
    ):
        """Enumerate structural transitions across the complete output-condition grid."""
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
        reaction_ids = data.decoded.edge_added_reaction_index.detach().cpu().tolist()
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
        data,
        molecular,
        candidate_node_index,
        candidate_sample_index,
        reaction_h,
        collision_energy_h,
        reaction_source_index,
        reaction_target_index,
        explanation_index,
        reaction_index,
    ):
        count = candidate_node_index.numel()
        ids = torch.arange(count, device=molecular.device)
        source = torch.cat((reaction_source_index, ids))
        target = torch.cat((reaction_target_index, ids))
        explanations = torch.cat(
            (explanation_index, data.physical_candidate_explanation_ptr[:-1])
        )
        reaction_features = molecular.new_zeros((source.numel(), self.reaction_dim))
        valid = reaction_index >= 0
        reaction_features[: reaction_index.numel()][valid] = reaction_h[
            reaction_index[valid]
        ]
        samples = candidate_sample_index[source]
        inputs = torch.cat(
            (
                molecular[candidate_node_index[source]],
                molecular[candidate_node_index[target]],
                reaction_features,
                collision_energy_h[samples],
            ),
            1,
        )
        mains = self._sample_main_adducts(data)
        condition_values = list(
            zip(
                data.explanation_adduct_index[explanations].tolist(),
                data.explanation_hydrogen_shift[explanations].tolist(),
            )
        )
        sample_values = samples.tolist()
        logits = molecular.new_zeros(source.numel())
        for main, conditions in self.main_adduct_conditions.items():
            rows = [
                i for i, sample in enumerate(sample_values) if mains[sample] == main
            ]
            if not rows:
                continue
            lookup = {value: i for i, value in enumerate(conditions)}
            condition = torch.tensor(
                [
                    lookup[
                        (
                            data.ion_type_strs[condition_values[i][0]],
                            condition_values[i][1],
                        )
                    ]
                    for i in rows
                ],
                device=molecular.device,
            )
            index = torch.tensor(rows, device=molecular.device)
            logits[index] = self.transition_heads[self.head_index[main]](
                inputs[index], condition
            )
        return source, target, segment_softmax(logits, source, count), ids

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
        valid_candidate = (formulas[:, :-1] >= 0).all(1) & (
            data.physical_candidate_mz > 0
        )
        valid_peak = torch.zeros(peak_count, dtype=torch.bool, device=formulas.device)
        valid_peak.scatter_reduce_(
            0, peak_group_index, valid_candidate, reduce="amax", include_self=True
        )
        maxima = final_probability.new_zeros(sample_count).scatter_reduce_(
            0,
            peak_sample,
            peak_probability * valid_peak,
            reduce="amax",
            include_self=True,
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
