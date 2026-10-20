"""CE-conditioned competing ion transitions and differentiable probability flow.

The structural graph supplies allowed reactions. Every child adduct/shift
condition competes with all other outgoing transitions and a self transition.
No tree-wide attention or independently regressed intensity is used.
"""

from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import torch
from torch import nn
from .components.condition.condition_linear import ConditionTransitionHead
from ..common.torch_utils.segment_ops import segment_softmax
from .fragment_spectrum_model import SpectrumModelOutput, SpectrumLossSettings
from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
from fredis.libs.mmkit.mmkit import Adduct


@lru_cache(maxsize=256)
def _adduct_without_hydrogen(adduct_type):
    """Match hydrogen-shift variants while preserving charge and other elements."""
    adduct = Adduct.parse(adduct_type)
    return (
        adduct.n_molecules,
        adduct.charge,
        tuple(
            sorted(
                (element, count)
                for element, count in adduct.element_diff.items()
                if element != "H" and count
            )
        ),
    )


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
        state_node_index = data.state_node_index
        state_sample_index = data.state_sample_index
        state_count = state_node_index.numel()
        sample_count = collision_energy_h.shape[0]
        if state_count == 0:
            raise ValueError("Competitive model requires ion states")

        initial_probability = self._initial_probability(
            data, molecular, state_node_index, state_sample_index, sample_count
        )
        (
            reaction_source_index,
            reaction_target_index,
            reaction_index,
        ) = self._enumerate_transitions(
            data, state_node_index, state_sample_index, sample_count, device
        )
        source, target, probabilities, ids = self._score_transitions(
            data,
            molecular,
            state_node_index,
            state_sample_index,
            reaction_h,
            collision_energy_h,
            reaction_source_index,
            reaction_target_index,
            reaction_index,
        )
        final_probability = propagate_probability(
            initial_probability, source, target, probabilities, self.propagation_steps
        )
        if data.target_intensity is None:
            from .fragment_spectrum_model import annotate_ion_conditions

            data = annotate_ion_conditions(data, self.formula_tensorizer)
        formulas = data.state_formula_tensor
        normalized, loss, loss_metrics = self._spectrum_loss(
            data,
            final_probability,
            state_sample_index,
            formulas,
            sample_count,
            settings,
        )
        metrics = {
            "intensity_loss": loss,
            **loss_metrics,
            "num_ion_transitions": final_probability.new_tensor(
                float(reaction_source_index.numel())
            ),
            "num_ion_states": final_probability.new_tensor(
                float(state_count)
            ),
            "probability_mass_error": (
                final_probability.new_zeros(sample_count).scatter_add_(
                    0, state_sample_index, final_probability
                )
                - 1
            )
            .abs()
            .max(),
        }
        # No independent presence classifier: confidence gating is inapplicable.
        return SpectrumModelOutput(
            normalized,
            state_sample_index,
            formulas,
            data.state_mz,
            loss,
            torch.ones_like(final_probability),
            final_probability,
            metrics,
            transition_source_index=source,
            transition_target_index=target,
            transition_probability=probabilities,
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
        state_node_index,
        state_sample_index,
        sample_count,
    ):
        """Seed source structures by adduct/shift labels, never by formula or m/z."""
        result = molecular.new_zeros(state_node_index.numel())
        groups = data.decoded.node_sample_index.tolist()
        roots = {}
        for node, group in enumerate(groups):
            roots.setdefault(group, node)
        mains = self._sample_main_adducts(data)
        for sample in range(sample_count):
            rows = (state_sample_index == sample).nonzero().flatten()
            if not rows.numel():
                raise ValueError("Every sample requires ion-condition states")
            root = roots[groups[int(state_node_index[rows[0]])]]
            ion = data.ion_type_strs.index(mains[sample])
            seeds = rows[
                (state_node_index[rows] == root)
                & (data.state_adduct_index[rows] == ion)
                & (data.state_hydrogen_shift[rows] == 0)
            ]
            if not seeds.numel():
                raise ValueError(
                    "Ion rules must include main adduct with hydrogen shift 0 at Source"
                )
            result[seeds] = 1.0 / seeds.numel()
        return result

    def _enumerate_transitions(
        self, data, state_node_index, state_sample_index, sample_count, device
    ):
        """Enumerate structural transitions across the complete output-condition grid."""
        state_nodes = state_node_index.detach().cpu().tolist()
        state_samples = state_sample_index.detach().cpu().tolist()
        states_by_node_and_sample = {}
        for state_index, (node_index, sample_index) in enumerate(
            zip(state_nodes, state_samples)
        ):
            states_by_node_and_sample.setdefault(
                (node_index, sample_index), []
            ).append(state_index)
        source = []
        target = []
        reactions = []
        seen = set()
        reaction_ids = data.decoded.edge_added_reaction_index.detach().cpu().tolist()
        for edge, (a, b) in enumerate(
            data.decoded.edge_index.detach().cpu().T.tolist()
        ):
            reaction_index = reaction_ids[edge]
            for sample_index in range(sample_count):
                for parent_index in states_by_node_and_sample.get(
                    (a, sample_index), ()
                ):
                    for child_index in states_by_node_and_sample.get(
                        (b, sample_index), ()
                    ):
                        event = (parent_index, child_index, reaction_index)
                        if event in seen:
                            continue
                        seen.add(event)
                        source.append(parent_index)
                        target.append(child_index)
                        reactions.append(reaction_index)
        reaction_source_index = torch.tensor(source, device=device, dtype=torch.long)
        reaction_target_index = torch.tensor(target, device=device, dtype=torch.long)
        reaction_index = torch.tensor(reactions, device=device, dtype=torch.long)
        return (
            reaction_source_index,
            reaction_target_index,
            reaction_index,
        )

    def _score_transitions(
        self,
        data,
        molecular,
        state_node_index,
        state_sample_index,
        reaction_h,
        collision_energy_h,
        reaction_source_index,
        reaction_target_index,
        reaction_index,
    ):
        count = state_node_index.numel()
        ids = torch.arange(count, device=molecular.device)
        source = torch.cat((reaction_source_index, ids))
        target = torch.cat((reaction_target_index, ids))
        reaction_features = molecular.new_zeros((source.numel(), self.reaction_dim))
        valid = reaction_index >= 0
        reaction_features[: reaction_index.numel()][valid] = reaction_h[
            reaction_index[valid]
        ]
        samples = state_sample_index[source]
        inputs = torch.cat(
            (
                molecular[state_node_index[source]],
                molecular[state_node_index[target]],
                reaction_features,
                collision_energy_h[samples],
            ),
            1,
        )
        mains = self._sample_main_adducts(data)
        condition_values = list(
            zip(
                data.state_adduct_index[target].tolist(),
                data.state_hydrogen_shift[target].tolist(),
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
        state_sample_index,
        formulas,
        sample_count,
        settings,
    ):
        """Merge formula-equivalent probabilities for normalization and supervision."""
        # Compare observed peaks only after summing all structures/paths that
        # yield the same formula in a sample. Keep state rows for viewers.
        keys = torch.cat(
            (state_sample_index[:, None].to(formulas.dtype), formulas), 1
        )
        _, peak_group_index = torch.unique(keys, dim=0, return_inverse=True)
        peak_count = int(peak_group_index.max()) + 1
        peak_probability = final_probability.new_zeros(peak_count).scatter_add_(
            0, peak_group_index, final_probability
        )
        peak_sample = state_sample_index.new_zeros(peak_count).scatter_reduce_(
            0,
            peak_group_index,
            state_sample_index,
            reduce="amax",
            include_self=True,
        )
        valid_candidate = (formulas[:, :-1] >= 0).all(1) & (
            data.state_mz > 0
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
        normalized = final_probability / maxima[state_sample_index].clamp_min(1e-12)
        zero = final_probability.sum() * 0
        metrics = {
            "spectrum_loss_with_precursor": zero,
            "spectrum_loss_without_precursor": zero,
            "transition_cross_entropy": zero,
            "full_spectrum_cosine": zero,
            "transition_cross_entropy_without_precursor": zero,
            "spectrum_cosine_without_precursor": zero,
        }
        if data.target_intensity is None:
            return normalized, zero, metrics
        target_peak = final_probability.new_zeros(peak_count).scatter_reduce_(
            0, peak_group_index, data.target_intensity.clamp_min(0),
            reduce="amax", include_self=True,
        )
        precursor_peak = torch.zeros_like(valid_peak)
        # Identify Source conditions differing from the declared precursor only
        # by hydrogen count, then exclude their entire formula groups.
        roots = {}
        groups = data.decoded.node_sample_index.tolist()
        for node, group in enumerate(groups):
            roots.setdefault(group, node)
        state_nodes = data.state_node_index.tolist()
        state_samples = state_sample_index.tolist()
        precursor_signatures = [
            _adduct_without_hydrogen(adduct) for adduct in data.precursor_adducts
        ]
        precursor_candidate = torch.tensor(
            [
                node == roots[groups[node]]
                and _adduct_without_hydrogen(data.ion_type_strs[adduct_index])
                == precursor_signatures[sample]
                for node, sample, adduct_index in zip(
                    state_nodes, state_samples, data.state_adduct_index.tolist()
                )
            ],
            dtype=torch.bool,
            device=formulas.device,
        )
        precursor_peak.scatter_reduce_(
            0, peak_group_index, precursor_candidate, reduce="amax", include_self=True
        )

        def objective(probability, target):
            target_total = probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, target
            )
            observed = target_total > 0
            if not bool(observed.any()):
                return zero, zero, zero
            weights = target / target_total[peak_sample].clamp_min(1e-12)
            cross_entropy = -(weights * probability.clamp_min(1e-12).log()).sum()
            cross_entropy = cross_entropy / observed.sum()
            dot = probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, probability * target
            )
            pred_norm = probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, probability.square()
            )
            target_norm = probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, target.square()
            )
            cosine = (
                dot / (pred_norm * target_norm).clamp_min(1e-24).sqrt()
            )[observed].mean()
            loss = settings.intensity_loss_weight * (
                cross_entropy + settings.cosine_loss_weight * (1 - cosine)
            )
            return loss, cross_entropy, cosine

        full_loss, cross_entropy, cosine = objective(peak_probability, target_peak)
        fragment_probability = peak_probability.masked_fill(precursor_peak, 0)
        fragment_total = fragment_probability.new_zeros(sample_count).scatter_add_(
            0, peak_sample, fragment_probability
        )
        fragment_probability = (
            fragment_probability / fragment_total[peak_sample].clamp_min(1e-12)
        )
        fragment_loss, fragment_cross_entropy, fragment_cosine = objective(
            fragment_probability, target_peak.masked_fill(precursor_peak, 0)
        )
        metrics.update(
            spectrum_loss_with_precursor=full_loss,
            spectrum_loss_without_precursor=fragment_loss,
            transition_cross_entropy=cross_entropy,
            full_spectrum_cosine=cosine,
            transition_cross_entropy_without_precursor=fragment_cross_entropy,
            spectrum_cosine_without_precursor=fragment_cosine,
        )
        return normalized, full_loss + fragment_loss, metrics
