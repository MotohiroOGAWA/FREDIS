"""Fragment DAG survival followed by conditional ion/shift state probabilities."""

from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import torch
from torch import nn
from .components.condition.condition_linear import ConditionHead
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


def propagate_probability(initial, source, target, probability):
    """Process each DAG state once; self absorbs, target -1 denotes terminal sink.

    Kahn frontiers ensure all parents contribute before a state is processed.
    Only topology is inspected on CPU; probability operations retain autograd.
    Returns final retained mass and sink mass, both indexed by state.
    """
    count = initial.numel()
    sources = source.detach().cpu().tolist()
    targets = target.detach().cpu().tolist()
    outgoing = [[] for _ in range(count)]
    children = [[] for _ in range(count)]
    indegree = [0] * count
    for event, (parent, child) in enumerate(zip(sources, targets)):
        if not 0 <= parent < count or not -1 <= child < count:
            raise ValueError("Invalid probability event endpoint")
        outgoing[parent].append(event)
        if child >= 0 and child != parent:
            children[parent].append(child)
            indegree[child] += 1
    frontier = [state for state, degree in enumerate(indegree) if degree == 0]
    levels = []
    visited = 0
    while frontier:
        levels.append([event for state in frontier for event in outgoing[state]])
        visited += len(frontier)
        following = []
        for parent in frontier:
            for child in children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    following.append(child)
        frontier = following
    if visited != count:
        raise ValueError("Fragment probability graph must be a DAG")

    incoming = initial
    final = initial * 0
    sink = initial * 0
    for events in levels:
        index = torch.tensor(events, dtype=torch.long, device=source.device)
        parents, children_index = source[index], target[index]
        mass = incoming[parents] * probability[index]
        self_event = children_index == parents
        sink_event = children_index == -1
        break_event = ~(self_event | sink_event)
        final = final.index_add(0, parents[self_event], mass[self_event])
        sink = sink.index_add(0, parents[sink_event], mass[sink_event])
        incoming = incoming.index_add(
            0, children_index[break_event], mass[break_event]
        )
    return final, sink


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
        state_probability=None,
        state_fragment_index=None,
    )


class CompetitiveSpectrumModel(nn.Module):
    """Separate fragment events and conditional ion states per main adduct.

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
        fragment_fc_dims=(128, 64),
        fragment_dropout=0.0,
        state_fc_dims=(128, 64),
        state_dropout=0.0,
        descriptors=(),
    ):
        super().__init__()
        self.formula_tensorizer = formula_tensorizer
        self.mol_encoder = mol_encoder
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
        self.fragment_heads = nn.ModuleDict({
            self.head_index[main]: ConditionHead(
                3, input_dim, tuple(fragment_fc_dims), fragment_dropout
            )
            for main in self.main_adduct_conditions
        })
        state_input_dim = 2 * (mol_encoder.graph_dim + len(descriptors)) + collision_energy_dim
        self.state_heads = nn.ModuleDict({
            self.head_index[main]: ConditionHead(
                len(conditions), state_input_dim, tuple(state_fc_dims), state_dropout
            )
            for main, conditions in self.main_adduct_conditions.items()
        })
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
        """Propagate fragment mass, distribute into ion states, then supervise peaks."""
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

        # A fragment row is (sample, structural node), independent of ion states.
        fragments, state_fragment_index = torch.unique(
            torch.stack((state_sample_index, state_node_index), dim=1),
            dim=0, return_inverse=True,
        )
        fragment_samples, fragment_nodes = fragments.unbind(1)
        initial, source, target, event_reactions, event_type, root_index = self._fragment_events(
            data, molecular, fragments, sample_count
        )
        probabilities = self._score_fragment_events(
            data, molecular, collision_energy_h, reaction_h,
            fragment_nodes, fragment_samples, source, target, event_reactions, event_type,
        )
        fragment_probability, fragment_sink = propagate_probability(
            initial, source, target, probabilities
        )
        # State Head is evaluated after fragment propagation. It has no path
        # back into Fragment Head logits or DAG event probabilities.
        state_probability = self._score_states(
            data, molecular, collision_energy_h, fragments, state_fragment_index, root_index
        )
        final_probability = fragment_probability[state_fragment_index] * state_probability
        sink_probability = molecular.new_zeros(sample_count).scatter_add_(
            0, fragment_samples, fragment_sink
        )
        retained_probability = molecular.new_zeros(sample_count).scatter_add_(
            0, fragment_samples, fragment_probability
        )
        initial_mass = molecular.new_zeros(sample_count).scatter_add_(0, fragment_samples, initial)
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
            "num_fragment_transitions": final_probability.new_tensor(
                float((event_type == 0).sum())
            ),
            "num_ion_states": final_probability.new_tensor(
                float(state_count)
            ),
            "sink_probability": sink_probability.mean(),
            "retained_probability": retained_probability.mean(),
            "probability_mass_error": (
                retained_probability + sink_probability - initial_mass
            ).abs().max(),
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
            transition_reaction_index=event_reactions,
            sink_probability=sink_probability,
            fragment_probability=fragment_probability,
            fragment_node_index=fragment_nodes,
            fragment_sample_index=fragment_samples,
            state_fragment_index=state_fragment_index,
            state_probability=state_probability,
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

    def _fragment_events(self, data, molecular, fragments, sample_count):
        """Build Child/Self/Sink events on sample-specific Fragment DAGs."""
        keys = fragments.detach().cpu().tolist()
        lookup = {(sample, node): i for i, (sample, node) in enumerate(keys)}
        groups = data.decoded.node_sample_index.detach().cpu().tolist()
        roots = {}
        for node, group in enumerate(groups):
            roots.setdefault(group, node)
        root_rows = [lookup[(sample, roots[groups[node]])] for sample, node in keys]
        root_index = torch.tensor(root_rows, dtype=torch.long, device=molecular.device)
        root_mask = [i == root for i, root in enumerate(root_rows)]
        if sum(root_mask) != sample_count:
            raise ValueError("Each spectrum sample requires exactly one fragment root")
        initial = molecular.new_tensor(root_mask)
        edges = data.decoded.edge_index.detach().cpu().T.tolist()
        reactions = data.decoded.edge_added_reaction_index.detach().cpu().tolist()
        events = set()
        for sample in range(sample_count):
            for (parent, child), reaction in zip(edges, reactions):
                parent_row, child_row = lookup.get((sample, parent)), lookup.get((sample, child))
                if parent_row is not None and child_row is not None:
                    if parent_row == child_row:
                        raise ValueError("Break events must connect distinct fragments")
                    events.add((parent_row, child_row, reaction, 0))
        rows = sorted(events)
        rows.extend((i, i, -1, 1) for i in range(len(keys)))
        rows.extend((i, -1, -2, 2) for i in range(len(keys)))
        tensors = torch.tensor(rows, dtype=torch.long, device=molecular.device)
        return initial, *tensors.unbind(1), root_index

    def _score_fragment_events(
        self, data, molecular, collision_energy_h, reaction_h,
        fragment_nodes, fragment_samples, source, target, reactions, event_type,
    ):
        reaction_features = molecular.new_zeros((source.numel(), self.reaction_dim))
        breaks = reactions >= 0
        reaction_features[breaks] = reaction_h[reactions[breaks]]
        # Self/Sink use the parent's structure twice and zero reaction features.
        child = torch.where(target >= 0, target, source)
        samples = fragment_samples[source]
        inputs = torch.cat((
            molecular[fragment_nodes[source]], molecular[fragment_nodes[child]],
            reaction_features, collision_energy_h[samples],
        ), dim=1)
        mains = self._sample_main_adducts(data)
        sample_values = samples.tolist()
        logits = molecular.new_zeros(source.numel())
        for main in self.main_adduct_conditions:
            rows = torch.tensor(
                [i for i, sample in enumerate(sample_values) if mains[sample] == main],
                dtype=torch.long, device=molecular.device,
            )
            if rows.numel():
                logits[rows] = self.fragment_heads[self.head_index[main]](inputs[rows], event_type[rows])
        return segment_softmax(logits, source, fragment_nodes.numel())

    def _score_states(self, data, molecular, collision_energy_h, fragments, inverse, roots):
        """Shared across fragments; normalize ion/shift states within each fragment."""
        samples = data.state_sample_index
        nodes = data.state_node_index
        root_nodes = fragments[roots[inverse], 1]
        inputs = torch.cat((molecular[nodes], molecular[root_nodes], collision_energy_h[samples]), dim=1)
        mains = self._sample_main_adducts(data)
        sample_values = samples.tolist()
        adducts = data.state_adduct_index.tolist()
        shifts = data.state_hydrogen_shift.tolist()
        logits = molecular.new_zeros(nodes.numel())
        for main, conditions in self.main_adduct_conditions.items():
            rows = [i for i, sample in enumerate(sample_values) if mains[sample] == main]
            if not rows:
                continue
            lookup = {value: i for i, value in enumerate(conditions)}
            condition = torch.tensor([
                lookup[(data.ion_type_strs[adducts[i]], shifts[i])] for i in rows
            ], dtype=torch.long, device=molecular.device)
            index = torch.tensor(rows, dtype=torch.long, device=molecular.device)
            logits[index] = self.state_heads[self.head_index[main]](inputs[index], condition)
        return segment_softmax(logits, inverse, fragments.shape[0])

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
