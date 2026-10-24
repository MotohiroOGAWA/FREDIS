"""Fragment reaction-path survival followed by conditional ion/shift state probabilities."""

from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import torch
from torch import nn
from .components.condition.condition_linear import ConditionHead
from .fragment_spectrum_model import SpectrumModelOutput, SpectrumLossSettings
from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
from ..mol.selected_descriptors import selected_descriptor_values
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


def segment_logsumexp(values, groups, size):
    """Sum grouped probabilities in log space, including empty/unreachable groups."""
    maximum = values.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, groups, values.detach(), reduce="amax", include_self=True)
    shift = maximum.masked_fill(~torch.isfinite(maximum), 0)
    weights = (values - shift[groups]).exp()
    total = values.new_zeros(size).scatter_add_(0, groups, weights)
    # Avoid undefined gradients from log(0) for unreachable groups.
    return (shift + total.clamp_min(torch.finfo(values.dtype).tiny).log()).masked_fill(
        total == 0, -torch.inf
    )


def segment_log_softmax(logits, groups, size):
    return logits - segment_logsumexp(logits, groups, size)[groups]


def probability_log(probability):
    return probability.clamp_min(torch.finfo(probability.dtype).tiny).log().masked_fill(
        probability == 0, -torch.inf
    )


def propagate_log_probability(initial, source, target, probability):
    """Process each reaction state once; self absorbs, target -1 denotes terminal sink.

    Kahn frontiers ensure all parents contribute before a state is processed.
    Only topology is inspected on CPU; probability operations retain autograd.
    Inputs and outputs are log probabilities; outputs are indexed by state.
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
        raise ValueError("Fragment reaction paths must not contain cycles")

    incoming = initial
    final = initial.new_full(initial.shape, -torch.inf)
    sink = initial.new_full(initial.shape, -torch.inf)
    states = torch.arange(count, device=source.device)
    for events in levels:
        index = torch.tensor(events, dtype=torch.long, device=source.device)
        parents, children_index = source[index], target[index]
        mass = incoming[parents] + probability[index]
        self_event = children_index == parents
        sink_event = children_index == -1
        break_event = ~(self_event | sink_event)
        def accumulate(existing, destinations, values):
            return segment_logsumexp(
                torch.cat((existing, values)), torch.cat((states, destinations)), count
            )
        final = accumulate(final, parents[self_event], mass[self_event])
        sink = accumulate(sink, parents[sink_event], mass[sink_event])
        incoming = accumulate(incoming, children_index[break_event], mass[break_event])
    return final, sink


def propagate_probability(initial, source, target, probability):
    """Compatibility wrapper returning ordinary retained and sink probabilities."""
    final, sink = propagate_log_probability(
        probability_log(initial), source, target, probability_log(probability)
    )
    return final.exp(), sink.exp()


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
        formula_explanation_index=None,
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
        state_fc_dims=(128, 64),
        dropout=0.0,
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
        self.descriptor_names = tuple(descriptors)
        input_dim = (
            2 * (mol_encoder.graph_dim + len(descriptors))
            + reaction_dim
            + collision_energy_dim
        )
        self.fragment_heads = nn.ModuleDict(
            {
                self.head_index[main]: ConditionHead(
                    3, input_dim, tuple(fragment_fc_dims), dropout
                )
                for main in self.main_adduct_conditions
            }
        )
        state_input_dim = (
            2 * (mol_encoder.graph_dim + len(descriptors)) + collision_energy_dim
        )
        self.state_heads = nn.ModuleDict(
            {
                self.head_index[main]: ConditionHead(
                    len(conditions),
                    state_input_dim,
                    tuple(state_fc_dims),
                    dropout,
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
        """Propagate fragment mass, distribute into ion states, then supervise peaks."""
        if getattr(data, "canonical_data", None) is not None:
            return self._forward_canonical(
                data,
                reaction_h,
                collision_energy_h,
                source_embeddings,
                molecular_embeddings,
                loss_settings,
            )
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
            dim=0,
            return_inverse=True,
        )
        fragment_samples, fragment_nodes = fragments.unbind(1)
        initial, source, target, event_reactions, event_type, root_index = (
            self._fragment_events(data, molecular, fragments, sample_count)
        )
        probabilities = self._score_fragment_events(
            data,
            molecular,
            collision_energy_h,
            reaction_h,
            fragment_nodes,
            fragment_samples,
            source,
            target,
            event_reactions,
            event_type,
            log=True,
        )
        log_fragment, log_sink = propagate_log_probability(
            probability_log(initial), source, target, probabilities
        )
        fragment_probability, fragment_sink = log_fragment.exp(), log_sink.exp()
        probabilities = probabilities.exp()
        # State Head is evaluated after fragment propagation. It has no path
        # back into Fragment Head logits or reaction-event probabilities.
        state_probability = self._score_states(
            data,
            molecular,
            collision_energy_h,
            fragments,
            state_fragment_index,
            root_index,
            log=True,
        )
        final_log_probability = log_fragment[state_fragment_index] + state_probability
        state_probability = state_probability.exp()
        final_probability = final_log_probability.exp()
        sink_probability = molecular.new_zeros(sample_count).scatter_add_(
            0, fragment_samples, fragment_sink
        )
        retained_probability = molecular.new_zeros(sample_count).scatter_add_(
            0, fragment_samples, fragment_probability
        )
        initial_mass = molecular.new_zeros(sample_count).scatter_add_(
            0, fragment_samples, initial
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
            final_log_probability=final_log_probability,
        )
        metrics = {
            "intensity_loss": loss,
            **loss_metrics,
            "num_fragment_transitions": final_probability.new_tensor(
                float((event_type == 0).sum())
            ),
            "num_ion_states": final_probability.new_tensor(float(state_count)),
            "sink_probability": sink_probability.mean(),
            "retained_probability": retained_probability.mean(),
            "probability_mass_error": (
                retained_probability + sink_probability - initial_mass
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
            transition_reaction_index=event_reactions,
            sink_probability=sink_probability,
            fragment_probability=fragment_probability,
            fragment_node_index=fragment_nodes,
            fragment_sample_index=fragment_samples,
            state_fragment_index=state_fragment_index,
            state_probability=state_probability,
        )

    def _forward_canonical(
        self,
        data,
        reaction_h,
        collision_energy_h,
        source_embeddings,
        molecular_embeddings,
        loss_settings,
    ):
        """Propagate finite reaction histories, then merge by canonical Compound IDs."""
        core = data.canonical_data
        execution = core.execution_graph
        molecular = self._encode_molecules(
            data, source_embeddings, molecular_embeddings
        )
        device = molecular.device
        execution_compounds = []
        execution_samples = []
        root_rows = []
        source = []
        target = []
        reactions = []
        event_type = []
        physical = []
        physical_lookup = {}
        execution_to_physical = []
        physical_roots = []
        for sample, group in enumerate(core.sample_branch_group_index.tolist()):
            ids = (execution.branch_group_index == group).nonzero().flatten().tolist()
            mapping = {old: len(execution_compounds) + i for i, old in enumerate(ids)}
            root_compound = int(execution.compound_index[execution.root_index[group]])
            compounds = sorted(set(execution.compound_index[ids].tolist()))
            for compound in compounds:
                physical_lookup[(sample, compound)] = len(physical)
                physical.append((sample, compound))
            root_physical = physical_lookup[(sample, root_compound)]
            physical_roots.extend([root_physical] * len(compounds))
            root_rows.append(mapping[int(execution.root_index[group])])
            for old in ids:
                compound = int(execution.compound_index[old])
                execution_compounds.append(compound)
                execution_samples.append(sample)
                execution_to_physical.append(physical_lookup[(sample, compound)])
            for parent, child, reaction in zip(
                execution.source_index.tolist(),
                execution.target_index.tolist(),
                execution.reaction_index.tolist(),
            ):
                if parent in mapping:
                    source.append(mapping[parent])
                    target.append(mapping[child])
                    reactions.append(reaction)
                    event_type.append(0)
        for i in range(len(execution_compounds)):
            source.extend((i, i))
            target.extend((i, -1))
            reactions.extend((-1, -2))
            event_type.extend((1, 2))

        def ids(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        source, target, reactions, event_type = map(
            ids, (source, target, reactions, event_type)
        )
        execution_compounds = ids(execution_compounds)
        execution_samples = ids(execution_samples)
        probability = self._score_fragment_events(
            data,
            molecular,
            collision_energy_h,
            reaction_h,
            execution_compounds,
            execution_samples,
            source,
            target,
            reactions,
            event_type,
            log=True,
        )
        initial = molecular.new_zeros(execution_compounds.numel())
        initial[ids(root_rows)] = 1
        log_retained, log_sink = propagate_log_probability(
            probability_log(initial), source, target, probability
        )
        retained, sink = log_retained.exp(), log_sink.exp()
        log_event_probability = probability
        probability = probability.exp()
        execution_to_physical = ids(execution_to_physical)
        log_fragment = segment_logsumexp(log_retained, execution_to_physical, len(physical))
        fragment_probability = log_fragment.exp()
        fragments = ids(physical).reshape(-1, 2)
        state_fragment = ids(
            [
                physical_lookup[(sample, node)]
                for sample, node in zip(
                    data.state_sample_index.tolist(), data.state_node_index.tolist()
                )
            ]
        )
        conditional = self._score_states(
            data,
            molecular,
            collision_energy_h,
            fragments,
            state_fragment,
            ids(physical_roots),
            log=True,
        )
        final_log_probability = log_fragment[state_fragment] + conditional
        conditional = conditional.exp()
        final = final_log_probability.exp()
        normalized, loss, metrics = self._spectrum_loss(
            data,
            final,
            data.state_sample_index,
            data.state_formula_tensor,
            core.num_samples,
            loss_settings or SpectrumLossSettings(),
            final_log_probability=final_log_probability,
        )
        sink_probability = molecular.new_zeros(core.num_samples).scatter_add_(
            0, execution_samples, sink
        )
        retained_probability = molecular.new_zeros(core.num_samples).scatter_add_(
            0, fragments[:, 0], fragment_probability
        )
        metrics.update(
            intensity_loss=loss,
            num_fragment_transitions=final.new_tensor(float((event_type == 0).sum())),
            num_ion_states=final.new_tensor(float(final.numel())),
            sink_probability=sink_probability.mean(),
            retained_probability=retained_probability.mean(),
            probability_mass_error=(sink_probability + retained_probability - 1)
            .abs()
            .max(),
        )
        child_physical = torch.where(
            target >= 0, execution_to_physical[target.clamp_min(0)], target
        )
        # Report one effective event per physical Compound/reaction, while the
        # actual propagation above remains history constrained and acyclic.
        log_incoming = log_retained - log_event_probability[event_type == 1]
        log_event_mass = log_incoming[source] + log_event_probability
        diagnostic_keys = torch.stack(
            (execution_to_physical[source], child_physical, reactions), 1
        )
        diagnostic_rows, diagnostic_inverse = torch.unique(
            diagnostic_keys, dim=0, return_inverse=True
        )
        aggregate_mass = segment_logsumexp(
            log_event_mass, diagnostic_inverse, len(diagnostic_rows)
        )
        totals = segment_logsumexp(
            log_event_mass, execution_to_physical[source], len(physical)
        )
        normalizer = totals[diagnostic_rows[:, 0]]
        diagnostic_probability = (aggregate_mass - normalizer.masked_fill(
            ~torch.isfinite(normalizer), 0
        )).exp()
        return SpectrumModelOutput(
            normalized,
            data.state_sample_index,
            data.state_formula_tensor,
            data.state_mz,
            loss,
            torch.ones_like(final),
            final,
            metrics,
            transition_source_index=diagnostic_rows[:, 0],
            transition_target_index=diagnostic_rows[:, 1],
            transition_probability=diagnostic_probability,
            transition_reaction_index=diagnostic_rows[:, 2],
            sink_probability=sink_probability,
            fragment_probability=fragment_probability,
            fragment_node_index=fragments[:, 1],
            fragment_sample_index=fragments[:, 0],
            state_fragment_index=state_fragment,
            state_probability=conditional,
            formula_explanation_index=core.spectrum_explanations.formula_explanation_index,
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
        if self.descriptor_names:
            # Model-specific optional features are selected after preparation.
            # Nothing is precomputed or persisted on canonical graph tables.
            values = molecular.new_tensor(
                [
                    selected_descriptor_values(compound.smiles, self.descriptor_names)
                    for compound in data.decoded.compounds
                ]
            )
            molecular = torch.cat(
                (molecular, values.sign() * torch.log1p(values.abs())), 1
            )
        return molecular

    def _sample_main_adducts(self, data):
        if getattr(data, "canonical_data", None) is not None:
            core = data.canonical_data
            return [
                core.adducts.main_adducts[
                    int(core.branch_groups.main_adduct_index[group])
                ]
                for group in core.sample_branch_group_index.tolist()
            ]
        return [self.precursor_main_adducts[value] for value in data.precursor_adducts]

    def _fragment_events(self, data, molecular, fragments, sample_count):
        """Build Child/Self/Sink events on sample-specific fragment reaction paths."""
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
                parent_row, child_row = lookup.get((sample, parent)), lookup.get(
                    (sample, child)
                )
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
        self,
        data,
        molecular,
        collision_energy_h,
        reaction_h,
        fragment_nodes,
        fragment_samples,
        source,
        target,
        reactions,
        event_type,
        *, log=False,
    ):
        reaction_features = molecular.new_zeros((source.numel(), self.reaction_dim))
        breaks = reactions >= 0
        reaction_features[breaks] = reaction_h[reactions[breaks]]
        # Self/Sink use the parent's structure twice and zero reaction features.
        child = torch.where(target >= 0, target, source)
        samples = fragment_samples[source]
        inputs = torch.cat(
            (
                molecular[fragment_nodes[source]],
                molecular[fragment_nodes[child]],
                reaction_features,
                collision_energy_h[samples],
            ),
            dim=1,
        )
        mains = self._sample_main_adducts(data)
        sample_values = samples.tolist()
        logits = molecular.new_zeros(source.numel())
        for main in self.main_adduct_conditions:
            rows = torch.tensor(
                [i for i, sample in enumerate(sample_values) if mains[sample] == main],
                dtype=torch.long,
                device=molecular.device,
            )
            if rows.numel():
                logits[rows] = self.fragment_heads[self.head_index[main]](
                    inputs[rows], event_type[rows]
                )
        result = segment_log_softmax(logits, source, fragment_nodes.numel())
        return result if log else result.exp()

    def _score_states(
        self, data, molecular, collision_energy_h, fragments, inverse, roots, *, log=False
    ):
        """Shared across fragments; normalize ion/shift states within each fragment."""
        samples = data.state_sample_index
        nodes = data.state_node_index
        root_nodes = fragments[roots[inverse], 1]
        inputs = torch.cat(
            (molecular[nodes], molecular[root_nodes], collision_energy_h[samples]),
            dim=1,
        )
        mains = self._sample_main_adducts(data)
        sample_values = samples.tolist()
        adducts = data.state_adduct_index.tolist()
        shifts = data.state_hydrogen_shift.tolist()
        logits = molecular.new_zeros(nodes.numel())
        for main, conditions in self.main_adduct_conditions.items():
            rows = [
                i for i, sample in enumerate(sample_values) if mains[sample] == main
            ]
            if not rows:
                continue
            lookup = {value: i for i, value in enumerate(conditions)}
            condition = torch.tensor(
                [lookup[(data.ion_type_strs[adducts[i]], shifts[i])] for i in rows],
                dtype=torch.long,
                device=molecular.device,
            )
            index = torch.tensor(rows, dtype=torch.long, device=molecular.device)
            logits[index] = self.state_heads[self.head_index[main]](
                inputs[index], condition
            )
        result = segment_log_softmax(logits, inverse, fragments.shape[0])
        return result if log else result.exp()

    def _spectrum_loss(
        self,
        data,
        final_probability,
        state_sample_index,
        formulas,
        sample_count,
        settings,
        *, final_log_probability=None,
    ):
        """Merge formula-equivalent probabilities for normalization and supervision."""
        # Compare observed peaks only after summing all structures/paths that
        # yield the same formula in a sample. Keep state rows for viewers.
        keys = torch.cat((state_sample_index[:, None].to(formulas.dtype), formulas), 1)
        _, peak_group_index = torch.unique(keys, dim=0, return_inverse=True)
        peak_count = int(peak_group_index.max()) + 1
        if final_log_probability is None:
            final_log_probability = probability_log(final_probability)
        peak_log_probability = segment_logsumexp(
            final_log_probability, peak_group_index, peak_count
        )
        peak_sample = state_sample_index.new_zeros(peak_count).scatter_reduce_(
            0,
            peak_group_index,
            state_sample_index,
            reduce="amax",
            include_self=True,
        )
        valid_candidate = (formulas[:, :-1] >= 0).all(1) & (data.state_mz > 0)
        # Some supported PyTorch CUDA builds cannot reduce bool tensors.
        valid_peak = torch.zeros(peak_count, dtype=torch.long, device=formulas.device)
        valid_peak.scatter_reduce_(
            0,
            peak_group_index,
            valid_candidate.long(),
            reduce="amax",
            include_self=True,
        )
        valid_peak = valid_peak.bool()
        maxima = final_probability.new_full((sample_count,), -torch.inf).scatter_reduce_(
            0,
            peak_sample,
            peak_log_probability.masked_fill(~valid_peak, -torch.inf),
            reduce="amax",
            include_self=True,
        )
        maxima = maxima.masked_fill(~torch.isfinite(maxima), 0)
        normalized = (final_log_probability - maxima[state_sample_index]).exp()
        zero = final_probability.sum() * 0
        metrics = {
            "spectrum_loss_with_precursor": zero,
            "spectrum_loss_without_precursor": zero,
            "transition_cross_entropy": zero,
            "full_spectrum_cosine": zero,
            "transition_cross_entropy_without_precursor": zero,
            "spectrum_cosine_without_precursor": zero,
        }
        for name in ("spectrum_loss", "cross_entropy_loss", "cosine_loss"):
            for suffix in ("", "_with_precursor", "_without_precursor"):
                metrics[name + suffix] = zero
        if data.target_intensity is None:
            return normalized, zero, metrics
        target_peak = final_probability.new_zeros(peak_count).scatter_reduce_(
            0,
            peak_group_index,
            data.target_intensity.clamp_min(0),
            reduce="amax",
            include_self=True,
        )
        supervision = getattr(data, "target_mask", None)
        if supervision is not None:
            available = torch.zeros_like(valid_peak, dtype=torch.long)
            available.scatter_reduce_(
                0,
                peak_group_index,
                supervision.long(),
                reduce="amax",
                include_self=True,
            )
            valid_peak = valid_peak & available.bool()
        # Masked formulas participate in neural softmaxes, but are excluded from
        # measured intensity objectives, including cosine normalization.
        peak_log_probability = peak_log_probability.masked_fill(~valid_peak, -torch.inf)
        target_peak = target_peak.masked_fill(~valid_peak, 0)
        precursor_peak = torch.zeros_like(valid_peak, dtype=torch.long)
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
                node
                == (
                    int(
                        data.canonical_data.source_compounds.compound_index[
                            data.canonical_data.reaction_contexts.root_source_compound_index[
                                data.canonical_data.sample_branch_group_index[sample]
                            ]
                        ]
                    )
                    if getattr(data, "canonical_data", None) is not None
                    else roots[groups[node]]
                )
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
            0,
            peak_group_index,
            precursor_candidate.long(),
            reduce="amax",
            include_self=True,
        )

        precursor_peak = precursor_peak.bool()

        def objective(log_probability, target):
            target = target.pow(settings.intensity_exponent)
            target_total = log_probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, target
            )
            observed = target_total > 0
            if not bool(observed.any()):
                return zero, zero, zero
            weights = target / target_total[peak_sample].clamp_min(1e-12)
            cross_entropy = -(weights * log_probability.masked_fill(weights == 0, 0)).sum()
            cross_entropy = cross_entropy / observed.sum()
            # Cosine is scale invariant: center in log space before exponentiation.
            maximum = log_probability.new_full((sample_count,), -torch.inf)
            maximum.scatter_reduce_(0, peak_sample, log_probability.detach(),
                                    reduce="amax", include_self=True)
            maximum = maximum.masked_fill(~torch.isfinite(maximum), 0)
            probability = ((log_probability - maximum[peak_sample])
                           * settings.intensity_exponent).exp()
            dot = probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, probability * target
            )
            pred_norm = probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, probability.square()
            )
            target_norm = probability.new_zeros(sample_count).scatter_add_(
                0, peak_sample, target.square()
            )
            cosine = (dot / (pred_norm * target_norm).clamp_min(1e-24).sqrt())[
                observed
            ].mean()
            loss = settings.intensity_loss_weight * (
                cross_entropy + settings.cosine_loss_weight * (1 - cosine)
            )
            return loss, cross_entropy, cosine

        full_loss, cross_entropy, cosine = objective(peak_log_probability, target_peak)
        fragment_log_probability = peak_log_probability.masked_fill(precursor_peak, -torch.inf)
        fragment_total = segment_logsumexp(fragment_log_probability, peak_sample, sample_count)
        fragment_total = fragment_total.masked_fill(~torch.isfinite(fragment_total), 0)
        fragment_log_probability = fragment_log_probability - fragment_total[peak_sample]
        fragment_loss, fragment_cross_entropy, fragment_cosine = objective(
            fragment_log_probability, target_peak.masked_fill(precursor_peak, 0)
        )
        metrics.update(
            spectrum_loss_with_precursor=full_loss,
            spectrum_loss_without_precursor=fragment_loss,
            transition_cross_entropy=cross_entropy,
            full_spectrum_cosine=cosine,
            transition_cross_entropy_without_precursor=fragment_cross_entropy,
            spectrum_cosine_without_precursor=fragment_cosine,
        )
        full_ce = settings.intensity_loss_weight * cross_entropy
        fragment_ce = settings.intensity_loss_weight * fragment_cross_entropy
        full_cosine_loss = full_loss - full_ce
        fragment_cosine_loss = fragment_loss - fragment_ce
        metrics.update(
            spectrum_loss=full_loss + fragment_loss,
            cross_entropy_loss_with_precursor=full_ce,
            cross_entropy_loss_without_precursor=fragment_ce,
            cross_entropy_loss=full_ce + fragment_ce,
            cosine_loss_with_precursor=full_cosine_loss,
            cosine_loss_without_precursor=fragment_cosine_loss,
            cosine_loss=full_cosine_loss + fragment_cosine_loss,
        )
        return normalized, full_loss + fragment_loss, metrics
