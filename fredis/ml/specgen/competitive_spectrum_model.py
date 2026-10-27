"""Fragment reaction-path survival followed by conditional ion/shift state probabilities."""

from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import torch
from torch import nn
from .spectrum_topology import canonical_spectrum_topology, probability_levels
from .components.condition.condition_linear import ConditionHead
from .fragment_spectrum_model import SpectrumModelOutput, SpectrumLossSettings
from fredis.domain.molecule.descriptors import SUPPORTED_DESCRIPTOR_NAMES
from ..mol.selected_descriptors import selected_descriptor_values
from ..common.layers.set_transformer import SetTransformer
from ..common.layers.gat import GAT
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
    maximum.scatter_reduce_(
        0, groups, values.detach(), reduce="amax", include_self=True
    )
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
    return (
        probability.clamp_min(torch.finfo(probability.dtype).tiny)
        .log()
        .masked_fill(probability == 0, -torch.inf)
    )


def propagate_log_probability(initial, source, target, probability, *, levels=None):
    """Propagate log mass in parallel within each dependency level.

    Static event indices can be reused across forwards. All probability operations
    remain on the input device and retain autograd, including merged histories.
    """
    count = initial.numel()
    if levels is None:
        levels = probability_levels(source, target, count)
    incoming = initial
    final = initial.new_full(initial.shape, -torch.inf)
    sink = initial.new_full(initial.shape, -torch.inf)
    states = torch.arange(count, device=source.device)

    def accumulate(existing, destinations, values):
        return segment_logsumexp(
            torch.cat((existing, values)), torch.cat((states, destinations)), count
        )

    for self_events, sink_events, child_events in levels:
        final = accumulate(final, source[self_events],
                           incoming[source[self_events]] + probability[self_events])
        sink = accumulate(sink, source[sink_events],
                          incoming[source[sink_events]] + probability[sink_events])
        incoming = accumulate(incoming, target[child_events],
                              incoming[source[child_events]] + probability[child_events])
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
    # Formula sums can round slightly above one. Apply output normalization
    # before the inclusive intensity threshold, using the merged peak maxima.
    sample = output.sample_index[first]
    maxima = intensity.new_zeros(int(sample.max()) + 1 if sample.numel() else 0)
    maxima.scatter_reduce_(0, sample, intensity, reduce="amax", include_self=True)
    intensity = intensity / maxima[sample].clamp_min(torch.finfo(intensity.dtype).tiny)
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
        gat_num_layers=2,
        state_set_num_heads=1,
        state_set_num_layers=2,
        gat_ffn_mult=4,
        state_set_ffn_mult=2,
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
        if (
            type(state_set_num_heads) is not int
            or state_set_num_heads < 1
            or reaction_dim % state_set_num_heads
        ):
            raise ValueError(
                "state_set_num_heads must be positive and divide reaction_dim"
            )
        if type(state_set_num_layers) is not int or state_set_num_layers < 1:
            raise ValueError("state_set_num_layers must be a positive integer")
        if any(
            type(value) is not int or value < 1
            for value in (gat_ffn_mult, state_set_ffn_mult)
        ):
            raise ValueError("Spectrum FFN multipliers must be positive integers")
        self.state_set_encoder = SetTransformer(
            reaction_dim,
            reaction_dim,
            reaction_dim,
            num_heads=state_set_num_heads,
            num_layers=state_set_num_layers,
            ffn_mult=state_set_ffn_mult,
            dropout=dropout,
        )
        state_input_dim = (
            2 * (mol_encoder.graph_dim + len(descriptors))
            + collision_energy_dim
            + reaction_dim
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
        if type(gat_num_layers) is not int or gat_num_layers < 1:
            raise ValueError("gat_num_layers must be a positive integer")
        node_dim = mol_encoder.graph_dim + len(descriptors)
        self.main_adduct_embedding = nn.Embedding(len(self.head_index), reaction_dim)
        self.fragment_projection = nn.Linear(
            2 * node_dim + collision_energy_dim + reaction_dim, node_dim
        )
        self.fragment_gat = GAT(
            node_dim,
            reaction_dim + 1,
            gat_num_layers,
            ffn_hidden_mult=gat_ffn_mult,
            dropout=dropout,
            directed=True,
            softmax_per="dst",
        )

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
            root_nodes=fragment_nodes[root_index],
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
        """Propagate histories, score applied sets, then merge physical ion mass."""
        core = data.canonical_data
        execution = core.execution_graph
        molecular = self._encode_molecules(
            data, source_embeddings, molecular_embeddings
        )
        topology = data.__dict__.get("_canonical_topology")
        if topology is None:
            topology = canonical_spectrum_topology(data)
            data.__dict__["_canonical_topology"] = topology
        execution_rows = topology.execution_rows
        execution_compounds = topology.compounds
        execution_samples = topology.samples
        root_rows = topology.roots
        source, target = topology.source, topology.target
        reactions, event_type = topology.reactions, topology.event_type
        execution_to_physical = topology.history_to_fragment
        fragments = topology.fragments
        state_fragment = topology.state_fragment
        physical_count = fragments.shape[0]
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
            root_nodes=execution_compounds[root_rows],
            log=True,
        )
        initial = molecular.new_zeros(execution_compounds.numel())
        initial[root_rows] = 1
        log_retained, log_sink = propagate_log_probability(
            probability_log(initial), source, target, probability, levels=topology.levels
        )
        sink = log_sink.exp()
        log_event_probability = probability
        log_fragment = segment_logsumexp(
            log_retained, execution_to_physical, physical_count
        )
        fragment_probability = log_fragment.exp()
        final_log_probability, conditional = self._score_reaction_set_states(
            data,
            molecular,
            collision_energy_h,
            reaction_h,
            core.reaction_sets,
            execution.reaction_set_index[execution_rows],
            log_retained,
            execution_to_physical,
            log_fragment,
            fragments,
            state_fragment,
            topology.physical_roots,
        )
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
            num_fragment_transitions=(event_type == 0).sum().to(final.dtype),
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
            log_event_mass, execution_to_physical[source], physical_count
        )
        normalizer = totals[diagnostic_rows[:, 0]]
        diagnostic_probability = (
            aggregate_mass - normalizer.masked_fill(~torch.isfinite(normalizer), 0)
        ).exp()
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

    def _sample_head_indices(self, data, device=None):
        """Map categorical metadata once; select GPU rows with tensor indices."""
        device = data.state_node_index.device if device is None else device
        key = (tuple(self.head_index), tuple(self.precursor_main_adducts.items()), str(device))
        cache = data.__dict__.setdefault("_sample_heads", {})
        if key not in cache:
            if getattr(data, "canonical_data", None) is not None:
                core = data.canonical_data
                vocabulary = torch.tensor(
                    [int(self.head_index[main]) for main in core.adducts.main_adducts],
                    dtype=torch.long, device=device,
                )
                cache[key] = vocabulary[core.branch_groups.main_adduct_index[
                    core.sample_branch_group_index]]
            else:
                cache[key] = torch.tensor(
                    [int(self.head_index[self.precursor_main_adducts[value]])
                     for value in data.precursor_adducts], dtype=torch.long, device=device,
                )
        return cache[key]

    def _state_condition_indices(self, data):
        """Resolve ion/shift conditions without a Python loop over ion states."""
        key = (tuple(self.main_adduct_conditions.items()), tuple(self.precursor_main_adducts.items()))
        cache = data.__dict__.setdefault("_state_conditions", {})
        if key not in cache:
            conditions = tuple(self.main_adduct_conditions.values())
            shifts = [shift for values in conditions for _, shift in values]
            minimum, maximum = min(shifts), max(shifts)
            vocabulary = {ion: index for index, ion in enumerate(data.ion_type_strs)}
            table = torch.full((len(conditions), len(vocabulary), maximum - minimum + 1),
                               -1, dtype=torch.long)
            for head, values in enumerate(conditions):
                for index, (ion, shift) in enumerate(values):
                    if ion in vocabulary:
                        table[head, vocabulary[ion], shift - minimum] = index
            table = table.to(data.state_node_index.device)
            positions = data.state_hydrogen_shift - minimum
            selected = table[
                self._sample_head_indices(data)[data.state_sample_index],
                data.state_adduct_index, positions.clamp(0, maximum - minimum),
            ]
            if bool(((positions < 0) | (positions > maximum - minimum) | (selected < 0)).any()):
                raise ValueError("Ion state has no condition in its main-adduct head")
            cache[key] = selected
        return cache[key]

    @staticmethod
    def _precursor_candidates(data, samples):
        """Compute root/signature masks with one gather per complete state vector."""
        cached = data.__dict__.get("_precursor_candidates")
        if cached is not None and cached[0] is samples:
            return cached[1]
        device = data.state_node_index.device
        if getattr(data, "canonical_data", None) is not None:
            core = data.canonical_data
            roots = core.source_compounds.compound_index[
                core.reaction_contexts.root_source_compound_index[
                    core.sample_branch_group_index]]
            is_root = data.state_node_index == roots[samples]
        else:
            groups = data.decoded.node_sample_index
            nodes = torch.arange(groups.numel(), device=device)
            roots = nodes.new_full((groups.numel(),), groups.numel())
            roots.scatter_reduce_(0, groups, nodes, reduce="amin", include_self=True)
            is_root = data.state_node_index == roots[groups[data.state_node_index]]
        signatures = [_adduct_without_hydrogen(ion) for ion in data.ion_type_strs]
        same_ion = torch.tensor(
            [[_adduct_without_hydrogen(precursor) == signature for signature in signatures]
             for precursor in data.precursor_adducts], dtype=torch.bool, device=device,
        )
        result = is_root & same_ion[samples, data.state_adduct_index]
        data.__dict__["_precursor_candidates"] = (samples, result)
        return result

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
        *,
        root_nodes,
        log=False,
    ):
        # Merge physical structures within each spectrum before graph attention.
        # Execution histories retain separate outgoing candidate sets below.
        physical, inverse = torch.unique(
            torch.stack((fragment_samples, fragment_nodes), dim=1),
            dim=0,
            return_inverse=True,
        )
        sample, compound = physical.unbind(1)
        main_index = self._sample_head_indices(data, molecular.device)
        node_h = self.fragment_projection(
            torch.cat(
                (
                    molecular[compound],
                    molecular[root_nodes[sample]],
                    collision_energy_h[sample],
                    self.main_adduct_embedding(main_index[sample]),
                ),
                dim=1,
            )
        )
        breaks = reactions >= 0
        physical_edges = torch.unique(
            torch.stack(
                (
                    inverse[source[breaks]],
                    inverse[target[breaks]],
                    reactions[breaks],
                ),
                dim=1,
            ),
            dim=0,
        )
        parent, child, reaction = physical_edges.unbind(1)
        edges = torch.stack((parent, child))
        edge_h = reaction_h[reaction]
        # Both message directions share reaction features but carry opposite
        # direction flags. Probability still flows only along physical reactions.
        message_index = torch.cat((edges, edges.flip(0)), dim=1)
        forward = torch.cat((edge_h, edge_h.new_ones((edge_h.shape[0], 1))), dim=1)
        reverse = torch.cat((edge_h, -edge_h.new_ones((edge_h.shape[0], 1))), dim=1)
        message_h = torch.cat((forward, reverse))
        node_h, _ = self.fragment_gat(node_h, message_index, message_h)
        node_h = node_h[inverse]
        reaction_features = molecular.new_zeros((source.numel(), self.reaction_dim))
        reaction_features[breaks] = reaction_h[reactions[breaks]]
        # Self/Sink retain the parent's representation as both endpoints.
        child = torch.where(target >= 0, target, source)
        samples = fragment_samples[source]
        inputs = torch.cat(
            (
                node_h[source],
                node_h[child],
                reaction_features,
                collision_energy_h[samples],
            ),
            dim=1,
        )
        logits = molecular.new_zeros(source.numel())
        for main, index in self.head_index.items():
            rows = (main_index[samples] == int(index)).nonzero().flatten()
            if rows.numel():
                logits[rows] = self.fragment_heads[self.head_index[main]](
                    inputs[rows], event_type[rows]
                )
        result = segment_log_softmax(logits, source, fragment_nodes.numel())
        return result if log else result.exp()

    def _encode_reaction_sets(self, reaction_h, sets, set_index):
        """Encode each requested unique applied-reaction set once, across spectra.

        The empty set is represented by masked padding; SetTransformer's PMA
        seed supplies a finite learned representation. Packing is vectorized and
        attention is bounded to 128 unique sets per call.
        """
        unique, inverse = torch.unique(set_index, return_inverse=True)
        if not unique.numel():
            return reaction_h.new_empty((0, self.reaction_dim))
        ptr, values = sets.gather(unique)
        counts = ptr.diff()
        width = max(1, int(counts.max()))
        owner = torch.repeat_interleave(
            torch.arange(unique.numel(), device=unique.device), counts
        )
        slot = torch.arange(values.numel(), device=unique.device) - ptr[owner]
        tokens = reaction_h.new_zeros((unique.numel(), width, self.reaction_dim))
        mask = torch.ones(
            (unique.numel(), width), dtype=torch.bool, device=unique.device
        )
        tokens[owner, slot] = reaction_h[values]
        mask[owner, slot] = False
        encoded = torch.cat(
            [
                self.state_set_encoder(
                    tokens[start : start + 128],
                    key_padding_mask=mask[start : start + 128],
                )
                for start in range(0, unique.numel(), 128)
            ]
        )
        return encoded[inverse]

    def _score_reaction_set_states(
        self,
        data,
        molecular,
        collision_energy_h,
        reaction_h,
        sets,
        set_index,
        log_retained,
        history_to_fragment,
        log_fragment,
        fragments,
        state_fragment,
        roots,
    ):
        """Score each (spectrum, Compound, applied set) independently, then sum.

        Histories differing only in reaction order share their set-conditioned
        state distribution. Different sets never mix neural features. Their joint
        ion probabilities are summed in log space after state normalization.
        """
        groups, history_group = torch.unique(
            torch.stack((history_to_fragment, set_index), dim=1),
            dim=0,
            return_inverse=True,
        )
        group_fragment = groups[:, 0]
        log_group_mass = segment_logsumexp(log_retained, history_group, groups.shape[0])
        set_h = self._encode_reaction_sets(reaction_h, sets, groups[:, 1])
        # Expand only the ion-state grid for each distinct applied set, not for
        # every history or reaction order. torch.unique sorts groups by fragment.
        counts = torch.bincount(group_fragment, minlength=fragments.shape[0])
        starts = counts.cumsum(0) - counts
        repeats = counts[state_fragment]
        state_rows = torch.repeat_interleave(
            torch.arange(state_fragment.numel(), device=state_fragment.device), repeats
        )
        row_starts = repeats.cumsum(0) - repeats
        slots = (
            torch.arange(state_rows.numel(), device=state_rows.device)
            - row_starts[state_rows]
        )
        state_group = starts[state_fragment[state_rows]] + slots
        group_roots = fragments[roots[group_fragment], 1]
        log_state = self._score_states(
            data,
            molecular,
            collision_energy_h,
            fragments[group_fragment],
            state_group,
            roots=None,
            reaction_set_h=set_h,
            state_rows=state_rows,
            root_nodes=group_roots[state_group],
            log=True,
        )
        joint = log_group_mass[state_group] + log_state
        final_log = segment_logsumexp(joint, state_rows, state_fragment.numel())
        # Physical-fragment conditional probabilities are reporting marginals.
        # An unreachable fragment has no conditional mixture; use a normalized
        # uniform mixture over its sets for diagnostics only, never for intensity.
        normalizer = log_fragment[state_fragment]
        conditional = final_log - normalizer.masked_fill(~torch.isfinite(normalizer), 0)
        fallback = segment_logsumexp(
            log_state - counts[group_fragment[state_group]].to(log_state.dtype).log(),
            state_rows,
            state_fragment.numel(),
        )
        conditional = torch.where(torch.isfinite(normalizer), conditional, fallback)
        return final_log, conditional

    def _score_states(
        self,
        data,
        molecular,
        collision_energy_h,
        fragments,
        inverse,
        roots,
        *,
        reaction_set_h=None,
        state_rows=None,
        root_nodes=None,
        log=False,
    ):
        """Normalize ion/shift states separately for each supplied state group."""
        if state_rows is None:
            state_rows = torch.arange(
                data.state_node_index.numel(), device=molecular.device
            )
        samples = data.state_sample_index[state_rows]
        nodes = data.state_node_index[state_rows]
        if root_nodes is None:
            root_nodes = fragments[roots[inverse], 1]
        if reaction_set_h is None:
            # Tensor-only graphs without applied-set metadata use the empty set.
            tokens = molecular.new_zeros((1, 1, self.reaction_dim))
            mask = torch.ones((1, 1), dtype=torch.bool, device=molecular.device)
            reaction_set_h = self.state_set_encoder(
                tokens, key_padding_mask=mask
            ).expand(fragments.shape[0], -1)
        inputs = torch.cat(
            (
                molecular[nodes],
                molecular[root_nodes],
                collision_energy_h[samples],
                reaction_set_h[inverse],
            ),
            dim=1,
        )
        main_index = self._sample_head_indices(data)[samples]
        condition = self._state_condition_indices(data)[state_rows]
        logits = molecular.new_zeros(nodes.numel())
        for head in self.state_heads:
            index = (main_index == int(head)).nonzero(as_tuple=True)[0]
            if index.numel():
                logits[index] = self.state_heads[head](inputs[index], condition[index])
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
        *,
        final_log_probability=None,
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
        maxima = final_probability.new_full(
            (sample_count,), -torch.inf
        ).scatter_reduce_(
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
            "supervised_peak_cosine_with_precursor": zero,
            "transition_cross_entropy_without_precursor": zero,
            "supervised_peak_cosine_without_precursor": zero,
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
        precursor_candidate = self._precursor_candidates(data, state_sample_index)
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
            cross_entropy = -(
                weights * log_probability.masked_fill(weights == 0, 0)
            ).sum()
            cross_entropy = cross_entropy / observed.sum()
            # Cosine is scale invariant: center in log space before exponentiation.
            maximum = log_probability.new_full((sample_count,), -torch.inf)
            maximum.scatter_reduce_(
                0,
                peak_sample,
                log_probability.detach(),
                reduce="amax",
                include_self=True,
            )
            maximum = maximum.masked_fill(~torch.isfinite(maximum), 0)
            probability = (
                (log_probability - maximum[peak_sample]) * settings.intensity_exponent
            ).exp()
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
        fragment_log_probability = peak_log_probability.masked_fill(
            precursor_peak, -torch.inf
        )
        fragment_total = segment_logsumexp(
            fragment_log_probability, peak_sample, sample_count
        )
        fragment_total = fragment_total.masked_fill(~torch.isfinite(fragment_total), 0)
        fragment_log_probability = (
            fragment_log_probability - fragment_total[peak_sample]
        )
        fragment_loss, fragment_cross_entropy, fragment_cosine = objective(
            fragment_log_probability, target_peak.masked_fill(precursor_peak, 0)
        )
        metrics.update(
            spectrum_loss_with_precursor=full_loss,
            spectrum_loss_without_precursor=fragment_loss,
            transition_cross_entropy=cross_entropy,
            supervised_peak_cosine_with_precursor=cosine,
            transition_cross_entropy_without_precursor=fragment_cross_entropy,
            supervised_peak_cosine_without_precursor=fragment_cosine,
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
