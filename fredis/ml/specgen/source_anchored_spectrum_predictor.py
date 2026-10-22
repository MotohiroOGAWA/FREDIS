"""Source -> reactions -> Torch decode -> selected molecules -> tree/intensity."""

from __future__ import annotations
from dataclasses import dataclass, replace
from collections.abc import Sequence
import math
from fredis.ml.input.adduct_data import AdductData
import torch
from torch import nn
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Adduct, Compound
from ..input.source_reaction_structure import (
    SourceReactionStructure,
)
from ..mol.mol_encoder import MolEncoder
from ..mol.formula_encoder import FormulaTensorizer
from .components.condition.adduct_embedding import AdductEmbeddingLayer
from .components.condition.collision_energy_feature import CollisionEnergyFeatureLayer
from .components.reaction.reaction_decoder import ReactionDecoderOutput
from .source_reaction_feature_model import (
    SourceReactionFeatureModel,
    SourceReactionFeatures,
)
from .materialization import materialize_reaction_states, DecodedFragmentTreeBatch
from .generation import GenerationSettings
from .fragment_spectrum_model import (
    prepare_spectrum_model_batch,
    SpectrumModelOutput,
    SpectrumModelBatch,
)


@dataclass(frozen=True)
class CompoundEdgeSelection:
    edge_index: torch.Tensor
    edge_probability: torch.Tensor
    branch_group_index: torch.Tensor


@dataclass(frozen=True)
class SourceAnchoredSelectionOutput:
    features: SourceReactionFeatures
    decoded: CompoundEdgeSelection


@dataclass(frozen=True)
class SourceAnchoredSpectrumOutput:
    selection: SourceAnchoredSelectionOutput
    fragments: DecodedFragmentTreeBatch
    spectra: SpectrumModelOutput
    # state_node_index links each ion state to its
    # fragment-tree node; consumers that need that mapping (e.g.
    # a fragment-tree viewer) read it from here instead of re-deriving it.
    spectrum_batch: SpectrumModelBatch
    # Grouped preparation may reorder records; this maps internal -> input.
    sample_input_index: tuple[int, ...]


class SourceAnchoredFragmentSpectrumGenerator(nn.Module):
    architecture = "fragment-tree-competitive-ion"

    def __init__(
        self,
        fragmenter_params: dict,
        mol_encoder_params: dict,
        reaction_model_params: dict | None = None,
        spectrum_model_params: dict | None = None,
        generation_params: dict | None = None,
        max_samples: int = 128,
        adduct_type_strs: Sequence[str] | None = None,
        fine_tuning: dict | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self.fragmenter = Fragmenter.from_dict(fragmenter_params)
        self.adduct_type_strs = tuple(
            dict.fromkeys(
                str(Adduct.parse(value))
                for value in (
                    adduct_type_strs
                    or (str(adduct) for adduct in self.fragmenter.adduct_types)
                )
            )
        )
        params = dict(reaction_model_params or {})
        spectrum_params = dict(spectrum_model_params or {})
        hidden = params.get("hidden_dim", 128)
        collision_energy_dim = spectrum_params.get("collision_energy_dim", 16)
        mol_params = dict(mol_encoder_params)
        frozen = {str(pid) for pid in (fine_tuning or {}).get("frozen_pattern_ids", [])}
        if any(
            value.get("descriptors")
            for pid, value in params.get("pattern_model_params", {}).items()
            if str(pid) not in frozen
        ):
            raise ValueError(
                "New reaction patterns do not accept descriptors; select intensity descriptors instead"
            )
        if dropout is not None:
            if not math.isfinite(dropout) or not 0 <= dropout < 1:
                raise ValueError("dropout must be finite and in [0, 1)")
            if not fine_tuning:
                mol_params["dropout"] = dropout
            params["edge_dropout"] = dropout
            params["reactant_dropout"] = dropout
            params["pattern_model_params"] = {
                pid: {**value, **({"dropout": dropout} if str(pid) not in frozen else {})}
                for pid, value in params.get("pattern_model_params", {}).items()
            }
            spectrum_params.update(fragment_dropout=dropout, state_dropout=dropout)
        self.mol_encoder = MolEncoder(**mol_params)
        adduct_embedding_dim = spectrum_params.pop("adduct_embedding_dim", 32)
        if type(adduct_embedding_dim) is not int or adduct_embedding_dim < 1:
            raise ValueError("adduct_embedding_dim must be a positive integer")
        adduct_encoder = AdductEmbeddingLayer(
            self.adduct_type_strs, adduct_embedding_dim
        )
        collision_energy_encoder = CollisionEnergyFeatureLayer(
            collision_energy_dim,
            fc_dims=spectrum_params.pop("ce_fc_dims", (32,)),
            center=spectrum_params.pop("ce_center", 0.0),
            scale=spectrum_params.pop("ce_scale", 100.0),
        )
        patterns = self.fragmenter.fragment_ion_tree_builder.reaction_patterns
        sizes = (
            max((pattern.pattern_id for pattern in patterns), default=0) + 1,
            max(
                (rule.id for pattern in patterns for rule in pattern.reaction_rules),
                default=0,
            )
            + 1,
            max(
                (
                    len(rule.prod_temp)
                    for pattern in patterns
                    for rule in pattern.reaction_rules
                ),
                default=1,
            ),
        )
        self.feature_model = SourceReactionFeatureModel(
            self.mol_encoder,
            sizes,
            max_reaction_count=self.fragmenter.tree_max_reaction_count,
            adduct_encoder=adduct_encoder,
            collision_energy_encoder=collision_energy_encoder,
            pattern_smarts={
                pattern.pattern_id: pattern.reactant_smarts for pattern in patterns
            },
            pattern_category_sizes={
                pattern.pattern_id: (
                    max((rule.id for rule in pattern.reaction_rules), default=0) + 1,
                    max(
                        (len(rule.prod_temp) for rule in pattern.reaction_rules),
                        default=1,
                    ),
                )
                for pattern in patterns
            },
            **params,
        )
        self.tensorizer = FormulaTensorizer.from_symbols_and_adducts(
            symbols=self.mol_encoder.symbols,
            adducts=tuple(self.fragmenter.adduct_types)
            + tuple(
                ion
                for rule in self.fragmenter.adduct_rule_set.adduct_rules
                for ion in rule.fragment_ion_adducts
            ),
        )
        ion_types = tuple(
            dict.fromkeys(
                str(value)
                for rule in self.fragmenter.adduct_rule_set.adduct_rules
                for value in rule.fragment_ion_adducts
            )
        )
        shifts = self.fragmenter.adduct_rule_set.hydrogen_shifts
        main_adduct_conditions = {
            str(rule.adduct_type): [
                (str(ion), shift)
                for ion in rule.fragment_ion_adducts
                for shift in shifts
            ]
            for rule in self.fragmenter.adduct_rule_set.adduct_rules
        }
        precursor_main_adducts = {
            value: str(self.fragmenter._resolve_main_adduct_type(Adduct.parse(value)))
            for value in self.adduct_type_strs
        }
        from .competitive_spectrum_model import CompetitiveSpectrumModel

        self.spectrum_model = CompetitiveSpectrumModel(
            self.mol_encoder,
            hidden,
            main_adduct_conditions=main_adduct_conditions,
            precursor_main_adducts=precursor_main_adducts,
            formula_tensorizer=self.tensorizer,
            **spectrum_params,
        )
        self.ion_type_strs = ion_types
        self.set_generation(GenerationSettings.from_dict(generation_params))
        if fine_tuning:
            from .fine_tuning import configure_pattern_training

            configure_pattern_training(self, fine_tuning)

        if dropout is not None:
            # Imported source features stay frozen and retain their saved rates.
            protected = set()
            if fine_tuning:
                protected.update(map(id, self.mol_encoder.modules()))
            reaction = self.feature_model.reaction_instance_encoder
            for pid in frozen:
                for group in (reaction.reactant_encoders, reaction.pattern_heads, reaction.neighborhood_encoders):
                    if pid in group:
                        protected.update(map(id, group[pid].modules()))
            for module in self.modules():
                if id(module) in protected:
                    continue
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout

    def train(self, mode: bool = True):
        super().train(mode)
        # MolEncoder is shared with spectrum_model, whose train() runs last.
        if getattr(self.feature_model, "_freeze_mol_encoder", False):
            self.mol_encoder.eval()
        return self

    def set_generation(self, settings: GenerationSettings) -> None:
        """Search and output settings used by every subsequent prediction."""
        self.generation = settings

    def forward(self, data):
        """Neural edge scoring on shared canonical inputs; no chemistry."""
        features = self.feature_model(data)
        logits, _ = self.feature_model.score_edges(data, features)
        return SourceAnchoredSelectionOutput(
            features,
            CompoundEdgeSelection(
                data.reaction_contexts.edge_index,
                logits.sigmoid(),
                torch.repeat_interleave(
                    torch.arange(data.num_branch_groups, device=data.device),
                    data.reaction_contexts.edge_ptr.diff(),
                ),
            ),
        )

    def prepare(self, sources, precursor_types, collision_energy, *, precomputed=None):
        from ..input.structure_builder import ReactionStructureBuilder
        from ..input.source_reaction_structure import SpectrumPredictionData

        if not sources or not len(sources) == len(precursor_types) == len(
            collision_energy
        ):
            raise ValueError(
                "sources, precursor_types and collision_energy need equal nonzero lengths"
            )
        grouped = {}
        for i, source in enumerate(sources):
            grouped.setdefault(source.smiles, []).append(i)
        inputs = []
        unique = []
        reactions = []
        adducts = []
        indices = []
        for rows in grouped.values():
            source = sources[rows[0]]
            builder = ReactionStructureBuilder(self)
            cached = precomputed.get(source.smiles) if precomputed else None
            kwargs = (
                {}
                if cached is None
                else {
                    "precomputed_reactions": cached["reactions"],
                    "precomputed_relations": cached["relations"],
                }
            )
            data, kept = builder.build_prediction(
                source,
                [precursor_types[i] for i in rows],
                [collision_energy[i] for i in rows],
                **kwargs,
            )
            inputs.append(data)
            unique.append(source)
            reactions.append(builder.last_reactions)
            adducts.extend(precursor_types[rows[i]] for i in kept)
            indices.extend(rows[i] for i in kept)
        return (
            SpectrumPredictionData.from_structures(inputs),
            tuple(unique),
            tuple(reactions),
            tuple(adducts),
            tuple(indices),
        )

    def _select_graph(self, data, logits):
        """Keep high-probability reachable branches with one budget per main-adduct group."""
        from ..input.source_reaction_structure import csr
        from ..input.annotated_spectrum_data import SpectrumExplanationData
        import heapq, math

        execution = data.execution_graph
        graph = data.compound_graph
        ctx = data.reaction_contexts
        edge_lookup = {
            (int(a), int(b), int(r)): i
            for i, (a, b, r) in enumerate(
                zip(
                    graph.source_compound_index,
                    graph.target_compound_index,
                    graph.reaction_index,
                )
            )
        }
        edge_rows = []
        selected_compounds = []
        threshold = (
            -math.inf
            if self.generation.branch_path_threshold == 0
            else math.log(self.generation.branch_path_threshold)
        )
        for group in range(data.num_branch_groups):
            first, last = ctx.edge_ptr[group : group + 2]
            scores = {
                int(edge): float(torch.nn.functional.logsigmoid(logits[int(first) + j]))
                for j, edge in enumerate(ctx.edge_index[first:last])
            }
            outgoing = {}
            for parent, child, reaction in zip(
                execution.source_index.tolist(),
                execution.target_index.tolist(),
                execution.reaction_index.tolist(),
            ):
                if int(execution.branch_group_index[parent]) != group:
                    continue
                eid = edge_lookup[
                    (
                        int(execution.compound_index[parent]),
                        int(execution.compound_index[child]),
                        reaction,
                    )
                ]
                outgoing.setdefault(parent, []).append((child, eid))
            root = int(execution.root_index[group])
            reached = {root: 0.0}
            compounds = {int(execution.compound_index[root])}
            kept = set()
            queue = [(0.0, root)]
            while queue:
                negative_score, parent = heapq.heappop(queue)
                if -negative_score < reached[parent]:
                    continue
                for child, eid in outgoing.get(parent, []):
                    score = -negative_score + scores[eid]
                    compound = int(execution.compound_index[child])
                    if score < threshold:
                        continue
                    if (
                        compound not in compounds
                        and len(compounds) >= self.generation.max_fragment_nodes
                    ):
                        continue
                    compounds.add(compound)
                    kept.add(eid)
                    if score > reached.get(child, -math.inf):
                        reached[child] = score
                        heapq.heappush(queue, (-score, child))
            edge_rows.append(sorted(kept))
            selected_compounds.append(compounds)
        ep, ei = csr(edge_rows, data.device)
        candidates = []
        e = data.formula_explanations
        for sample, group in enumerate(data.sample_branch_group_index.tolist()):
            a, b = data.spectrum_explanations.formula_explanation_ptr[
                sample : sample + 2
            ]
            candidates.append(
                [
                    i
                    for i in data.spectrum_explanations.formula_explanation_index[
                        a:b
                    ].tolist()
                    if int(e.compound_index[i]) in selected_compounds[group]
                ]
            )
        xp, xi = csr(candidates, data.device)
        return replace(
            data,
            reaction_contexts=replace(ctx, edge_ptr=ep, edge_index=ei, edge_label=None),
            spectrum_explanations=SpectrumExplanationData(xp, xi),
        )

    @torch.no_grad()
    def predict_batches(
        self,
        sources,
        precursor_types,
        collision_energy,
        *,
        max_samples=None,
        precomputed=None,
    ):
        from .fragment_spectrum_model import spectrum_batch_from_input

        if self.training:
            raise ValueError("Call eval() before predict_batches()")
        limit = self.max_samples if max_samples is None else max_samples
        if type(limit) is not int or limit < 1:
            raise ValueError("max_samples must be positive")
        if not sources or not len(sources) == len(precursor_types) == len(
            collision_energy
        ):
            raise ValueError("Input lengths must match and be nonzero")
        groups = {}
        for i, source in enumerate(sources):
            groups.setdefault(source.smiles, []).append(i)
        packs = []
        pack = []
        for rows in groups.values():
            if pack and len(pack) + len(rows) > limit:
                packs.append(pack)
                pack = []
            pack.extend(rows)
            if len(pack) >= limit:
                packs.append(pack)
                pack = []
        if pack:
            packs.append(pack)
        device = next(self.parameters()).device
        for rows in packs:
            data, _, _, _, order = self.prepare(
                [sources[i] for i in rows],
                [precursor_types[i] for i in rows],
                [collision_energy[i] for i in rows],
                precomputed=precomputed,
            )
            data = data.to(device)
            static = self.feature_model.encode_static(data, max_graphs=limit)
            features = self.feature_model(data, static_features=static)
            molecular = self.feature_model.encode_compounds(
                data, features, max_graphs=limit
            )
            logits, _ = self.feature_model.score_edges(data, features, molecular)
            selected = self._select_graph(data, logits)
            for start in range(0, selected.num_samples, limit):
                indices = list(range(start, min(start + limit, selected.num_samples)))
                chunk = selected.select_samples(indices)
                batch = spectrum_batch_from_input(chunk)
                chunk_features = self.feature_model(chunk, static_features=static)
                spectrum = self.spectrum_model(
                    batch,
                    reaction_h=chunk_features.reaction_h,
                    collision_energy_h=chunk_features.collision_energy_h,
                    source_embeddings=chunk_features.source_embeddings,
                    molecular_embeddings=molecular,
                )
                chunk_logits, _ = self.feature_model.score_edges(
                    chunk, chunk_features, molecular
                )
                selection = SourceAnchoredSelectionOutput(
                    chunk_features,
                    CompoundEdgeSelection(
                        chunk.reaction_contexts.edge_index,
                        chunk_logits.sigmoid(),
                        torch.repeat_interleave(
                            torch.arange(chunk.num_branch_groups, device=device),
                            chunk.reaction_contexts.edge_ptr.diff(),
                        ),
                    ),
                )
                yield SourceAnchoredSpectrumOutput(
                    selection,
                    batch.decoded,
                    spectrum,
                    batch,
                    tuple(rows[order[i]] for i in indices),
                )

    @torch.no_grad()
    def predict(self, sources, precursor_types, collision_energy, *, precomputed=None):
        if len(sources) > self.max_samples:
            raise ValueError("Use predict_batches() for inputs exceeding max_samples")
        outputs = list(
            self.predict_batches(
                sources,
                precursor_types,
                collision_energy,
                max_samples=self.max_samples,
                precomputed=precomputed,
            )
        )
        if len(outputs) != 1:
            raise ValueError("Use predict_batches() for multi-pack inputs")
        return outputs[0]
