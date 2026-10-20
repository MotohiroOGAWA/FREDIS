"""Source -> reactions -> Torch decode -> selected molecules -> tree/intensity."""

from __future__ import annotations
from dataclasses import dataclass, replace
from collections.abc import Sequence
import torch
from torch import nn
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Adduct, Compound
from ..input.source_reaction_structure import (
    SourceReactionStructure,
    prepare_source_reactions,
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
class SourceAnchoredSelectionOutput:
    features: SourceReactionFeatures
    decoded: ReactionDecoderOutput


@dataclass(frozen=True)
class SourceAnchoredSpectrumOutput:
    selection: SourceAnchoredSelectionOutput
    fragments: DecodedFragmentTreeBatch
    spectra: SpectrumModelOutput
    # physical_candidate_node_index links each candidate ion to its
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
        main_adduct_dim = spectrum_params.get("main_adduct_dim", 128)
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
        self.mol_encoder = MolEncoder(**mol_params)
        adduct_embedding_dim = spectrum_params.pop("adduct_embedding_dim", 32)
        if type(adduct_embedding_dim) is not int or adduct_embedding_dim < 1:
            raise ValueError("adduct_embedding_dim must be a positive integer")
        adduct_encoder = AdductEmbeddingLayer(self.adduct_type_strs, adduct_embedding_dim)
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
            intensity_main_adduct_dim=main_adduct_dim,
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
            symbols=self.mol_encoder.symbols, adducts=self.fragmenter.adduct_types
        )
        ion_types = tuple(
            dict.fromkeys(
                str(value)
                for rule in self.fragmenter.adduct_rule_set.adduct_rules
                for value in rule.fragment_ion_adducts
            )
        )
        spectrum_params.setdefault("ion_type_count", len(ion_types))
        spectrum_params.setdefault(
            "hydrogen_shift_min", self.fragmenter.adduct_rule_set.hydrogen_shift_min
        )
        spectrum_params.setdefault(
            "hydrogen_shift_max", self.fragmenter.adduct_rule_set.hydrogen_shift_max
        )
        from .competitive_spectrum_model import CompetitiveSpectrumModel

        self.spectrum_model = CompetitiveSpectrumModel(
            self.mol_encoder, hidden, **spectrum_params
        )
        self.ion_type_strs = ion_types
        self.set_generation(GenerationSettings.from_dict(generation_params))
        if fine_tuning:
            from .fine_tuning import configure_pattern_training

            configure_pattern_training(self, fine_tuning)

    def set_generation(self, settings: GenerationSettings) -> None:
        """Search and output settings used by every subsequent prediction."""
        self.generation = settings
        self.feature_model.decoder.apply_generation(settings)

    def forward(self, data: SourceReactionStructure) -> SourceAnchoredSelectionOutput:
        """Neural forward, including validation, never invokes RDKit."""
        features = self.feature_model(data)
        return SourceAnchoredSelectionOutput(
            features,
            self.feature_model.decoder(features.pool, features.branch_main_adduct_h),
        )

    def prepare(
        self,
        sources: Sequence[Compound],
        precursor_types: Sequence[Adduct],
        collision_energy: Sequence[float],
        *,
        precomputed: dict[str, dict] | None = None,
    ) -> tuple[
        SourceReactionStructure,
        tuple[Compound, ...],
        tuple[tuple, ...],
        tuple[Adduct, ...],
        tuple[int, ...],
    ]:
        if not sources or not len(sources) == len(precursor_types) == len(
            collision_energy
        ):
            raise ValueError(
                "sources, precursor_types and collision_energy need equal nonzero lengths"
            )
        grouped: dict[str, list[int]] = {}
        for i, source in enumerate(sources):
            grouped.setdefault(source.smiles, []).append(i)
        structures, unique_sources, universes, ordered_adducts, input_indices = (
            [],
            [],
            [],
            [],
            [],
        )
        for rows in grouped.values():
            source = sources[rows[0]]
            cached = precomputed.get(source.smiles) if precomputed else None
            if cached is not None:
                # Primitive reactions came from a prior, possibly subprocess-
                # isolated preparation pass and are reused verbatim.
                reactions = cached["reactions"]
                relations = cached["relations"]
            else:
                relations = None
                reactions = (
                    self.fragmenter.fragment_ion_tree_builder.create_reaction_instances(
                        source
                    )
                )
            main = [
                self.fragmenter._resolve_main_adduct_type(precursor_types[i])
                for i in rows
            ]
            conditions = torch.tensor(
                [
                    [self.adduct_type_strs.index(str(adduct)), collision_energy[i]]
                    for adduct, i in zip(main, rows)
                ],
                dtype=torch.float32,
            )
            group_keys = {}
            sample_group = []
            for adduct in main:
                sample_group.append(group_keys.setdefault(str(adduct), len(group_keys)))
            group_adduct = torch.tensor(
                [self.adduct_type_strs.index(key) for key in group_keys],
                dtype=torch.long,
            )
            structures.append(
                prepare_source_reactions(
                    source=source,
                    reactions=reactions,
                    relations=relations,
                    graph_builder=self.mol_encoder.graph_builder,
                    condition_features=conditions,
                    max_reaction_count=self.feature_model.max_reaction_count,
                    sample_branch_group_index=torch.tensor(sample_group),
                    branch_group_adduct_index=group_adduct,
                )
            )
            unique_sources.append(source)
            universes.append(reactions)
            ordered_adducts.extend(precursor_types[i] for i in rows)
            input_indices.extend(rows)
        data = SourceReactionStructure.from_structures(structures)
        return (
            data,
            tuple(unique_sources),
            tuple(universes),
            tuple(ordered_adducts),
            tuple(input_indices),
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
        limit = self.max_samples if max_samples is None else max_samples
        if self.training:
            raise ValueError("Call eval() before predict_batches()")
        if type(limit) is not int or limit < 1:
            raise ValueError("max_samples must be positive")
        if not sources or not len(sources) == len(precursor_types) == len(
            collision_energy
        ):
            raise ValueError("Input lengths must match and be nonzero")
        # Keep every condition of a shared source in one group. Packs bound the
        # source/reaction tensors too, rather than merely slicing final scores.
        groups = {}
        for i, source in enumerate(sources):
            groups.setdefault(source.smiles, []).append(i)
        packs = []
        current = []
        for rows in groups.values():
            if current and len(current) + len(rows) > limit:
                packs.append(current)
                current = []
            current.extend(rows)
            if len(current) >= limit:
                packs.append(current)
                current = []
        if current:
            packs.append(current)
        for rows in packs:
            for output in self._predict_group_batches(
                [sources[i] for i in rows],
                [precursor_types[i] for i in rows],
                [collision_energy[i] for i in rows],
                max_samples=limit,
                precomputed=precomputed,
            ):
                yield replace(
                    output,
                    sample_input_index=tuple(
                        rows[i] for i in output.sample_input_index
                    ),
                )

    @torch.no_grad()
    def _predict_group_batches(
        self,
        sources,
        precursor_types,
        collision_energy,
        *,
        max_samples=None,
        precomputed=None,
    ):
        """Yield bounded outputs, with global input indices and shared static tokens."""
        if self.training:
            raise ValueError("Call eval() before predict_batches()")
        limit = self.max_samples if max_samples is None else max_samples
        if type(limit) is not int or limit < 1:
            raise ValueError("max_samples must be positive")
        data, unique_sources, reactions, adducts, indices = self.prepare(
            sources, precursor_types, collision_energy, precomputed=precomputed
        )
        from ..input.reaction_batching import select_samples
        from .fragment_spectrum_model import deduplicate_molecular_graphs

        device = next(self.parameters()).device
        static = self.feature_model.encode_static(data, max_graphs=limit)
        effect_cache = {}
        molecular_cache = {
            source.smiles: static[1][i] for i, source in enumerate(unique_sources)
        }
        # Never split a branch group across chunks. A group with more spectra
        # than ``limit`` is processed intact so search and its node budget are
        # still applied exactly once for (compound, main adduct).
        chunks = []
        current = []
        for group in range(data.num_branch_groups):
            rows = (
                (data.sample_branch_group_index == group).nonzero().flatten().tolist()
            )
            if current and len(current) + len(rows) > limit:
                chunks.append(current)
                current = []
            current.extend(rows)
            if len(current) >= limit:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)
        for selected in chunks:
            chunk = select_samples(data, selected).to(device)
            features = self.feature_model(chunk, static_features=static)
            selection = SourceAnchoredSelectionOutput(
                features,
                self.feature_model.decoder(
                    features.pool, features.branch_main_adduct_h
                ),
            )
            decoded = materialize_reaction_states(
                selection.decoded,
                unique_sources,
                reactions,
                chunk.branch_group_tree_index,
                self.mol_encoder.graph_builder,
                effect_cache=effect_cache,
            )
            spectrum_batch = prepare_spectrum_model_batch(
                decoded,
                self.fragmenter,
                [adducts[i] for i in selected],
                self.tensorizer,
                sample_branch_group_index=chunk.sample_branch_group_index,
                ion_type_strs=self.ion_type_strs,
            )
            spectrum_batch = deduplicate_molecular_graphs(
                spectrum_batch, tuple(source.smiles for source in unique_sources)
            )
            graph_keys = [
                key
                for key, index in zip(
                    spectrum_batch.unique_smiles,
                    spectrum_batch.unique_source_index.tolist(),
                )
                if index < 0
            ]
            graphs = spectrum_batch.unique_node_graph.to_data_list()
            missing = [
                i for i, key in enumerate(graph_keys) if key not in molecular_cache
            ]
            if missing:
                from torch_geometric.data import Batch

                # Also bound molecular graph encoding for large selected trees.
                for offset in range(0, len(missing), limit):
                    ids = missing[offset : offset + limit]
                    embeddings = self.mol_encoder(
                        Batch.from_data_list([graphs[i] for i in ids]).to(device)
                    )
                    molecular_cache.update(
                        (graph_keys[i], embeddings[j]) for j, i in enumerate(ids)
                    )
            molecular = torch.stack(
                [molecular_cache[key] for key in spectrum_batch.unique_smiles]
            )[spectrum_batch.node_graph_inverse.to(device)]
            spectrum_batch = spectrum_batch.to(device)
            spectra = self.spectrum_model(
                spectrum_batch,
                reaction_h=features.reaction_h,
                main_adduct_h=features.sample_main_adduct_h,
                collision_energy_h=features.collision_energy_h,
                source_embeddings=features.source_embeddings,
                molecular_embeddings=molecular,
            )
            yield SourceAnchoredSpectrumOutput(
                selection,
                decoded,
                spectra,
                spectrum_batch,
                tuple(indices[i] for i in selected),
            )

    @torch.no_grad()
    def predict(self, sources, precursor_types, collision_energy, *, precomputed=None):
        if len(sources) > self.max_samples:
            raise ValueError("Use predict_batches() for inputs exceeding max_samples")
        return next(
            self.predict_batches(
                sources, precursor_types, collision_energy, precomputed=precomputed
            )
        )
