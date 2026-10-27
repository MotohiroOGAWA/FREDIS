"""Source -> incremental reaction search -> selected graph -> spectrum intensity."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import math
import torch
from torch import nn
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Adduct, Compound
from ..mol.mol_encoder import MolEncoder
from ..mol.formula_encoder import FormulaTensorizer
from .components.condition.adduct_embedding import AdductEmbeddingLayer
from .components.condition.collision_energy_feature import CollisionEnergyFeatureLayer
from .source_reaction_feature_model import (
    SourceReactionFeatureModel,
    SourceReactionFeatures,
)
from .materialization import DecodedFragmentTreeBatch
from .generation import GenerationSettings
from .fragment_spectrum_model import (
    SpectrumModelOutput,
    SpectrumModelBatch,
)


@dataclass(frozen=True)
class CompoundEdgeSelection:
    edge_index: torch.Tensor
    edge_log_probability: torch.Tensor
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
        dropout: float = 0.0,
        branch_mil: dict | None = None,
        reaction_transformer: dict | None = None,
    ) -> None:
        super().__init__()
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self.branch_mil = dict(branch_mil or {})
        if set(self.branch_mil) - {"temperature", "negative_weight", "enabled", "pooling", "depth_reduction", "peak_reduction"}:
            raise ValueError("Unknown branch_mil settings")
        for key, expected in {"enabled": True, "pooling": "softmax_weighted", "depth_reduction": "mean", "peak_reduction": "mean"}.items():
            if self.branch_mil.get(key, expected) != expected:
                raise ValueError(f"Unsupported branch_mil {key}")
        temperature = self.branch_mil.get("temperature", .1)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("MIL temperature must be finite and positive")
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
        transformer = dict(reaction_transformer or params.pop("reaction_transformer", {}) or {})
        expected_sections = {
            "grpe": {"enabled": True, "use_spatial_encoding": True, "use_edge_encoding": True,
                     "use_graph_encoded_value": True, "use_reaction_relations": True},
            "decoder": {"state_representation": "permutation_invariant", "use_bos": True,
                        "use_absolute_positional_encoding": False},
            "branch": {"output": "log_probability", "activation": "logsigmoid"},
            "spectrum": {"activation": "log_softmax", "include_self_event": True, "include_sink_event": True},
        }
        grpe = dict(transformer.pop("grpe", {}))
        if "max_spatial_distance" in grpe:
            transformer["max_spatial_distance"] = grpe.pop("max_spatial_distance")
        for section, expected in expected_sections.items():
            values = grpe if section == "grpe" else transformer.pop(section, {})
            if any(key not in expected or value != expected[key] for key, value in values.items()):
                raise ValueError(f"Unsupported reaction_transformer {section} settings")
        for key, expected in {"architecture": "grpe_encoder_decoder", "norm_first": True}.items():
            if transformer.pop(key, expected) != expected:
                raise ValueError(f"Unsupported reaction_transformer {key}")
        if "dropout" in transformer:
            dropout = transformer.pop("dropout")
        if "hidden_dim" in transformer:
            params["hidden_dim"] = transformer.pop("hidden_dim")
        params["reaction_transformer"] = transformer
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
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not fine_tuning:
            mol_params["dropout"] = dropout
        # Dropout is owned by the generator, never by individual components.
        for key in ("fragment_dropout", "state_dropout"):
            spectrum_params.pop(key, None)
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
            dropout=dropout,
            frozen_pattern_ids=frozen,
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
            dropout=dropout,
            **spectrum_params,
        )
        self.ion_type_strs = ion_types
        self.set_generation(GenerationSettings.from_dict(generation_params))
        if fine_tuning:
            from .fine_tuning import configure_pattern_training

            configure_pattern_training(self, fine_tuning)

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
                torch.nn.functional.logsigmoid(logits),
                torch.repeat_interleave(
                    torch.arange(data.num_branch_groups, device=data.device),
                    data.reaction_contexts.edge_ptr.diff(),
                ),
            ),
        )

    @torch.no_grad()
    def prepare(
        self,
        sources,
        precursor_types,
        collision_energy,
        *,
        precomputed=None,
        prepared=None,
    ):
        from .incremental_generation import prepare_incremental

        return prepare_incremental(
            self,
            sources,
            precursor_types,
            collision_energy,
            precomputed=precomputed,
            prepared=prepared,
        )

    @torch.no_grad()
    def predict_prepared_batches(self, data, *, max_samples=None):
        """Reuse saved chemistry without reading observed peaks or teacher labels."""
        sources = [
            data.compounds.compounds[
                int(data.source_compounds.compound_index[int(tree)])
            ]
            for tree in data.sample_tree_index.tolist()
        ]
        yield from self.predict_batches(
            sources,
            [Adduct.parse(a) for a in data.conditions.observed_adducts],
            data.collision_energy_eV.tolist(),
            max_samples=max_samples,
            prepared=data,
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
        prepared=None,
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
        from .incremental_generation import source_keys

        groups = {}
        for i, key in enumerate(source_keys(sources, prepared)):
            groups.setdefault(key, []).append(i)
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
                prepared=prepared,
            )
            data = data.to(device)
            static = self.feature_model.encode_static(data, max_graphs=limit)
            features = self.feature_model(data, static_features=static)
            molecular = self.feature_model.encode_compounds(
                data, features, max_graphs=limit
            )
            # prepare() has already performed probability-pruned incremental search.
            selected = data
            for start in range(0, selected.num_samples, limit):
                indices = list(range(start, min(start + limit, selected.num_samples)))
                chunk = selected.select_samples(indices)
                batch = spectrum_batch_from_input(chunk)
                chunk_features = self.feature_model(chunk, static_features=static)
                spectrum = self.spectrum_model(
                    batch,
                    reaction_h=chunk_features.local_reaction_h,
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
                        torch.nn.functional.logsigmoid(chunk_logits),
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
    def predict(
        self,
        sources,
        precursor_types,
        collision_energy,
        *,
        precomputed=None,
        prepared=None,
    ):
        if len(sources) > self.max_samples:
            raise ValueError("Use predict_batches() for inputs exceeding max_samples")
        outputs = list(
            self.predict_batches(
                sources,
                precursor_types,
                collision_energy,
                max_samples=self.max_samples,
                precomputed=precomputed,
                prepared=prepared,
            )
        )
        if len(outputs) != 1:
            raise ValueError("Use predict_batches() for multi-pack inputs")
        return outputs[0]
