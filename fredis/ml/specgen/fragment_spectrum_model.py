"""Prepared physical-ion candidates and CE-conditioned intensity model."""

from __future__ import annotations
import math
from dataclasses import dataclass, replace
from collections.abc import Sequence
import torch
from torch import Tensor
from torch_geometric.data import Batch
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Adduct
from ..mol.formula_encoder import FormulaTensorizer
from .materialization import DecodedFragmentTreeBatch


@dataclass(frozen=True)
class SpectrumModelBatch:
    """Ion candidates attached to the decoded structural graph.

    Candidate rows reference structural nodes and spectrum samples independently:
    several CE conditions may share one structural graph. The explanation pointer
    is CSR storage: explanations for candidate ``i`` occupy ``ptr[i]:ptr[i + 1]``.
    Formula tensors store elemental counts followed by charge.
    """

    decoded: DecodedFragmentTreeBatch
    physical_candidate_node_index: Tensor
    physical_candidate_sample_index: Tensor
    physical_candidate_formula_tensor: Tensor
    physical_candidate_mz: Tensor
    physical_candidate_charge: Tensor
    physical_candidate_explanation_ptr: Tensor
    explanation_adduct_index: Tensor
    explanation_hydrogen_shift: Tensor
    target_intensity: Tensor | None = None
    precursor_adducts: tuple[str, ...] = ()
    physical_candidate_adduct: tuple[str, ...] = ()
    ion_type_strs: tuple[str, ...] = ()
    unique_node_graph: Batch | None = None
    node_graph_inverse: Tensor | None = None
    unique_source_index: Tensor | None = None
    unique_smiles: tuple[str, ...] = ()
    node_smiles: tuple[str, ...] = ()

    def to(self, device: str | torch.device):
        names = (
            "physical_candidate_node_index",
            "physical_candidate_sample_index",
            "physical_candidate_formula_tensor",
            "physical_candidate_mz",
            "physical_candidate_charge",
            "physical_candidate_explanation_ptr",
            "explanation_adduct_index",
            "explanation_hydrogen_shift",
            "target_intensity",
            "node_graph_inverse",
            "unique_source_index",
        )
        return replace(
            self,
            decoded=self.decoded.to(device),
            unique_node_graph=(
                self.unique_node_graph.to(device)
                if self.unique_node_graph is not None
                else None
            ),
            **{
                name: getattr(self, name).to(device)
                for name in names
                if getattr(self, name) is not None
            },
        )


@dataclass(frozen=True)
class SpectrumModelOutput:
    """Candidate predictions and optional sparse transition diagnostics.

    Transition endpoints index candidate rows. Explanation and reaction indices
    use -1 for persistence events, which leave an ion in its current state.
    """

    intensity: Tensor
    sample_index: Tensor
    formula_tensor: Tensor
    mz: Tensor
    loss: Tensor
    confidence: Tensor
    ion_probability: Tensor
    metrics: dict[str, Tensor]
    transition_source_index: Tensor | None = None
    transition_target_index: Tensor | None = None
    transition_probability: Tensor | None = None
    transition_explanation_index: Tensor | None = None
    transition_reaction_index: Tensor | None = None


@dataclass(frozen=True)
class SpectrumLossSettings:
    """Training objective of :class:`CompetitiveSpectrumModel`; not part of the architecture."""

    intensity_loss_weight: float = 1.0
    cosine_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in ("intensity_loss_weight", "cosine_loss_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def prepare_spectrum_model_batch(
    decoded: DecodedFragmentTreeBatch,
    fragmenter: Fragmenter,
    precursor_types: Sequence[Adduct],
    tensorizer: FormulaTensorizer,
    sample_branch_group_index: Tensor | None = None,
    ion_type_strs: Sequence[str] | None = None,
    annotate: bool = True,
) -> SpectrumModelBatch:
    """Prepare all ion-condition states; mmkit formula/mz are annotation only."""
    sample_count = len(precursor_types)
    sample_branch_group_index = torch.as_tensor(
        (
            sample_branch_group_index
            if sample_branch_group_index is not None
            else torch.arange(sample_count)
        ),
        dtype=torch.long,
    )
    ion_vocab = list(
        dict.fromkeys(
            ion_type_strs
            or [
                str(value)
                for rule in fragmenter.adduct_rule_set.adduct_rules
                for value in rule.fragment_ion_adducts
            ]
        )
    )
    ion_to_index = {value: i for i, value in enumerate(ion_vocab)}
    nodes = []
    samples = []
    charges = []
    explanation_rows = []
    adducts = []
    node_offset = 0
    for branch_group, tree in enumerate(decoded.trees):
        group_samples = (
            (sample_branch_group_index == branch_group).nonzero().flatten().tolist()
        )
        if not group_samples:
            node_offset += tree.num_nodes
            continue
        main = fragmenter._resolve_main_adduct_type(precursor_types[group_samples[0]])
        rule = fragmenter.adduct_rule_set.get_rule_by_adduct_type(main)
        # State topology is the full ion x shift grid, independent of formula.
        for sample in group_samples:
            for node, compound in enumerate(
                decoded.compounds[node_offset : node_offset + tree.num_nodes]
            ):
                for ion in rule.fragment_ion_adducts:
                    for shift in fragmenter.adduct_rule_set.hydrogen_shifts:
                        hydrogen = Adduct.from_dict({"H": shift})
                        net = ion.add_prefer_self(hydrogen)
                        nodes.append(node_offset + node)
                        samples.append(sample)
                        charges.append(net.charge)
                        adducts.append(str(net))
                        explanation_rows.append(((ion_to_index[str(ion)], shift),))
        node_offset += tree.num_nodes
    explanation_ptr, flat = __import__(
        "fredis.ml.input.source_reaction_structure", fromlist=["csr"]
    ).csr(explanation_rows)
    pairs = (
        flat.reshape(-1, 2) if flat.numel() else torch.empty((0, 2), dtype=torch.long)
    )
    result = SpectrumModelBatch(
        decoded=decoded,
        physical_candidate_node_index=torch.tensor(nodes, dtype=torch.long),
        physical_candidate_sample_index=torch.tensor(samples, dtype=torch.long),
        physical_candidate_formula_tensor=torch.empty((len(nodes), tensorizer.dim)),
        physical_candidate_mz=torch.empty(len(nodes), dtype=torch.float64),
        physical_candidate_charge=torch.tensor(charges, dtype=torch.long),
        physical_candidate_explanation_ptr=explanation_ptr,
        explanation_adduct_index=pairs[:, 0],
        explanation_hydrogen_shift=pairs[:, 1],
        precursor_adducts=tuple(str(a) for a in precursor_types),
        physical_candidate_adduct=tuple(adducts),
        ion_type_strs=tuple(ion_vocab),
        node_smiles=tuple(c.smiles for c in decoded.compounds),
    )
    return annotate_ion_conditions(result, tensorizer) if annotate else result


def annotate_ion_conditions(
    data: SpectrumModelBatch, tensorizer: FormulaTensorizer
) -> SpectrumModelBatch:
    """mmkit annotations only; call after scoring in inference, before targets in preparation."""
    formulas = []
    mzs = []
    starts = data.physical_candidate_explanation_ptr[:-1].tolist()
    for node, start in zip(data.physical_candidate_node_index.tolist(), starts):
        ion = Adduct.parse(
            data.ion_type_strs[int(data.explanation_adduct_index[start])]
        )
        shift = Adduct.from_dict({"H": int(data.explanation_hydrogen_shift[start])})
        formula = ion.apply_to_formula(
            shift.apply_to_formula(data.decoded.compounds[node].formula)
        ).normalized
        formulas.append(formula)
        mzs.append(formula.exact_mass / abs(ion.charge))
    device = data.physical_candidate_node_index.device
    return replace(
        data,
        physical_candidate_formula_tensor=tensorizer.formulas_to_tensor(
            formulas, device=device
        ),
        physical_candidate_mz=torch.tensor(
            mzs, dtype=torch.float64, device=device
        ),
    )


def collate_spectrum_model_batches(
    items: Sequence[SpectrumModelBatch], reaction_offsets: Sequence[int]
) -> SpectrumModelBatch:
    molecules = []
    compounds = []
    metadata = []
    edges = []
    node_groups = []
    edge_reactions = []
    terminals = []
    scores = []
    nodes = []
    samples = []
    formulas = []
    mzs = []
    charges = []
    explanations = []
    targets = []
    adducts = []
    node_smiles = []
    node_offset = sample_offset = 0
    ion_vocab = list(
        dict.fromkeys(value for item in items for value in item.ion_type_strs)
    )
    ion_index = {v: i for i, v in enumerate(ion_vocab)}
    for item, reaction_offset in zip(items, reaction_offsets):
        d = item.decoded
        molecules.extend(d.node_graph.to_data_list())
        compounds.extend(d.compounds)
        metadata.extend(d.trees)
        edges.append(d.edge_index + node_offset)
        node_groups.append(d.node_sample_index + len(metadata) - len(d.trees))
        edge_reactions.append(
            torch.where(
                d.edge_added_reaction_index >= 0,
                d.edge_added_reaction_index + reaction_offset,
                d.edge_added_reaction_index,
            )
        )
        terminals.append(d.terminal_node_index + node_offset)
        scores.append(d.node_log_score)
        nodes.append(item.physical_candidate_node_index + node_offset)
        samples.append(item.physical_candidate_sample_index + sample_offset)
        formulas.append(item.physical_candidate_formula_tensor)
        mzs.append(item.physical_candidate_mz)
        charges.append(item.physical_candidate_charge)
        for candidate in range(item.physical_candidate_node_index.numel()):
            start, stop = item.physical_candidate_explanation_ptr[
                candidate : candidate + 2
            ]
            explanations.append(
                [
                    (ion_index[item.ion_type_strs[int(i)]], int(hydrogen_shift))
                    for i, hydrogen_shift in zip(
                        item.explanation_adduct_index[start:stop],
                        item.explanation_hydrogen_shift[start:stop],
                    )
                ]
            )
        if item.target_intensity is not None:
            targets.append(item.target_intensity)
        adducts.extend(item.physical_candidate_adduct)
        node_smiles.extend(item.node_smiles)
        node_offset += len(d.compounds)
        sample_offset += (
            int(item.physical_candidate_sample_index.max()) + 1
            if item.physical_candidate_sample_index.numel()
            else 0
        )
    from ..input.source_reaction_structure import csr

    ptr, flat = csr(explanations)
    pairs = (
        flat.reshape(-1, 2) if flat.numel() else torch.empty((0, 2), dtype=torch.long)
    )
    decoded = DecodedFragmentTreeBatch(
        Batch.from_data_list(molecules),
        torch.cat(edges, 1),
        torch.cat(node_groups),
        torch.cat(edge_reactions),
        torch.cat(terminals),
        torch.cat(scores),
        tuple(metadata),
        tuple(compounds),
        sum(i.decoded.rdkit_run_count for i in items),
        sum(i.decoded.failed_effect_count for i in items),
        reaction_state_graph=all(
            getattr(i.decoded, "reaction_state_graph", False) for i in items
        ),
    )
    return SpectrumModelBatch(
        decoded=decoded,
        physical_candidate_node_index=torch.cat(nodes),
        physical_candidate_sample_index=torch.cat(samples),
        physical_candidate_formula_tensor=torch.cat(formulas),
        physical_candidate_mz=torch.cat(mzs),
        physical_candidate_charge=torch.cat(charges),
        physical_candidate_explanation_ptr=ptr,
        explanation_adduct_index=pairs[:, 0],
        explanation_hydrogen_shift=pairs[:, 1],
        target_intensity=torch.cat(targets) if targets else None,
        precursor_adducts=tuple(
            a for item in items for a in getattr(item, "precursor_adducts", ())
        ),
        physical_candidate_adduct=tuple(adducts),
        ion_type_strs=tuple(ion_vocab),
        node_smiles=tuple(node_smiles),
    )


def deduplicate_molecular_graphs(data: SpectrumModelBatch, source_smiles=()):
    graphs = data.decoded.node_graph.to_data_list()
    source_index = {s: i for i, s in enumerate(source_smiles)}
    keys = {}
    unique = []
    inverse = []
    shared = []
    if len(data.node_smiles) != len(graphs):
        raise ValueError("Molecular identities must be prepared before training")
    for graph, key in zip(graphs, data.node_smiles):
        if key not in keys:
            keys[key] = len(shared)
            shared.append(source_index.get(key, -1))
            if key not in source_index:
                unique.append(graph)
        inverse.append(keys[key])
    unique_batch = Batch.from_data_list(unique if unique else [graphs[0]])
    return replace(
        data,
        unique_node_graph=unique_batch,
        node_graph_inverse=torch.tensor(inverse),
        unique_source_index=torch.tensor(shared),
        unique_smiles=tuple(keys),
    )
