"""Ion-condition states and CE-conditioned spectrum model."""

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
    """One row per structural node, ion condition and spectrum sample.

    Conditions stay separate before prediction even when formulas are equal.
    Formula tensors and m/z are annotations, not transition inputs.
    """

    decoded: DecodedFragmentTreeBatch
    state_node_index: Tensor
    state_sample_index: Tensor
    state_formula_tensor: Tensor
    state_mz: Tensor
    state_adduct_index: Tensor
    state_hydrogen_shift: Tensor
    target_intensity: Tensor | None = None
    precursor_adducts: tuple[str, ...] = ()
    ion_type_strs: tuple[str, ...] = ()
    unique_node_graph: Batch | None = None
    node_graph_inverse: Tensor | None = None
    unique_source_index: Tensor | None = None
    unique_smiles: tuple[str, ...] = ()
    node_smiles: tuple[str, ...] = ()

    @property
    def state_adducts(self) -> tuple[str, ...]:
        """Derive display labels without storing redundant per-state strings."""
        labels = {}
        result = []
        for index, shift in zip(
            self.state_adduct_index.tolist(), self.state_hydrogen_shift.tolist()
        ):
            key = (index, shift)
            if key not in labels:
                labels[key] = str(
                    Adduct.parse(self.ion_type_strs[index]).add_prefer_self(
                        Adduct.from_dict({"H": shift})
                    )
                )
            result.append(labels[key])
        return tuple(result)

    def to(self, device: str | torch.device):
        names = (
            "state_node_index",
            "state_sample_index",
            "state_formula_tensor",
            "state_mz",
            "state_adduct_index",
            "state_hydrogen_shift",
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
    """State predictions and optional sparse transition diagnostics.

    Transition endpoints index state rows. Reaction index -1 denotes persistence.
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
    adduct_indices = []
    hydrogen_shifts = []
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
            for node in range(tree.num_nodes):
                for ion in rule.fragment_ion_adducts:
                    for shift in fragmenter.adduct_rule_set.hydrogen_shifts:
                        nodes.append(node_offset + node)
                        samples.append(sample)
                        adduct_indices.append(ion_to_index[str(ion)])
                        hydrogen_shifts.append(shift)
        node_offset += tree.num_nodes
    result = SpectrumModelBatch(
        decoded=decoded,
        state_node_index=torch.tensor(nodes, dtype=torch.long),
        state_sample_index=torch.tensor(samples, dtype=torch.long),
        state_formula_tensor=torch.empty((len(nodes), tensorizer.dim)),
        state_mz=torch.empty(len(nodes), dtype=torch.float64),
        state_adduct_index=torch.tensor(adduct_indices, dtype=torch.long),
        state_hydrogen_shift=torch.tensor(hydrogen_shifts, dtype=torch.long),
        precursor_adducts=tuple(str(a) for a in precursor_types),
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
    for row, node in enumerate(data.state_node_index.tolist()):
        ion = Adduct.parse(
            data.ion_type_strs[int(data.state_adduct_index[row])]
        )
        shift = Adduct.from_dict({"H": int(data.state_hydrogen_shift[row])})
        formula = ion.apply_to_formula(
            shift.apply_to_formula(data.decoded.compounds[node].formula)
        ).normalized
        formulas.append(formula)
        mzs.append(formula.exact_mass / abs(ion.charge))
    device = data.state_node_index.device
    return replace(
        data,
        state_formula_tensor=tensorizer.formulas_to_tensor(
            formulas, device=device
        ),
        state_mz=torch.tensor(
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
    adduct_indices = []
    hydrogen_shifts = []
    targets = []
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
        nodes.append(item.state_node_index + node_offset)
        samples.append(item.state_sample_index + sample_offset)
        formulas.append(item.state_formula_tensor)
        mzs.append(item.state_mz)
        remap = torch.tensor(
            [ion_index[value] for value in item.ion_type_strs],
            dtype=torch.long, device=item.state_adduct_index.device,
        )
        adduct_indices.append(remap[item.state_adduct_index])
        hydrogen_shifts.append(item.state_hydrogen_shift)
        if item.target_intensity is not None:
            targets.append(item.target_intensity)
        node_smiles.extend(item.node_smiles)
        node_offset += len(d.compounds)
        sample_offset += (
            int(item.state_sample_index.max()) + 1
            if item.state_sample_index.numel()
            else 0
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
        state_node_index=torch.cat(nodes),
        state_sample_index=torch.cat(samples),
        state_formula_tensor=torch.cat(formulas),
        state_mz=torch.cat(mzs),
        state_adduct_index=torch.cat(adduct_indices),
        state_hydrogen_shift=torch.cat(hydrogen_shifts),
        target_intensity=torch.cat(targets) if targets else None,
        precursor_adducts=tuple(
            a for item in items for a in getattr(item, "precursor_adducts", ())
        ),
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
