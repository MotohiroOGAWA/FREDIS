"""Shared canonical inputs for training and teacher-free spectrum prediction."""

from __future__ import annotations
from dataclasses import dataclass, fields, replace
from functools import cached_property
from pathlib import Path
import torch
from torch import Tensor
from torch_geometric.data import Batch
from .source_compound import SourceCompound
from .reaction_tensor_data import ReactionTensorData
from .reaction_set import ReactionSet
from .adduct_data import AdductData
from .branch_group import BranchGroup
from .compound_batch import CompoundBatch
from .reaction_context_data import ReactionContextData
from .compound_reaction_graph import CompoundReactionGraph, ReactionExecutionGraph
from .annotated_spectrum_data import (
    SpectrumData,
    FormulaData,
    FormulaExplanationData,
    PeakAssignmentData,
    AnnotatedSpectrumData,
    SpectrumExplanationData,
    SpectrumFormulaTarget,
)
from .tensor_table import TensorTable
from ._reaction_preparation import UnresolvedPrecursorError

SCHEMA = "fredis.annotated-spectrum-training"


def csr(rows, device=None):
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long, device=device)
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0))), torch.tensor(
        [v for row in rows for v in row], dtype=torch.long, device=device
    )


def tensor(values, device=None):
    return torch.tensor(values, dtype=torch.long, device=device)


def rows(ptr, index):
    return [index[a:b].tolist() for a, b in zip(ptr[:-1], ptr[1:])]


@dataclass(frozen=True)
class SpectrumConditions(TensorTable):
    branch_group_index: Tensor
    collision_energy_eV: Tensor
    observed_adducts: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SpectrumPredictionData(TensorTable):
    source_compounds: SourceCompound
    reactions: ReactionTensorData
    reaction_sets: ReactionSet
    adducts: AdductData
    branch_groups: BranchGroup
    compounds: CompoundBatch
    compound_graph: CompoundReactionGraph
    reaction_contexts: ReactionContextData
    formulas: FormulaData
    formula_explanations: FormulaExplanationData
    spectrum_explanations: SpectrumExplanationData
    conditions: SpectrumConditions
    formula_elements: tuple[str, ...]
    max_reaction_role_count: int = 0
    source_atom_capacity: int = 0

    @property
    def source_graph(self):
        return self.source_compounds.source_graph

    @property
    def source_smiles(self):
        return self.source_compounds.source_smiles

    @property
    def sample_branch_group_index(self):
        return self.conditions.branch_group_index

    @property
    def collision_energy_eV(self):
        return self.conditions.collision_energy_eV

    @property
    def sample_tree_index(self):
        return self.branch_groups.source_compound_index[self.sample_branch_group_index]

    @property
    def num_samples(self):
        return self.sample_branch_group_index.numel()

    @property
    def num_branch_groups(self):
        return len(self.branch_groups)

    @property
    def device(self):
        return self.reactions.reaction_type.device

    def for_prediction(self):
        values = {f.name: getattr(self, f.name) for f in fields(SpectrumPredictionData)}
        values["reaction_contexts"] = replace(self.reaction_contexts, edge_label=None)
        return SpectrumPredictionData(**values)

    def without_teacher(self):
        return self.for_prediction()

    def __getstate__(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("execution_graph", "spectrum_batch")
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    @cached_property
    def execution_graph(self):
        """Expand only valid reaction-set prefixes; no chemistry or graph duplication."""
        sets = self.reaction_sets.rows()
        set_lookup = {tuple(row): i for i, row in enumerate(sets)}
        producing = [
            set(row)
            for row in rows(
                self.compounds.generating_reaction_sets_ptr,
                self.compounds.generating_reaction_sets_index,
            )
        ]
        edge_rows = rows(
            self.reaction_contexts.edge_ptr, self.reaction_contexts.edge_index
        )
        graph = self.compound_graph
        compounds, groups, sources, targets, reactions, roots, history_sets = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        conflicts = set(map(tuple, self.reactions.reaction_conflict_index.T.tolist()))
        invalidations = set(
            map(tuple, self.reactions.reaction_invalidation_index.T.tolist())
        )
        for group, edges in enumerate(edge_rows):
            root = int(
                self.source_compounds.compound_index[
                    self.reaction_contexts.root_source_compound_index[group]
                ]
            )
            lookup = {(root, ()): len(compounds)}
            compounds.append(root)
            groups.append(group)
            roots.append(len(compounds) - 1)
            history_sets.append(set_lookup[()])
            pending = [(root, ())]
            outgoing = {}
            for edge in edges:
                outgoing.setdefault(int(graph.source_compound_index[edge]), []).append(
                    edge
                )
            for compound, used in pending:
                for edge in outgoing.get(compound, []):
                    reaction = int(graph.reaction_index[edge])
                    if reaction in used or any(
                        (r, reaction) in conflicts
                        or (reaction, r) in conflicts
                        or (r, reaction) in invalidations
                        or (reaction, r) in invalidations
                        for r in used
                    ):
                        continue
                    added = tuple(sorted((*used, reaction)))
                    destination = int(graph.target_compound_index[edge])
                    set_id = set_lookup.get(added)
                    if set_id is None or set_id not in producing[destination]:
                        continue
                    key = (destination, added)
                    if key not in lookup:
                        lookup[key] = len(compounds)
                        pending.append(key)
                        compounds.append(destination)
                        groups.append(group)
                        history_sets.append(set_id)
                    sources.append(lookup[(compound, used)])
                    targets.append(lookup[key])
                    reactions.append(reaction)
        return ReactionExecutionGraph(
            *(
                tensor(v, self.device)
                for v in (
                    compounds,
                    groups,
                    sources,
                    targets,
                    reactions,
                    roots,
                    history_sets,
                )
            )
        )

    @cached_property
    def spectrum_batch(self):
        from ..specgen.fragment_spectrum_model import spectrum_batch_from_input

        return spectrum_batch_from_input(self, getattr(self, "targets", None))

    @property
    def state_reaction_count(self):
        ids = self.execution_graph.reaction_set_index
        ptr = self.reaction_sets.reaction_ptr
        return ptr[ids + 1] - ptr[ids]

    def validate(self):
        def index(name, value, size):
            if (
                value.dtype != torch.long
                or value.ndim != 1
                or (
                    value.numel() and (int(value.min()) < 0 or int(value.max()) >= size)
                )
            ):
                raise ValueError(f"Invalid foreign key: {name}")

        def check_csr(name, ptr, values, count):
            if (
                ptr.dtype != torch.long
                or ptr.shape != (count + 1,)
                or int(ptr[0]) != 0
                or int(ptr[-1]) != values.numel()
                or bool((ptr[1:] < ptr[:-1]).any())
            ):
                raise ValueError(f"Invalid CSR: {name}")

        c = len(self.compounds.compounds)
        r = self.reactions.reaction_type.shape[0]
        g = self.num_branch_groups
        s = self.num_samples
        f = len(self.formulas.mz)
        e = self.formula_explanations
        self.adducts.validate()
        self.branch_groups.validate(len(self.source_smiles), self.adducts)
        self.reaction_sets.validate(self.reactions.reaction_source_compound_index)
        if c != self.compounds.graph.num_graphs or len(set(self.compounds.smiles)) != c:
            raise ValueError("Compounds must be unique")
        index("Source->Compound", self.source_compounds.compound_index, c)
        if self.source_compounds.compound_index.shape != (len(self.source_smiles),):
            raise ValueError("Source mapping length mismatch")
        if any(
            self.compounds.smiles[int(i)] != sm
            for sm, i in zip(self.source_smiles, self.source_compounds.compound_index)
        ):
            raise ValueError("Source/Compound structures disagree")
        ctx = self.reaction_contexts
        graph = self.compound_graph
        if not torch.equal(
            ctx.root_source_compound_index, self.branch_groups.source_compound_index
        ):
            raise ValueError("Root Source differs from BranchGroup")
        for name, col, size in [
            ("edge source", graph.source_compound_index, c),
            ("edge target", graph.target_compound_index, c),
            ("edge reaction", graph.reaction_index, r),
            ("sample group", self.sample_branch_group_index, g),
            ("explanation compound", e.compound_index, c),
            ("explanation formula", e.formula_index, f),
            (
                "explanation adduct",
                e.fragment_ion_adduct_index,
                len(self.adducts.fragment_ion_adducts),
            ),
            (
                "explanation shift",
                e.hydrogen_shift_index,
                len(self.adducts.hydrogen_shifts),
            ),
        ]:
            index(name, col, size)
        n = graph.reaction_index.numel()
        if graph.source_compound_index.shape != (
            n,
        ) or graph.target_compound_index.shape != (n,):
            raise ValueError("Edge columns mismatch")
        keys = torch.stack(
            (
                graph.source_compound_index,
                graph.target_compound_index,
                graph.reaction_index,
            ),
            1,
        )
        if keys.unique(dim=0).shape[0] != n:
            raise ValueError("Duplicate structural edge")
        check_csr("group edges", ctx.edge_ptr, ctx.edge_index, g)
        index("group edge", ctx.edge_index, n)
        if ctx.edge_label is not None and (
            ctx.edge_label.dtype != torch.int8
            or ctx.edge_label.shape != ctx.edge_index.shape
            or not bool(
                torch.isin(ctx.edge_label, tensor([-1, 0, 1], self.device)).all()
            )
        ):
            raise ValueError("Invalid branch labels")
        if (
            self.collision_energy_eV.dtype != torch.float32
            or self.collision_energy_eV.shape != (s,)
            or len(self.conditions.observed_adducts) != s
        ):
            raise ValueError("Spectrum condition shape mismatch")
        if not bool(torch.isfinite(self.collision_energy_eV).all()):
            raise ValueError("Collision energies must be finite")
        check_csr(
            "spectrum explanations",
            self.spectrum_explanations.formula_explanation_ptr,
            self.spectrum_explanations.formula_explanation_index,
            s,
        )
        index(
            "spectrum explanation",
            self.spectrum_explanations.formula_explanation_index,
            len(e.formula_index),
        )
        check_csr(
            "Compound generating sets",
            self.compounds.generating_reaction_sets_ptr,
            self.compounds.generating_reaction_sets_index,
            c,
        )
        index(
            "Compound set",
            self.compounds.generating_reaction_sets_index,
            len(self.reaction_sets),
        )
        if (
            self.formulas.composition.dtype != torch.long
            or self.formulas.composition.shape != (f, len(self.formula_elements) + 1)
        ):
            raise ValueError("Formula composition must be int64 with charge last")
        if (
            self.formulas.mz.dtype != torch.float64
            or self.formulas.valid.dtype != torch.bool
            or self.formulas.valid.shape != (f,)
            or self.formulas.mz.shape != (f,)
        ):
            raise ValueError("Invalid Formula property shape/dtype")
        if not bool(torch.isfinite(self.formulas.mz).all()):
            raise ValueError("Formula mass must be finite")
        if self.source_graph.num_graphs != len(self.source_smiles):
            raise ValueError("Source graph count mismatch")
        columns = [
            e.formula_index,
            e.compound_index,
            e.fragment_ion_adduct_index,
            e.hydrogen_shift_index,
        ]
        if any(col.shape != columns[0].shape for col in columns):
            raise ValueError("Explanation columns must align")
        if torch.stack(columns, 1).unique(dim=0).shape[0] != columns[0].numel():
            raise ValueError("Duplicate formula explanation")
        if self.formulas.composition.unique(dim=0).shape[0] != f:
            raise ValueError("Duplicate Formula composition")
        execution = self.execution_graph
        reachable = [
            set(execution.compound_index[execution.branch_group_index == i].tolist())
            for i in range(g)
        ]
        for group, edge_ids in enumerate(rows(ctx.edge_ptr, ctx.edge_index)):
            if edge_ids != sorted(set(edge_ids)):
                raise ValueError("Group edges must be sorted and unique")
            ids = tensor(edge_ids, self.device)
            if not bool(
                (
                    self.reactions.reaction_source_compound_index[
                        graph.reaction_index[ids]
                    ]
                    == ctx.root_source_compound_index[group]
                ).all()
            ):
                raise ValueError("Edge Source ownership mismatch")
        for sample, ids in enumerate(
            rows(
                self.spectrum_explanations.formula_explanation_ptr,
                self.spectrum_explanations.formula_explanation_index,
            )
        ):
            if ids != sorted(set(ids)):
                raise ValueError("Spectrum explanations must be sorted and unique")
            group = int(self.sample_branch_group_index[sample])
            main = int(self.branch_groups.main_adduct_index[group])
            first, last = self.adducts.fragment_ion_adduct_ptr[main : main + 2]
            allowed = set(self.adducts.fragment_ion_adduct_index[first:last].tolist())
            for eid in ids:
                if (
                    int(e.compound_index[eid]) not in reachable[group]
                    or int(e.fragment_ion_adduct_index[eid]) not in allowed
                ):
                    raise ValueError(
                        "Spectrum explanation is not admissible in its group"
                    )
        return self

    def select_samples(self, indices):
        ids = tensor(indices, self.device)
        if ids.ndim != 1 or (
            ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.num_samples)
        ):
            raise ValueError("Invalid spectrum selection")
        selected_groups = self.sample_branch_group_index[ids].unique(sorted=True)
        group_map = torch.full(
            (self.num_branch_groups,), -1, dtype=torch.long, device=self.device
        )
        group_map[selected_groups] = torch.arange(
            selected_groups.numel(), device=self.device
        )
        ctx = self.reaction_contexts
        edge_rows = rows(ctx.edge_ptr, ctx.edge_index)
        ep, ei = csr([edge_rows[i] for i in selected_groups.tolist()], self.device)
        labels = (
            None
            if ctx.edge_label is None
            else torch.cat(
                [
                    ctx.edge_label[ctx.edge_ptr[i] : ctx.edge_ptr[i + 1]]
                    for i in selected_groups.tolist()
                ]
            )
        )
        exp = rows(
            self.spectrum_explanations.formula_explanation_ptr,
            self.spectrum_explanations.formula_explanation_index,
        )
        xp, xi = csr([exp[i] for i in ids.tolist()], self.device)
        values = dict(
            branch_groups=BranchGroup(
                self.branch_groups.source_compound_index[selected_groups],
                self.branch_groups.main_adduct_index[selected_groups],
            ),
            reaction_contexts=ReactionContextData(
                ctx.root_source_compound_index[selected_groups], ep, ei, labels
            ),
            conditions=SpectrumConditions(
                group_map[self.sample_branch_group_index[ids]],
                self.collision_energy_eV[ids],
                tuple(self.conditions.observed_adducts[i] for i in ids.tolist()),
            ),
            spectrum_explanations=SpectrumExplanationData(xp, xi),
        )
        if isinstance(self, SourceReactionStructure):
            a = self.annotations
            if a is not None:
                peak_rows = [
                    list(
                        range(
                            int(a.spectra.peak_ptr[i]), int(a.spectra.peak_ptr[i + 1])
                        )
                    )
                    for i in ids.tolist()
                ]
                pp, pi = csr(peak_rows, self.device)
                old = rows(
                    a.assignments.formula_explanation_ptr,
                    a.assignments.formula_explanation_index,
                )
                ap, ai = csr([old[i] for i in pi.tolist()], self.device)
                spectra = replace(
                    a.spectra,
                    peak_ptr=pp,
                    mz=a.spectra.mz[pi],
                    intensity=a.spectra.intensity[pi],
                    precursor=a.spectra.precursor[pi],
                    branch_group_index=values["conditions"].branch_group_index,
                    precursor_mz=a.spectra.precursor_mz[ids],
                    collision_energy=a.spectra.collision_energy[ids],
                    metadata=tuple(a.spectra.metadata[i] for i in ids.tolist()),
                )
                values["annotations"] = AnnotatedSpectrumData(
                    spectra, PeakAssignmentData(ap, ai)
                )
            values["targets"] = None
        result = replace(self, **values)
        return (
            result.with_targets()
            if isinstance(result, SourceReactionStructure)
            else result
        )

    @classmethod
    def from_structures(cls, items):
        if not items:
            raise ValueError("Cannot collate an empty batch")
        if any(isinstance(a, SourceReactionStructure) for a in items) and not all(
            isinstance(a, SourceReactionStructure) for a in items
        ):
            raise ValueError(
                "Cannot mix training and prediction batches; call for_prediction explicitly"
            )
        if any(
            a.formula_elements != items[0].formula_elements
            or not a.adducts.same_vocabulary(items[0].adducts)
            for a in items
        ):
            raise ValueError("Chemistry vocabularies must match")
        device = items[0].device
        compounds = []
        graphs = []
        lookup = {}
        cmaps = []
        reaction_offsets = []
        ro = 0
        for a in items:
            local = []
            local_graphs = a.compounds.graph.to_data_list()
            for i, compound in enumerate(a.compounds.compounds):
                if compound.smiles not in lookup:
                    lookup[compound.smiles] = len(compounds)
                    compounds.append(compound)
                    graphs.append(local_graphs[i])
                local.append(lookup[compound.smiles])
            cmaps.append(tensor(local, device))
            reaction_offsets.append(ro)
            ro += a.reactions.reaction_type.shape[0]
        rsets, smaps = ReactionSet.collate(
            [a.reaction_sets for a in items], reaction_offsets
        )
        generated = [set() for _ in compounds]
        formula_rows = []
        fmlookup = {}
        fmaps = []
        masses = []
        valid = []
        explanations = []
        elookup = {}
        emaps = []
        for a, cm, sm in zip(items, cmaps, smaps):
            for i, rs in enumerate(
                rows(
                    a.compounds.generating_reaction_sets_ptr,
                    a.compounds.generating_reaction_sets_index,
                )
            ):
                generated[int(cm[i])].update(sm[tensor(rs, device)].tolist())
            fm = []
            for i, row in enumerate(a.formulas.composition.tolist()):
                key = tuple(row)
                if key not in fmlookup:
                    fmlookup[key] = len(formula_rows)
                    formula_rows.append(row)
                    masses.append(float(a.formulas.mz[i]))
                    valid.append(bool(a.formulas.valid[i]))
                fm.append(fmlookup[key])
            fm = tensor(fm, device)
            fmaps.append(fm)
            em = []
            e = a.formula_explanations
            for i in range(len(e.formula_index)):
                key = (
                    int(fm[e.formula_index[i]]),
                    int(cm[e.compound_index[i]]),
                    int(e.fragment_ion_adduct_index[i]),
                    int(e.hydrogen_shift_index[i]),
                )
                if key not in elookup:
                    elookup[key] = len(explanations)
                    explanations.append(key)
                em.append(elookup[key])
            emaps.append(tensor(em, device))
        source_graphs = []
        source_smiles = []
        source_compounds = []
        edges = []
        edge_lookup = {}
        group_sources = []
        group_adducts = []
        roots = []
        ctx_rows = []
        labels = []
        conditions = []
        energies = []
        observed_adducts = []
        exp_rows = []
        so = go = 0
        for a, cm, em, ro in zip(items, cmaps, emaps, reaction_offsets):
            source_graphs.extend(a.source_graph.to_data_list())
            source_smiles.extend(a.source_smiles)
            source_compounds.extend(cm[a.source_compounds.compound_index].tolist())
            edge_map = []
            for src, dst, r in zip(
                a.compound_graph.source_compound_index.tolist(),
                a.compound_graph.target_compound_index.tolist(),
                a.compound_graph.reaction_index.tolist(),
            ):
                key = (int(cm[src]), int(cm[dst]), r + ro)
                if key not in edge_lookup:
                    edge_lookup[key] = len(edges)
                    edges.append(key)
                edge_map.append(edge_lookup[key])
            for group, old in enumerate(
                rows(a.reaction_contexts.edge_ptr, a.reaction_contexts.edge_index)
            ):
                pairs = sorted(
                    (
                        edge_map[e],
                        (
                            int(
                                a.reaction_contexts.edge_label[
                                    int(a.reaction_contexts.edge_ptr[group]) + j
                                ]
                            )
                            if a.reaction_contexts.edge_label is not None
                            else -1
                        ),
                    )
                    for j, e in enumerate(old)
                )
                ctx_rows.append([e for e, l in pairs])
                labels.extend(l for e, l in pairs)
            group_sources.extend((a.branch_groups.source_compound_index + so).tolist())
            group_adducts.extend(a.branch_groups.main_adduct_index.tolist())
            roots.extend((a.reaction_contexts.root_source_compound_index + so).tolist())
            conditions.extend((a.sample_branch_group_index + go).tolist())
            energies.extend(a.collision_energy_eV.tolist())
            observed_adducts.extend(a.conditions.observed_adducts)
            exp_rows.extend(
                sorted(em[tensor(row, device)].tolist())
                for row in rows(
                    a.spectrum_explanations.formula_explanation_ptr,
                    a.spectrum_explanations.formula_explanation_index,
                )
            )
            so += len(a.source_smiles)
            go += a.num_branch_groups
        cp, ci = csr([sorted(row) for row in generated], device)
        ep, ei = csr(ctx_rows, device)
        xp, xi = csr(exp_rows, device)
        ec = tensor(edges, device).reshape(-1, 3)
        fe = tensor(explanations, device).reshape(-1, 4)
        kwargs = dict(
            source_compounds=SourceCompound(
                Batch.from_data_list(source_graphs),
                tuple(source_smiles),
                items[0].source_compounds.symbols,
                tensor(source_compounds, device),
            ),
            reactions=ReactionTensorData.collate(
                [a.reactions for a in items],
                [a.source_graph.num_nodes for a in items],
                [len(a.source_smiles) for a in items],
            ),
            reaction_sets=rsets,
            adducts=items[0].adducts,
            branch_groups=BranchGroup(
                tensor(group_sources, device), tensor(group_adducts, device)
            ),
            compounds=CompoundBatch(
                tuple(compounds), Batch.from_data_list(graphs), cp, ci
            ),
            compound_graph=CompoundReactionGraph(*ec.unbind(1)),
            reaction_contexts=ReactionContextData(
                tensor(roots, device),
                ep,
                ei,
                (
                    None
                    if all(a.reaction_contexts.edge_label is None for a in items)
                    else torch.tensor(labels, dtype=torch.int8, device=device)
                ),
            ),
            formulas=FormulaData(
                tensor(formula_rows, device),
                torch.tensor(masses, dtype=torch.float64, device=device),
                torch.tensor(valid, dtype=torch.bool, device=device),
            ),
            formula_explanations=FormulaExplanationData(*fe.unbind(1)),
            spectrum_explanations=SpectrumExplanationData(xp, xi),
            conditions=SpectrumConditions(
                tensor(conditions, device),
                torch.tensor(energies, dtype=torch.float32, device=device),
                tuple(observed_adducts),
            ),
            formula_elements=items[0].formula_elements,
            source_atom_capacity=max(a.source_atom_capacity for a in items),
            max_reaction_role_count=max(a.max_reaction_role_count for a in items),
        )
        training = all(isinstance(a, SourceReactionStructure) for a in items)
        if training:
            peak_rows = []
            mz = []
            intensity = []
            precursor = []
            pm = []
            ce = []
            meta = []
            assign = []
            for a, em in zip(items, emaps):
                sp = a.annotations.spectra
                for first, last in zip(sp.peak_ptr[:-1], sp.peak_ptr[1:]):
                    peak_rows.append(
                        list(range(len(mz) + int(first), len(mz) + int(last)))
                    )
                mz.extend(sp.mz.tolist())
                intensity.extend(sp.intensity.tolist())
                precursor.extend(sp.precursor.tolist())
                pm.extend(sp.precursor_mz.tolist())
                ce.extend(sp.collision_energy.tolist())
                meta.extend(sp.metadata)
                assign.extend(
                    sorted(em[tensor(row, device)].tolist())
                    for row in rows(
                        a.annotations.assignments.formula_explanation_ptr,
                        a.annotations.assignments.formula_explanation_index,
                    )
                )
            pp, _ = csr(peak_rows, device)
            ap, ai = csr(assign, device)
            kwargs["annotations"] = AnnotatedSpectrumData(
                SpectrumData(
                    pp,
                    torch.tensor(mz, dtype=torch.float64, device=device),
                    torch.tensor(intensity, dtype=torch.float32, device=device),
                    torch.tensor(precursor, dtype=torch.bool, device=device),
                    tensor(conditions, device),
                    torch.tensor(pm, dtype=torch.float64, device=device),
                    torch.tensor(ce, dtype=torch.float64, device=device),
                    tuple(meta),
                ),
                PeakAssignmentData(ap, ai),
            )
        result = (SourceReactionStructure if training else SpectrumPredictionData)(
            **kwargs
        )
        return result.with_targets() if training else result


@dataclass(frozen=True, kw_only=True)
class SourceReactionStructure(SpectrumPredictionData):
    annotations: AnnotatedSpectrumData | None = None
    targets: SpectrumFormulaTarget | None = None

    def validate(self):
        super().validate()
        if self.annotations is None or self.targets is None:
            raise ValueError("Training data requires observations and targets")
        if self.reaction_contexts.edge_label is None:
            raise ValueError("Training data requires aligned edge labels")
        a = self.annotations
        s = a.spectra
        if not torch.equal(
            s.branch_group_index, self.sample_branch_group_index
        ) or not torch.equal(s.collision_energy.float(), self.collision_energy_eV):
            raise ValueError("Observed spectrum conditions disagree with model inputs")
        if (
            s.peak_ptr.dtype != torch.long
            or s.peak_ptr.shape != (self.num_samples + 1,)
            or int(s.peak_ptr[0]) != 0
            or int(s.peak_ptr[-1]) != s.mz.numel()
            or bool((s.peak_ptr.diff() < 0).any())
        ):
            raise ValueError("Invalid observed peak CSR")
        for name, dtype in [
            ("mz", torch.float64),
            ("intensity", torch.float32),
            ("precursor", torch.bool),
        ]:
            values = getattr(s, name)
            if values.shape != s.mz.shape or values.dtype != dtype:
                raise ValueError("Observed peak column shape/dtype mismatch")
        if (
            not bool(torch.isfinite(s.mz).all())
            or not bool(torch.isfinite(s.intensity).all())
            or bool((s.intensity < 0).any())
        ):
            raise ValueError("Observed peaks must be finite with nonnegative intensity")
        if len(s.metadata) != self.num_samples or s.precursor_mz.shape != (
            self.num_samples,
        ):
            raise ValueError("Spectrum metadata length mismatch")
        ptr = a.assignments.formula_explanation_ptr
        index = a.assignments.formula_explanation_index
        if (
            ptr.dtype != torch.long
            or ptr.shape != (s.mz.numel() + 1,)
            or int(ptr[0]) != 0
            or int(ptr[-1]) != index.numel()
            or bool((ptr.diff() < 0).any())
        ):
            raise ValueError("Invalid peak assignment CSR")
        if index.dtype != torch.long or (
            index.numel()
            and (
                int(index.min()) < 0
                or int(index.max()) >= len(self.formula_explanations.formula_index)
            )
        ):
            raise ValueError("Invalid peak explanation reference")
        for row in rows(ptr, index):
            if row != sorted(set(row)):
                raise ValueError("Peak assignments must be sorted and unique")
        for sample in range(self.num_samples):
            first, last = s.peak_ptr[sample : sample + 2]
            assigned = index[ptr[first] : ptr[last]]
            a0, b0 = self.spectrum_explanations.formula_explanation_ptr[
                sample : sample + 2
            ]
            if not bool(
                torch.isin(
                    assigned,
                    self.spectrum_explanations.formula_explanation_index[a0:b0],
                ).all()
            ):
                raise ValueError("Peak explanation is not a candidate for its spectrum")
        t = self.targets
        n = t.intensity.numel()
        for column, dtype in [
            (t.spectrum_index, torch.long),
            (t.formula_index, torch.long),
            (t.observed, torch.bool),
            (t.intensity, torch.float32),
            (t.supervision_mask, torch.bool),
        ]:
            if column.shape != (n,) or column.dtype != dtype:
                raise ValueError("Target columns must align")
        if n and (
            int(t.spectrum_index.min()) < 0
            or int(t.spectrum_index.max()) >= self.num_samples
            or int(t.formula_index.min()) < 0
            or int(t.formula_index.max()) >= len(self.formulas.mz)
        ):
            raise ValueError("Invalid target reference")
        if not bool(torch.isfinite(t.intensity).all()) or bool((t.intensity < 0).any()):
            raise ValueError("Invalid target intensity")
        if (
            torch.stack((t.spectrum_index, t.formula_index), 1).unique(dim=0).shape[0]
            != n
        ):
            raise ValueError("Duplicate spectrum/formula target")
        return self

    def with_targets(self):
        if self.annotations is None:
            return self
        a = self.annotations
        e = self.formula_explanations
        assignments = rows(
            a.assignments.formula_explanation_ptr,
            a.assignments.formula_explanation_index,
        )
        observed = {}
        for sample, (first, last) in enumerate(
            zip(a.spectra.peak_ptr[:-1], a.spectra.peak_ptr[1:])
        ):
            for peak in range(int(first), int(last)):
                for explanation in assignments[peak]:
                    key = (sample, int(e.formula_index[explanation]))
                    observed[key] = max(
                        observed.get(key, 0.0), float(a.spectra.intensity[peak])
                    )
        keys = sorted(
            {
                (sample, int(e.formula_index[i]))
                for sample, ids in enumerate(
                    rows(
                        self.spectrum_explanations.formula_explanation_ptr,
                        self.spectrum_explanations.formula_explanation_index,
                    )
                )
                for i in ids
            }
        )
        si = tensor([s for s, f in keys], self.device)
        fi = tensor([f for s, f in keys], self.device)
        target = SpectrumFormulaTarget(
            si,
            fi,
            torch.tensor(
                [k in observed for k in keys], dtype=torch.bool, device=self.device
            ),
            torch.tensor(
                [observed.get(k, 0.0) for k in keys],
                dtype=torch.float32,
                device=self.device,
            ),
            self.formulas.valid[fi] & (self.formulas.mz[fi] > 0),
        )
        return replace(self, targets=target)

    @classmethod
    def from_preparation(cls, raw):
        a = raw.annotations
        h = raw.states
        decoded = raw.spectrum_batch.decoded
        occurrence = a.occurrences.compound_index
        active = a.contexts.compound_index[h.candidate_index]
        groups = [set() for _ in range(raw.num_branch_groups)]
        positive = [set() for _ in groups]
        for (src, dst), reaction in zip(
            decoded.edge_index.T.tolist(), decoded.edge_added_reaction_index.tolist()
        ):
            if reaction >= 0:
                groups[int(decoded.node_sample_index[src])].add(
                    (int(occurrence[src]), int(occurrence[dst]), reaction)
                )
        for parent, child, reaction in zip(
            h.transition_parent_index.tolist(),
            h.transition_child_index.tolist(),
            h.transition_reaction_index.tolist(),
        ):
            key = (int(active[parent]), int(active[child]), reaction)
            group = int(h.branch_group_index[parent])
            groups[group].add(key)
            positive[group].add(key)
        keys = sorted(set().union(*groups))
        lookup = {key: i for i, key in enumerate(keys)}
        ep, ei = csr([sorted(lookup[key] for key in group) for group in groups])
        labels = torch.tensor(
            [
                1 if keys[i] in positive[g] else 0
                for g, row in enumerate(rows(ep, ei))
                for i in row
            ],
            dtype=torch.int8,
        )
        fe = a.formula_explanations
        xp, xi = csr(
            [
                sorted(
                    set(
                        a.runtime_explanations.formula_explanation_index[
                            a.runtime_explanations.sample_index == sample
                        ].tolist()
                    )
                )
                for sample in range(raw.num_samples)
            ]
        )
        sc = replace(
            raw.source_compounds,
            compound_index=tensor(
                [a.compounds.smiles.index(sm) for sm in raw.source_smiles]
            ),
        )
        edge = tensor(keys).reshape(-1, 3)
        result = cls(
            source_compounds=sc,
            reactions=raw.reactions,
            reaction_sets=raw.reaction_sets,
            adducts=raw.adducts,
            branch_groups=raw.branch_groups,
            compounds=a.compounds,
            compound_graph=CompoundReactionGraph(*edge.unbind(1)),
            reaction_contexts=ReactionContextData(
                raw.branch_groups.source_compound_index, ep, ei, labels
            ),
            formulas=a.formulas,
            formula_explanations=FormulaExplanationData(
                fe.formula_index,
                fe.compound_index,
                fe.fragment_ion_adduct_index,
                fe.hydrogen_shift_index,
            ),
            spectrum_explanations=SpectrumExplanationData(xp, xi),
            conditions=SpectrumConditions(
                raw.sample_branch_group_index,
                raw.collision_energy_eV,
                tuple(
                    a.spectra.metadata[i]["observed_adduct"]
                    for i in range(raw.num_samples)
                ),
            ),
            formula_elements=raw.formula_elements,
            max_reaction_role_count=raw.max_reaction_role_count,
            source_atom_capacity=raw.source_atom_capacity,
            annotations=AnnotatedSpectrumData(
                a.spectra,
                PeakAssignmentData(
                    a.assignments.formula_explanation_ptr,
                    a.assignments.formula_explanation_index,
                ),
            ),
        )
        return result.with_targets()

    def save(self, path):
        self.validate()
        torch.save(dict(schema=SCHEMA, structure=self.to("cpu")), path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != SCHEMA
            or not isinstance(payload.get("structure"), cls)
        ):
            raise ValueError("Expected an annotated-spectrum-training payload")
        payload["structure"].validate()
        return payload["structure"].to(device)

    @property
    def sample_annotations(self):
        if self.annotations is None:
            return ()
        from fredis.libs.mmkit.mmkit import Formula

        a = self.annotations
        out = []
        e = self.formula_explanations
        assignments = rows(
            a.assignments.formula_explanation_ptr,
            a.assignments.formula_explanation_index,
        )
        for sample, (first, last) in enumerate(
            zip(a.spectra.peak_ptr[:-1], a.spectra.peak_ptr[1:])
        ):
            group = int(self.sample_branch_group_index[sample])
            main = self.adducts.main_adducts[
                int(self.branch_groups.main_adduct_index[group])
            ]
            peaks = []
            for local, peak in enumerate(range(int(first), int(last))):
                matches = []
                for eid in assignments[peak]:
                    fid = int(e.formula_index[eid])
                    cid = int(e.compound_index[eid])
                    counts = self.formulas.composition[fid].tolist()
                    mz = float(self.formulas.mz[fid])
                    observed = float(a.spectra.mz[peak])
                    label = str(
                        Formula(
                            elements={
                                s: int(v)
                                for s, v in zip(self.formula_elements, counts[:-1])
                                if v
                            },
                            charge=int(counts[-1]),
                        )
                    )
                    matches.append(
                        dict(
                            compoundIndex=cid,
                            nodeIds=[cid],
                            formulaExplanationIndex=eid,
                            smiles=self.compounds.smiles[cid],
                            formula=label,
                            theoreticalMz=mz,
                            massErrorPpm=(observed - mz) / mz * 1e6 if mz else 0.0,
                            fragmentIonAdduct=self.adducts.fragment_ion_adducts[
                                int(e.fragment_ion_adduct_index[eid])
                            ],
                            hydrogenShift=int(
                                self.adducts.hydrogen_shifts[
                                    e.hydrogen_shift_index[eid]
                                ]
                            ),
                        )
                    )
                peaks.append(
                    dict(
                        index=local,
                        mz=float(a.spectra.mz[peak]),
                        intensity=float(a.spectra.intensity[peak]),
                        precursor=bool(a.spectra.precursor[peak]),
                        matches=matches,
                    )
                )
            meta = {
                k: v
                for k, v in a.spectra.metadata[sample].items()
                if k != "observed_adduct"
            }
            out.append(
                dict(
                    meta,
                    adduct=self.conditions.observed_adducts[sample],
                    mainAdduct=main,
                    collisionEnergy=float(a.spectra.collision_energy[sample]),
                    precursorMz=float(a.spectra.precursor_mz[sample]),
                    peaks=peaks,
                )
            )
        return tuple(out)


def make_structure_file_stem(smiles, *, index):
    import hashlib

    return f"smiles_{index:06d}_{hashlib.sha1(smiles.encode('utf-8')).hexdigest()[:16]}"


def save_fragment_tree_structure(*, structure, output_file, metadata=None):
    structure.validate()
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            schema=SCHEMA,
            structure=structure.to("cpu"),
            metadata={
                k: v for k, v in (metadata or {}).items() if k != "sample_annotations"
            },
        ),
        path,
    )
