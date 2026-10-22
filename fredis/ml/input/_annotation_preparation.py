"""Normalized spectrum annotations, independent of training targets.

Index spaces: S spectra, P observed peaks, C unique compounds, F unique charged
formulas, I unique formula explanations, V reaction contexts, A assignments.
Indices are global within this object, never dataset-global. CSR ptr arrays have
one more entry than their row count. Compound identity includes stereochemistry
and charge through canonical isomeric SMILES; formula identity includes charge.
"""

from __future__ import annotations
from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor
from torch_geometric.data import Batch
from ._preparation_histories import TensorTable
from .compound_batch import CompoundBatch
from .annotated_spectrum_data import (
    SpectrumData,
    FormulaData,
    FormulaExplanationData,
    PeakAssignmentData,
)


def csr(rows, device=None):
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long, device=device)
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0))), torch.tensor(
        [v for row in rows for v in row], dtype=torch.long, device=device
    )


@dataclass(frozen=True)
class FragmentOccurrenceData(TensorTable):
    # One row per topology occurrence; structures live in CompoundBatch.
    compound_index: Tensor
    branch_group_index: Tensor


@dataclass(frozen=True)
class MaterializedContextData(TensorTable):
    # One row per materialization context, not per unique Compound.
    reaction_set_index: Tensor
    compound_index: Tensor  # -1 if not materialized


@dataclass(frozen=True)
class RuntimeFormulaExplanationData(TensorTable):
    # Runtime condition row -> canonical explanation / spectrum.
    formula_explanation_index: Tensor
    sample_index: Tensor


@dataclass(frozen=True)
class AnnotatedSpectrumData:
    spectra: SpectrumData
    compounds: CompoundBatch
    formulas: FormulaData
    formula_explanations: FormulaExplanationData
    runtime_explanations: RuntimeFormulaExplanationData
    assignments: PeakAssignmentData
    # Element columns of formulas.composition, excluding the final charge column.
    elements: tuple[str, ...]
    occurrences: FragmentOccurrenceData
    contexts: MaterializedContextData

    def to(self, device):
        return replace(
            self,
            spectra=self.spectra.to(device),
            compounds=self.compounds.to(device),
            formulas=self.formulas.to(device),
            formula_explanations=self.formula_explanations.to(device),
            runtime_explanations=self.runtime_explanations.to(device),
            assignments=self.assignments.to(device),
            occurrences=self.occurrences.to(device),
            contexts=self.contexts.to(device),
        )

    @classmethod
    def from_runtime(
        cls,
        batch,
        observations,
        branch_group_index,
        elements=(),
        state_nodes=None,
        *,
        context_reaction_set_index: Tensor,
        adducts,
    ):
        """Normalize prepared chemistry; no losses, positives or negative labels are created."""
        decoded = batch.decoded
        device = batch.state_node_index.device
        graphs = decoded.node_graph.to_data_list()
        lookup, smiles, representatives, occurrence = {}, [], [], []
        for row, compound in enumerate(decoded.compounds):
            key = compound.smiles
            if key not in lookup:
                lookup[key] = len(smiles)
                smiles.append(key)
                representatives.append(row)
            occurrence.append(lookup[key])
        occurrence = torch.tensor(occurrence, dtype=torch.long, device=device)
        node_rows = (
            decoded.materialized_node_index if state_nodes is None else state_nodes
        )
        materialized = torch.where(
            node_rows >= 0, occurrence[node_rows.clamp_min(0)], node_rows
        )
        reaction_set_index = [[] for _ in smiles]
        for row, compound in enumerate(materialized.tolist()):
            if compound >= 0:
                reaction_set_index[compound].append(
                    int(context_reaction_set_index[row])
                )
        reaction_set_ptr, reaction_set_index = csr(
            [sorted(set(ids)) for ids in reaction_set_index], device
        )
        compounds = CompoundBatch(
            tuple(decoded.compounds[i] for i in representatives),
            Batch.from_data_list([graphs[i] for i in representatives]),
            reaction_set_ptr,
            reaction_set_index,
        )
        composition, formula_inverse = torch.unique(
            batch.state_formula_tensor, dim=0, return_inverse=True
        )
        fcount = len(composition)
        formula_mz = batch.state_mz.new_zeros(fcount)
        formula_mz[formula_inverse] = batch.state_mz
        fragment_map = torch.tensor(
            [adducts.fragment_ion_adducts.index(v) for v in batch.ion_type_strs],
            dtype=torch.long,
            device=device,
        )
        shift_lookup = {
            int(v): i for i, v in enumerate(adducts.hydrogen_shifts.tolist())
        }
        shift_inverse = torch.tensor(
            [shift_lookup[int(v)] for v in batch.state_hydrogen_shift.tolist()],
            dtype=torch.long,
            device=device,
        )
        raw = torch.stack(
            (
                formula_inverse,
                occurrence[batch.state_node_index],
                fragment_map[batch.state_adduct_index],
                shift_inverse,
            ),
            dim=1,
        )
        ion_rows, ion_inverse = torch.unique(raw, dim=0, return_inverse=True)
        formulas = FormulaData(
            composition.to(torch.long),
            formula_mz.to(torch.float64),
            (composition[:, :-1] >= 0).all(1),
        )
        ions = FormulaExplanationData(*ion_rows.unbind(1))
        runtime = RuntimeFormulaExplanationData(ion_inverse, batch.state_sample_index)
        # Match concrete annotations, preserving ambiguity rather than assigning by m/z alone.
        ion_lookup = {}
        for row, ion in enumerate(ion_inverse.tolist()):
            key = (
                int(batch.state_sample_index[row]),
                int(batch.state_node_index[row]),
                int(batch.state_adduct_index[row]),
                int(batch.state_hydrogen_shift[row]),
            )
            ion_lookup[key] = ion
        adduct_index = {value: i for i, value in enumerate(batch.ion_type_strs)}
        peak_mz, intensity, peak_rows, matched_ions = [], [], [], []
        peak_precursor = []
        precursor, energy, metadata = [], [], []
        for sample, annotation in enumerate(observations):
            peaks = annotation.get("peaks", [])
            peak_rows.append(list(range(len(peak_mz), len(peak_mz) + len(peaks))))
            precursor.append(annotation["precursorMz"])
            energy.append(annotation["collisionEnergy"])
            metadata.append(
                {
                    k: v
                    for k, v in annotation.items()
                    if k
                    not in (
                        "peaks",
                        "precursorMz",
                        "collisionEnergy",
                        "adduct",
                        "mainAdduct",
                    )
                }
            )
            if "adduct" in annotation:
                metadata[-1]["observed_adduct"] = annotation["adduct"]
            for peak in peaks:
                peak_mz.append(peak["mz"])
                intensity.append(peak["intensity"])
                matches = set()
                peak_precursor.append(peak.get("precursor", False))
                for match in peak.get("matches", []):
                    adduct = adduct_index.get(
                        match.get("fragmentIonAdduct", match.get("adduct"))
                    )
                    for node in match.get("nodeIndices", []):
                        key = (sample, node, adduct, match.get("hydrogenShift", 0))
                        if key in ion_lookup:
                            matches.add(ion_lookup[key])
                matched_ions.append(sorted(matches))
        peak_ptr, _ = csr(peak_rows, device)
        ip, ii = csr(matched_ions, device)
        spectra = SpectrumData(
            peak_ptr,
            torch.tensor(peak_mz, dtype=torch.float64, device=device),
            torch.tensor(intensity, dtype=torch.float32, device=device),
            torch.tensor(peak_precursor, dtype=torch.bool, device=device),
            branch_group_index,
            torch.tensor(precursor, dtype=torch.float64, device=device),
            torch.tensor(energy, dtype=torch.float64, device=device),
            tuple(metadata),
        )
        return cls(
            spectra,
            compounds,
            formulas,
            ions,
            runtime,
            PeakAssignmentData(ip, ii),
            tuple(elements),
            FragmentOccurrenceData(occurrence, decoded.node_sample_index),
            MaterializedContextData(context_reaction_set_index, materialized),
        )

    def export_samples(self, adducts, main_adduct_index):
        """Create viewer dictionaries on demand; the stored tables remain normalized."""
        from fredis.libs.mmkit.mmkit import Adduct, Formula

        result = []
        s, a, ions, formulas = (
            self.spectra,
            self.assignments,
            self.formula_explanations,
            self.formulas,
        )
        for sample in range(s.branch_group_index.numel()):
            peaks = []
            for local, peak in enumerate(
                range(int(s.peak_ptr[sample]), int(s.peak_ptr[sample + 1]))
            ):
                matches = []
                for ion in a.formula_explanation_index[
                    a.formula_explanation_ptr[peak] : a.formula_explanation_ptr[
                        peak + 1
                    ]
                ].tolist():
                    compound = int(ions.compound_index[ion])
                    formula = int(ions.formula_index[ion])
                    shift = int(adducts.hydrogen_shifts[ions.hydrogen_shift_index[ion]])
                    base = adducts.fragment_ion_adducts[
                        int(ions.fragment_ion_adduct_index[ion])
                    ]
                    mz = float(formulas.mz[formula])
                    observed = float(s.mz[peak])
                    nodes = (
                        (
                            (self.occurrences.compound_index == compound)
                            & (
                                self.occurrences.branch_group_index
                                == s.branch_group_index[sample]
                            )
                        )
                        .nonzero()
                        .flatten()
                        .tolist()
                    )
                    counts = formulas.composition[formula].tolist()
                    label = (
                        str(
                            Formula(
                                elements={
                                    e: int(v)
                                    for e, v in zip(self.elements, counts[:-1])
                                    if v
                                },
                                charge=int(counts[-1]),
                            )
                        )
                        if self.elements
                        else ""
                    )
                    matches.append(
                        dict(
                            nodeIndices=nodes,
                            smiles=self.compounds.smiles[compound],
                            formula=label,
                            theoreticalMz=mz,
                            massErrorPpm=(observed - mz) / mz * 1e6 if mz else 0,
                            fragmentIonAdduct=base,
                            hydrogenShift=shift,
                            adduct=str(
                                Adduct.parse(base).add_prefer_self(
                                    Adduct.from_dict({"H": shift})
                                )
                            ),
                        )
                    )
                peaks.append(
                    dict(
                        index=local,
                        mz=float(s.mz[peak]),
                        intensity=float(s.intensity[peak]),
                        precursor=bool(s.precursor[peak]),
                        matches=matches,
                    )
                )
            result.append(
                dict(
                    {
                        k: v
                        for k, v in s.metadata[sample].items()
                        if k != "observed_adduct"
                    },
                    adduct=s.metadata[sample].get(
                        "observed_adduct",
                        adducts.main_adducts[
                            int(main_adduct_index[int(s.branch_group_index[sample])])
                        ],
                    ),
                    mainAdduct=adducts.main_adducts[
                        int(main_adduct_index[int(s.branch_group_index[sample])])
                    ],
                    collisionEnergy=float(s.collision_energy[sample]),
                    precursorMz=float(s.precursor_mz[sample]),
                    peaks=peaks,
                )
            )
        return tuple(result)

    def validate(self, adducts):
        """Check CSR bounds, identities and foreign-key domains before serialization."""

        def check_csr(name, ptr, values, rows):
            if (
                ptr.dtype != torch.long
                or ptr.ndim != 1
                or ptr.numel() != rows + 1
                or int(ptr[0]) != 0
                or int(ptr[-1]) != values.shape[0]
                or bool((ptr[1:] < ptr[:-1]).any())
            ):
                raise ValueError(f"Invalid CSR table: {name}")

        def check_index(name, values, size, missing=False):
            if (
                values.dtype != torch.long
                or values.ndim != 1
                or (
                    values.numel()
                    and (
                        int(values.min()) < (-1 if missing else 0)
                        or int(values.max()) >= size
                    )
                )
            ):
                raise ValueError(f"Index outside referenced table: {name}")

        spectra, compounds, formulas, ions, assignment = (
            self.spectra,
            self.compounds,
            self.formulas,
            self.formula_explanations,
            self.assignments,
        )
        scount = spectra.branch_group_index.numel()
        ccount, fcount, icount = (
            len(compounds.smiles),
            len(formulas.composition),
            len(ions.compound_index),
        )
        if len(set(compounds.smiles)) != ccount or compounds.graph.num_graphs != ccount:
            raise ValueError(
                "CompoundBatch must contain exactly one graph per unique structure"
            )
        if self.elements and formulas.composition.shape[1] != len(self.elements) + 1:
            raise ValueError("Formula columns must match elements followed by charge")
        check_csr("spectrum->peak", spectra.peak_ptr, spectra.mz, scount)
        check_csr(
            "compound->state",
            compounds.generating_reaction_sets_ptr,
            compounds.generating_reaction_sets_index,
            ccount,
        )
        check_csr(
            "peak->explanation",
            assignment.formula_explanation_ptr,
            assignment.formula_explanation_index,
            len(spectra.mz),
        )
        for name, values, size, missing in (
            ("occurrence->compound", self.occurrences.compound_index, ccount, False),
            ("state->compound", self.contexts.compound_index, ccount, True),
            (
                "compound->set",
                compounds.generating_reaction_sets_index,
                (
                    int(self.contexts.reaction_set_index.max()) + 1
                    if self.contexts.reaction_set_index.numel()
                    else 0
                ),
                False,
            ),
            ("explanation->compound", ions.compound_index, ccount, False),
            ("explanation->formula", ions.formula_index, fcount, False),
            (
                "explanation->adduct",
                ions.fragment_ion_adduct_index,
                len(adducts.fragment_ion_adducts),
                False,
            ),
            (
                "explanation->shift",
                ions.hydrogen_shift_index,
                len(adducts.hydrogen_shifts),
                False,
            ),
            (
                "runtime->explanation",
                self.runtime_explanations.formula_explanation_index,
                icount,
                False,
            ),
            (
                "runtime->spectrum",
                self.runtime_explanations.sample_index,
                scount,
                False,
            ),
            ("peak->explanation", assignment.formula_explanation_index, icount, False),
        ):
            check_index(name, values, size, missing)
        if formulas.composition.dtype != torch.long or formulas.composition.ndim != 2:
            raise ValueError("Formula composition must be an int64 matrix")
        if (
            formulas.mz.dtype != torch.float64
            or formulas.mz.shape != (fcount,)
            or formulas.valid.dtype != torch.bool
            or formulas.valid.shape != (fcount,)
        ):
            raise ValueError("Formula properties must match formula rows")
        if not torch.equal(formulas.valid, (formulas.composition[:, :-1] >= 0).all(1)):
            raise ValueError("Formula validity must match elemental counts")
        if torch.unique(formulas.composition, dim=0).shape[0] != fcount:
            raise ValueError("Formula compositions must be unique")
        columns = [getattr(ions, f.name) for f in fields(ions)]
        if any(column.shape != (icount,) for column in columns):
            raise ValueError("Formula explanation columns must have matching shapes")
        combinations = torch.stack(
            (
                ions.compound_index,
                ions.fragment_ion_adduct_index,
                ions.hydrogen_shift_index,
            ),
            1,
        )
        if torch.unique(combinations, dim=0).shape[0] != icount:
            raise ValueError("Compound/adduct/shift explanations must be unique")
        runtime = self.runtime_explanations
        if runtime.formula_explanation_index.shape != runtime.sample_index.shape:
            raise ValueError("Runtime explanation columns must have matching shapes")
        for start, stop in zip(
            assignment.formula_explanation_ptr[:-1],
            assignment.formula_explanation_ptr[1:],
        ):
            ids = assignment.formula_explanation_index[start:stop]
            if not torch.equal(ids, ids.unique(sorted=True)):
                raise ValueError("Peak explanation IDs must be sorted and unique")
        for key, dtype in (
            ("mz", torch.float64),
            ("intensity", torch.float32),
            ("precursor", torch.bool),
        ):
            value = getattr(spectra, key)
            if value.dtype != dtype or value.ndim != 1:
                raise ValueError(f"Invalid observed peak column: {key}")
        for key in ("precursor_mz", "collision_energy"):
            value = getattr(spectra, key)
            if value.dtype != torch.float64 or value.shape != (scount,):
                raise ValueError(f"Invalid spectrum column: {key}")
        if len(spectra.metadata) != scount:
            raise ValueError("Spectrum metadata length must match spectrum rows")
        check_index(
            "spectrum->group",
            spectra.branch_group_index,
            int(spectra.branch_group_index.max()) + 1 if scount else 0,
        )
        if (
            self.occurrences.compound_index.shape
            != self.occurrences.branch_group_index.shape
        ):
            raise ValueError("Fragment occurrence columns must have matching shapes")
        context_compound = self.contexts.compound_index
        context_sets = self.contexts.reaction_set_index
        if context_sets.shape != context_compound.shape or bool(
            (context_sets < 0).any()
        ):
            raise ValueError(
                "Materialization contexts must reference one ReactionSet each"
            )
        expected = sorted(
            set(
                (compound, reaction_set)
                for compound, reaction_set in zip(
                    context_compound.tolist(), context_sets.tolist()
                )
                if compound >= 0
            )
        )
        owner = torch.repeat_interleave(
            torch.arange(ccount, device=context_compound.device),
            compounds.generating_reaction_sets_ptr[1:]
            - compounds.generating_reaction_sets_ptr[:-1],
        )
        actual = list(
            zip(owner.tolist(), compounds.generating_reaction_sets_index.tolist())
        )
        if actual != expected:
            raise ValueError(
                "Compound reaction-set CSR must contain each materialized pair once"
            )
        for name, value in (
            ("intensity", spectra.intensity),
            ("precursor", spectra.precursor),
        ):
            if value.shape != spectra.mz.shape:
                raise ValueError(f"Observed peak {name} shape differs from m/z")

    @classmethod
    def collate(cls, items, group_counts, reaction_set_maps, adducts):
        """Batch normalized tables directly; never round-trip through viewer dictionaries."""
        if not items:
            raise ValueError("Cannot collate empty annotations")
        if any(item.elements != items[0].elements for item in items):
            raise ValueError("Formula element vocabularies must match")
        device = items[0].spectra.mz.device

        def tensor(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        def concat_ptr(pointers):
            parts = [pointers[0]]
            offset = int(pointers[0][-1])
            for ptr in pointers[1:]:
                parts.append(ptr[1:] + offset)
                offset += int(ptr[-1])
            return torch.cat(parts)

        smiles, compounds, graphs, lookup, compound_maps = [], [], [], {}, []
        for item in items:
            local = []
            local_graphs = item.compounds.graph.to_data_list()
            for i, key in enumerate(item.compounds.smiles):
                if key not in lookup:
                    lookup[key] = len(smiles)
                    smiles.append(key)
                    compounds.append(item.compounds.compounds[i])
                    graphs.append(local_graphs[i])
                local.append(lookup[key])
            compound_maps.append(tensor(local))
        composition, all_formula_map = torch.unique(
            torch.cat([a.formulas.composition for a in items]),
            dim=0,
            return_inverse=True,
        )
        formula_maps = all_formula_map.split(
            [len(a.formulas.composition) for a in items]
        )
        mz = torch.cat([a.formulas.mz for a in items]).new_zeros(len(composition))
        mz[all_formula_map] = torch.cat([a.formulas.mz for a in items])
        raw = []
        for a, cm, fm in zip(items, compound_maps, formula_maps):
            shift = a.formula_explanations.hydrogen_shift_index
            raw.append(
                torch.stack(
                    (
                        fm[a.formula_explanations.formula_index],
                        cm[a.formula_explanations.compound_index],
                        a.formula_explanations.fragment_ion_adduct_index,
                        shift,
                    ),
                    dim=1,
                )
            )
        ion_rows, all_ion_map = torch.unique(torch.cat(raw), dim=0, return_inverse=True)
        ion_maps = all_ion_map.split(
            [len(a.formula_explanations.compound_index) for a in items]
        )
        occurrence = []
        states = []
        context_sets = []
        runtime = []
        runtime_sample = []
        groups = []
        occurrence_groups = []
        so = go = 0
        for a, cm, im, gc, set_map in zip(
            items, compound_maps, ion_maps, group_counts, reaction_set_maps, strict=True
        ):
            occurrence.append(cm[a.occurrences.compound_index])
            old = a.contexts.compound_index
            states.append(torch.where(old >= 0, cm[old.clamp_min(0)], old))
            context_sets.append(set_map[a.contexts.reaction_set_index])
            runtime.append(im[a.runtime_explanations.formula_explanation_index])
            runtime_sample.append(a.runtime_explanations.sample_index + so)
            groups.append(a.spectra.branch_group_index + go)
            occurrence_groups.append(a.occurrences.branch_group_index + go)
            so += a.spectra.branch_group_index.numel()
            go += gc
        state_index = torch.cat(states)
        context_reaction_set_index = torch.cat(context_sets)
        reaction_set_index = [[] for _ in smiles]
        for row, compound in enumerate(state_index.tolist()):
            if compound >= 0:
                reaction_set_index[compound].append(
                    int(context_reaction_set_index[row])
                )
        cp, ci = csr([sorted(set(ids)) for ids in reaction_set_index], device)
        c = CompoundBatch(tuple(compounds), Batch.from_data_list(graphs), cp, ci)
        f = FormulaData(composition, mz, (composition[:, :-1] >= 0).all(1))
        ions = FormulaExplanationData(*ion_rows.unbind(1))
        runtime_rows = RuntimeFormulaExplanationData(
            torch.cat(runtime), torch.cat(runtime_sample)
        )
        s = SpectrumData(
            concat_ptr([a.spectra.peak_ptr for a in items]),
            torch.cat([a.spectra.mz for a in items]),
            torch.cat([a.spectra.intensity for a in items]),
            torch.cat([a.spectra.precursor for a in items]),
            torch.cat(groups),
            torch.cat([a.spectra.precursor_mz for a in items]),
            torch.cat([a.spectra.collision_energy for a in items]),
            tuple(v for a in items for v in a.spectra.metadata),
        )
        assignment_rows = []
        for item, mapping in zip(items, ion_maps):
            old = item.assignments
            for start, stop in zip(
                old.formula_explanation_ptr[:-1], old.formula_explanation_ptr[1:]
            ):
                assignment_rows.append(
                    sorted(
                        set(mapping[old.formula_explanation_index[start:stop]].tolist())
                    )
                )
        assignments = PeakAssignmentData(*csr(assignment_rows, device))
        result = cls(
            s,
            c,
            f,
            ions,
            runtime_rows,
            assignments,
            items[0].elements,
            FragmentOccurrenceData(torch.cat(occurrence), torch.cat(occurrence_groups)),
            MaterializedContextData(context_reaction_set_index, state_index),
        )
        result.validate(adducts)
        return result
