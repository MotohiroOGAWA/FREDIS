"""Normalized spectrum annotations, independent of training targets.

Index spaces: S spectra, P observed peaks, C unique compounds, F unique charged
formulas, I unique ion candidates, V chemical reaction states, A assignments.
Indices are global within this object, never dataset-global. CSR ptr arrays have
one more entry than their row count. Compound identity includes stereochemistry
and charge through canonical isomeric SMILES; formula identity includes charge.
"""

from __future__ import annotations
from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor
from torch_geometric.data import Batch
from .reaction_context_data import TensorTable
from .compound_batch import CompoundBatch


def csr(rows, device=None):
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long, device=device)
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0))), torch.tensor(
        [v for row in rows for v in row], dtype=torch.long, device=device
    )


@dataclass(frozen=True)
class SpectrumData(TensorTable):
    # [S + 1]: observed peak rows belonging to each spectrum.
    peak_ptr: Tensor
    # [P]: observed m/z and relative intensity; these are measurements, not predictions.
    mz: Tensor
    intensity: Tensor
    # [P]: precursor membership determined by the configured mass tolerance.
    precursor: Tensor
    # [S]: source molecule and branch-group IDs, respectively.
    source_index: Tensor
    branch_group_index: Tensor
    # [S]: measured precursor m/z and collision energy.
    precursor_mz: Tensor
    collision_energy: Tensor
    # Length S: spectrum adduct labels; may differ from branch-group main adduct.
    adducts: tuple[str, ...]
    # Length S: display metadata excluding peaks, conditions and assignments.
    metadata: tuple[dict, ...]

    def to(self, device):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
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
class FormulaData(TensorTable):
    # [F, D]: unique elemental counts; the last column is charge.
    composition: Tensor
    # [F]: exact m/z; formulas with negative element counts are marked invalid.
    mz: Tensor
    valid: Tensor
    # [F + 1], [I]: inverse CSR, formula -> all structural ion explanations.
    ion_ptr: Tensor
    ion_index: Tensor


@dataclass(frozen=True)
class IonCandidateData(TensorTable):
    # [I]: references into CompoundBatch, FormulaData and the adduct vocabulary.
    compound_index: Tensor
    formula_index: Tensor
    adduct_index: Tensor
    # [I]: index into hydrogen_shifts, NOT the signed shift value itself.
    hydrogen_shift_index: Tensor
    # [Hshift]: signed hydrogen shift vocabulary.
    hydrogen_shifts: Tensor
    # [J]: runtime ion-condition row -> unique ion candidate. Repeated CE samples share IDs.
    runtime_index: Tensor
    # [J]: spectrum ID for each runtime row; topology remains separate from candidate identity.
    runtime_sample_index: Tensor


@dataclass(frozen=True)
class PeakAssignmentData(TensorTable):
    # [P + 1], [A]: observed peak -> alternative charged formulas (no structural duplication).
    formula_ptr: Tensor
    formula_index: Tensor
    # [P + 1], [B]: observed peak -> specific admissible ion explanations.
    ion_ptr: Tensor
    ion_index: Tensor


@dataclass(frozen=True)
class AnnotatedSpectrumData:
    spectra: SpectrumData
    compounds: CompoundBatch
    formulas: FormulaData
    ions: IonCandidateData
    assignments: PeakAssignmentData
    # Shared adduct vocabulary; ions.adduct_index refers here.
    ion_adducts: tuple[str, ...]
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
            ions=self.ions.to(device),
            assignments=self.assignments.to(device),
            occurrences=self.occurrences.to(device),
            contexts=self.contexts.to(device),
        )

    @classmethod
    def from_runtime(
        cls, batch, observations, source_index, branch_group_index, elements=(), state_nodes=None,
        *, context_reaction_set_index: Tensor
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
        node_rows = decoded.materialized_node_index if state_nodes is None else state_nodes
        materialized = torch.where(node_rows >= 0, occurrence[node_rows.clamp_min(0)], node_rows)
        reaction_set_index = [[] for _ in smiles]
        for row, compound in enumerate(materialized.tolist()):
            if compound >= 0:
                reaction_set_index[compound].append(int(context_reaction_set_index[row]))
        reaction_set_ptr, reaction_set_index = csr(
            [sorted(set(ids)) for ids in reaction_set_index], device
        )
        compounds = CompoundBatch(
            tuple(decoded.compounds[i] for i in representatives),
            Batch.from_data_list([graphs[i] for i in representatives]),
            reaction_set_ptr, reaction_set_index,
        )
        composition, formula_inverse = torch.unique(
            batch.state_formula_tensor, dim=0, return_inverse=True
        )
        fcount = len(composition)
        formula_mz = batch.state_mz.new_zeros(fcount)
        formula_mz[formula_inverse] = batch.state_mz
        shifts, shift_inverse = torch.unique(
            batch.state_hydrogen_shift, sorted=True, return_inverse=True
        )
        raw = torch.stack(
            (
                occurrence[batch.state_node_index],
                formula_inverse,
                batch.state_adduct_index,
                shift_inverse,
            ),
            dim=1,
        )
        ion_rows, ion_inverse = torch.unique(raw, dim=0, return_inverse=True)
        inverse_rows = [[] for _ in range(fcount)]
        for ion, formula in enumerate(ion_rows[:, 1].tolist()):
            inverse_rows[formula].append(ion)
        ion_ptr, ion_index = csr(inverse_rows, device)
        formulas = FormulaData(
            composition, formula_mz, (composition[:, :-1] >= 0).all(1), ion_ptr, ion_index
        )
        ions = IonCandidateData(*ion_rows.unbind(1), shifts, ion_inverse, batch.state_sample_index)
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
        peak_mz, intensity, peak_rows, formula_rows, matched_ions = [], [], [], [], []
        peak_precursor = []
        precursor, energy, adducts, metadata = [], [], [], []
        for sample, annotation in enumerate(observations):
            peaks = annotation.get("peaks", [])
            peak_rows.append(list(range(len(peak_mz), len(peak_mz) + len(peaks))))
            precursor.append(annotation["precursorMz"])
            energy.append(annotation["collisionEnergy"])
            adducts.append(annotation["adduct"])
            metadata.append(
                {
                    k: v
                    for k, v in annotation.items()
                    if k not in ("peaks", "precursorMz", "collisionEnergy", "adduct")
                }
            )
            for peak in peaks:
                peak_mz.append(peak["mz"])
                intensity.append(peak["intensity"])
                matches = set()
                peak_precursor.append(peak.get("precursor", False))
                for match in peak.get("matches", []):
                    adduct = adduct_index.get(match.get("fragmentIonAdduct", match.get("adduct")))
                    for node in match.get("nodeIndices", []):
                        key = (sample, node, adduct, match.get("hydrogenShift", 0))
                        if key in ion_lookup:
                            matches.add(ion_lookup[key])
                matched_ions.append(sorted(matches))
                formula_rows.append(sorted({int(ions.formula_index[i]) for i in matches}))
        peak_ptr, _ = csr(peak_rows, device)
        fp, fi = csr(formula_rows, device)
        ip, ii = csr(matched_ions, device)
        spectra = SpectrumData(
            peak_ptr,
            torch.tensor(peak_mz, dtype=torch.float64, device=device),
            torch.tensor(intensity, dtype=torch.float32, device=device),
            torch.tensor(peak_precursor, dtype=torch.bool, device=device),
            source_index,
            branch_group_index,
            torch.tensor(precursor, dtype=torch.float64, device=device),
            torch.tensor(energy, dtype=torch.float64, device=device),
            tuple(adducts),
            tuple(metadata),
        )
        return cls(
            spectra,
            compounds,
            formulas,
            ions,
            PeakAssignmentData(fp, fi, ip, ii),
            batch.ion_type_strs,
            tuple(elements),
            FragmentOccurrenceData(occurrence, decoded.node_sample_index),
            MaterializedContextData(context_reaction_set_index, materialized),
        )

    def export_samples(self):
        """Create viewer dictionaries on demand; the stored tables remain normalized."""
        from fredis.libs.mmkit.mmkit import Adduct, Formula

        result = []
        s, a, ions, formulas = self.spectra, self.assignments, self.ions, self.formulas
        for sample in range(len(s.adducts)):
            peaks = []
            for local, peak in enumerate(
                range(int(s.peak_ptr[sample]), int(s.peak_ptr[sample + 1]))
            ):
                matches = []
                for ion in a.ion_index[a.ion_ptr[peak] : a.ion_ptr[peak + 1]].tolist():
                    compound = int(ions.compound_index[ion])
                    formula = int(ions.formula_index[ion])
                    shift = int(ions.hydrogen_shifts[ions.hydrogen_shift_index[ion]])
                    base = self.ion_adducts[int(ions.adduct_index[ion])]
                    mz = float(formulas.mz[formula])
                    observed = float(s.mz[peak])
                    nodes = (
                        (
                            (self.occurrences.compound_index == compound)
                            & (self.occurrences.branch_group_index == s.branch_group_index[sample])
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
                                    e: int(v) for e, v in zip(self.elements, counts[:-1]) if v
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
                                Adduct.parse(base).add_prefer_self(Adduct.from_dict({"H": shift}))
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
                    s.metadata[sample],
                    adduct=s.adducts[sample],
                    collisionEnergy=float(s.collision_energy[sample]),
                    precursorMz=float(s.precursor_mz[sample]),
                    peaks=peaks,
                )
            )
        return tuple(result)

    def validate(self):
        """Check CSR bounds, identities and foreign-key domains before serialization."""

        def check_csr(name, ptr, values, rows):
            if (
                ptr.ndim != 1
                or ptr.numel() != rows + 1
                or int(ptr[0]) != 0
                or int(ptr[-1]) != values.shape[0]
                or bool((ptr[1:] < ptr[:-1]).any())
            ):
                raise ValueError(f"Invalid CSR table: {name}")

        def check_index(name, values, size, missing=False):
            if values.numel() and (
                int(values.min()) < (-1 if missing else 0) or int(values.max()) >= size
            ):
                raise ValueError(f"Index outside referenced table: {name}")

        spectra, compounds, formulas, ions, assignment = (
            self.spectra,
            self.compounds,
            self.formulas,
            self.ions,
            self.assignments,
        )
        scount = len(spectra.adducts)
        ccount, fcount, icount = (
            len(compounds.smiles),
            len(formulas.composition),
            len(ions.compound_index),
        )
        if len(set(compounds.smiles)) != ccount or compounds.graph.num_graphs != ccount:
            raise ValueError("CompoundBatch must contain exactly one graph per unique structure")
        if self.elements and formulas.composition.shape[1] != len(self.elements) + 1:
            raise ValueError("Formula columns must match elements followed by charge")
        check_csr("spectrum->peak", spectra.peak_ptr, spectra.mz, scount)
        check_csr("formula->ion", formulas.ion_ptr, formulas.ion_index, fcount)
        check_csr("compound->state", compounds.generating_reaction_sets_ptr, compounds.generating_reaction_sets_index, ccount)
        check_csr(
            "peak->formula", assignment.formula_ptr, assignment.formula_index, len(spectra.mz)
        )
        check_csr("peak->ion", assignment.ion_ptr, assignment.ion_index, len(spectra.mz))
        for name, values, size, missing in (
            ("occurrence->compound", self.occurrences.compound_index, ccount, False),
            ("state->compound", self.contexts.compound_index, ccount, True),
            ("compound->set", compounds.generating_reaction_sets_index,
             int(self.contexts.reaction_set_index.max()) + 1 if self.contexts.reaction_set_index.numel() else 0, False),
            ("formula->ion", formulas.ion_index, icount, False),
            ("ion->compound", ions.compound_index, ccount, False),
            ("ion->formula", ions.formula_index, fcount, False),
            ("ion->adduct", ions.adduct_index, len(self.ion_adducts), False),
            ("ion->shift", ions.hydrogen_shift_index, len(ions.hydrogen_shifts), False),
            ("runtime->ion", ions.runtime_index, icount, False),
            ("runtime->spectrum", ions.runtime_sample_index, scount, False),
            ("peak->formula", assignment.formula_index, fcount, False),
            ("peak->ion", assignment.ion_index, icount, False),
        ):
            check_index(name, values, size, missing)
        context_compound = self.contexts.compound_index
        context_sets = self.contexts.reaction_set_index
        if context_sets.shape != context_compound.shape or bool((context_sets < 0).any()):
            raise ValueError("Materialization contexts must reference one ReactionSet each")
        expected = sorted(set(
            (compound, reaction_set)
            for compound, reaction_set in zip(context_compound.tolist(), context_sets.tolist())
            if compound >= 0
        ))
        owner = torch.repeat_interleave(
            torch.arange(ccount, device=context_compound.device),
            compounds.generating_reaction_sets_ptr[1:] - compounds.generating_reaction_sets_ptr[:-1],
        )
        actual = list(zip(owner.tolist(), compounds.generating_reaction_sets_index.tolist()))
        if actual != expected:
            raise ValueError("Compound reaction-set CSR must contain each materialized pair once")
        for name, value in (("intensity", spectra.intensity), ("precursor", spectra.precursor)):
            if value.shape != spectra.mz.shape:
                raise ValueError(f"Observed peak {name} shape differs from m/z")

    @classmethod
    def collate(cls, items, source_counts, group_counts, reaction_set_maps):
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
            torch.cat([a.formulas.composition for a in items]), dim=0, return_inverse=True
        )
        formula_maps = all_formula_map.split([len(a.formulas.composition) for a in items])
        mz = torch.cat([a.formulas.mz for a in items]).new_zeros(len(composition))
        mz[all_formula_map] = torch.cat([a.formulas.mz for a in items])
        adducts = tuple(dict.fromkeys(v for a in items for v in a.ion_adducts))
        shifts = torch.unique(torch.cat([a.ions.hydrogen_shifts for a in items]), sorted=True)
        raw = []
        for a, cm, fm in zip(items, compound_maps, formula_maps):
            am = tensor([adducts.index(v) for v in a.ion_adducts])
            shift = torch.searchsorted(shifts, a.ions.hydrogen_shifts[a.ions.hydrogen_shift_index])
            raw.append(
                torch.stack(
                    (
                        cm[a.ions.compound_index],
                        fm[a.ions.formula_index],
                        am[a.ions.adduct_index],
                        shift,
                    ),
                    dim=1,
                )
            )
        ion_rows, all_ion_map = torch.unique(torch.cat(raw), dim=0, return_inverse=True)
        ion_maps = all_ion_map.split([len(a.ions.compound_index) for a in items])
        occurrence = []
        states = []
        context_sets = []
        runtime = []
        runtime_sample = []
        source = []
        groups = []
        occurrence_groups = []
        so = go = mo = 0
        for a, cm, im, sc, gc, set_map in zip(
            items, compound_maps, ion_maps, source_counts, group_counts, reaction_set_maps, strict=True
        ):
            occurrence.append(cm[a.occurrences.compound_index])
            old = a.contexts.compound_index
            states.append(torch.where(old >= 0, cm[old.clamp_min(0)], old))
            context_sets.append(set_map[a.contexts.reaction_set_index])
            runtime.append(im[a.ions.runtime_index])
            runtime_sample.append(a.ions.runtime_sample_index + so)
            source.append(a.spectra.source_index + mo)
            groups.append(a.spectra.branch_group_index + go)
            occurrence_groups.append(a.occurrences.branch_group_index + go)
            so += len(a.spectra.adducts)
            mo += sc
            go += gc
        state_index = torch.cat(states)
        context_reaction_set_index = torch.cat(context_sets)
        reaction_set_index = [[] for _ in smiles]
        for row, compound in enumerate(state_index.tolist()):
            if compound >= 0:
                reaction_set_index[compound].append(int(context_reaction_set_index[row]))
        cp, ci = csr([sorted(set(ids)) for ids in reaction_set_index], device)
        c = CompoundBatch(tuple(compounds), Batch.from_data_list(graphs), cp, ci)
        formula_ions = [[] for _ in range(len(composition))]
        for i, f in enumerate(ion_rows[:, 1].tolist()):
            formula_ions[f].append(i)
        fp, fi = csr(formula_ions, device)
        f = FormulaData(composition, mz, (composition[:, :-1] >= 0).all(1), fp, fi)
        ions = IonCandidateData(
            *ion_rows.unbind(1), shifts, torch.cat(runtime), torch.cat(runtime_sample)
        )
        s = SpectrumData(
            concat_ptr([a.spectra.peak_ptr for a in items]),
            torch.cat([a.spectra.mz for a in items]),
            torch.cat([a.spectra.intensity for a in items]),
            torch.cat([a.spectra.precursor for a in items]),
            torch.cat(source),
            torch.cat(groups),
            torch.cat([a.spectra.precursor_mz for a in items]),
            torch.cat([a.spectra.collision_energy for a in items]),
            tuple(v for a in items for v in a.spectra.adducts),
            tuple(v for a in items for v in a.spectra.metadata),
        )
        assignments = PeakAssignmentData(
            concat_ptr([a.assignments.formula_ptr for a in items]),
            torch.cat([fm[a.assignments.formula_index] for a, fm in zip(items, formula_maps)]),
            concat_ptr([a.assignments.ion_ptr for a in items]),
            torch.cat([im[a.assignments.ion_index] for a, im in zip(items, ion_maps)]),
        )
        result = cls(
            s,
            c,
            f,
            ions,
            assignments,
            adducts,
            items[0].elements,
            FragmentOccurrenceData(torch.cat(occurrence), torch.cat(occurrence_groups)),
            MaterializedContextData(context_reaction_set_index, state_index),
        )
        result.validate()
        return result
