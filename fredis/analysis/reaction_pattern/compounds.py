"""Phase 2: group every valid spectrum of every dataset by unique compound."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .constants import COMPOUND_DIR, PEAKS_SUFFIX


@dataclass
class SampleRecord:
    """One valid spectrum after peak filtering and normalization.

    ``peak_index`` is the position in ``mz``; it is the peak identity used to
    join Baseline and Pattern assignments (never the m/z float).
    """
    dataset_id: int
    record_index: int
    smiles: str
    adduct: str
    collision_energy: float
    precursor_mz: float
    mz: np.ndarray
    intensity: np.ndarray
    relative_intensity: np.ndarray
    sample_id: int = -1
    sample_index: int = -1
    is_precursor: np.ndarray | None = None


@dataclass
class CompoundGroup:
    compound_id: str
    canonical_smiles: str
    original_smiles: str
    samples: list[SampleRecord] = field(default_factory=list)

    @property
    def dataset_ids(self) -> list[int]:
        return sorted({sample.dataset_id for sample in self.samples})

    @property
    def peak_summary_file(self) -> str:
        return f"{COMPOUND_DIR}/{self.compound_id}{PEAKS_SUFFIX}"


def canonical_smiles(smiles: str) -> str | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def compound_id_for(canonical: str) -> str:
    """Stable, filename-safe compound identity derived from canonical SMILES."""
    return "c" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def group_compounds(samples: list[SampleRecord]) -> list[CompoundGroup]:
    """Return compounds sorted by id, with deterministic sample ids.

    Every spectrum (any dataset, adduct or collision energy) of the same
    canonical SMILES belongs to one group.  ``sample_id`` is assigned in
    compound order so sample tables are contiguous per compound.
    """
    canonical_cache: dict[str, str | None] = {}
    groups: dict[str, CompoundGroup] = {}
    for sample in samples:
        if sample.smiles not in canonical_cache:
            canonical_cache[sample.smiles] = canonical_smiles(sample.smiles)
        canonical = canonical_cache[sample.smiles]
        if canonical is None:
            continue
        compound_id = compound_id_for(canonical)
        group = groups.get(compound_id)
        if group is None:
            group = groups[compound_id] = CompoundGroup(compound_id, canonical, sample.smiles)
        group.samples.append(sample)
    ordered = [groups[key] for key in sorted(groups)]
    next_sample_id = 0
    for group in ordered:
        group.samples.sort(key=lambda value: (value.dataset_id, value.record_index))
        for position, sample in enumerate(group.samples):
            sample.sample_index = position
            sample.sample_id = next_sample_id
            next_sample_id += 1
    return ordered


def mark_precursor_peaks(groups: list[CompoundGroup], mass_tolerance) -> None:
    """Flag peaks within tolerance of each sample's theoretical precursor m/z."""
    from fredis.libs.mmkit.mmkit import Adduct, Compound

    for group in groups:
        formula = Compound.from_smiles(group.canonical_smiles).formula
        cache: dict[str, float] = {}
        for sample in group.samples:
            if sample.adduct not in cache:
                cache[sample.adduct] = Adduct.parse(sample.adduct).apply_to_formula(formula).normalized.exact_mass
            target = cache[sample.adduct]
            sample.is_precursor = np.fromiter(
                (bool(mass_tolerance.within(float(value), target)) for value in sample.mz),
                dtype=bool, count=len(sample.mz))
