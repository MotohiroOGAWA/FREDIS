from __future__ import annotations

from collections.abc import Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DEFAULT_DESCRIPTOR_NAMES = (
    "ExactMolWt",
    "HeavyAtomCount",
    "TPSA",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "NumHeteroatoms",
    "FormalCharge",
    "BertzCT",
)

# Every descriptor compute_descriptor_values can calculate, in a fixed order.
SUPPORTED_DESCRIPTOR_NAMES = ("MolWt", *DEFAULT_DESCRIPTOR_NAMES)


def compute_descriptor_values(
    mol: Chem.Mol,
    names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES,
) -> tuple[float, ...]:
    """Calculate the supported descriptors in the requested order."""
    values = []
    for name in names:
        if name == "MolWt":
            value = Descriptors.MolWt(mol)
        elif name == "ExactMolWt":
            value = Descriptors.ExactMolWt(mol)
        elif name == "TPSA":
            value = rdMolDescriptors.CalcTPSA(mol)
        elif name == "MolLogP":
            value = Descriptors.MolLogP(mol)
        elif name == "NumHAcceptors":
            value = rdMolDescriptors.CalcNumHBA(mol)
        elif name == "NumHDonors":
            value = rdMolDescriptors.CalcNumHBD(mol)
        elif name == "NumRotatableBonds":
            value = rdMolDescriptors.CalcNumRotatableBonds(mol)
        elif name == "RingCount":
            value = rdMolDescriptors.CalcNumRings(mol)
        elif name == "NumAromaticRings":
            value = rdMolDescriptors.CalcNumAromaticRings(mol)
        elif name == "NumAliphaticRings":
            value = rdMolDescriptors.CalcNumAliphaticRings(mol)
        elif name == "FractionCSP3":
            value = rdMolDescriptors.CalcFractionCSP3(mol)
        elif name == "HeavyAtomCount":
            value = mol.GetNumHeavyAtoms()
        elif name == "NumHeteroatoms":
            value = Descriptors.NumHeteroatoms(mol)
        elif name == "FormalCharge":
            value = Chem.GetFormalCharge(mol)
        elif name == "BertzCT":
            value = Descriptors.BertzCT(mol)
        else:
            raise ValueError(f"Unsupported descriptor: {name}")
        values.append(float(value))
    return tuple(values)


def compute_descriptors(
    mol: Chem.Mol,
    names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES,
) -> Mapping[str, float]:
    """Return descriptor names and values while preserving requested order."""
    requested_names = tuple(names)
    return dict(zip(requested_names, compute_descriptor_values(mol, requested_names)))
