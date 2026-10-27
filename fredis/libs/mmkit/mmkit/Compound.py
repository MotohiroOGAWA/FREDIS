from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from bidict import bidict
from rdkit import Chem

from .Formula import Formula


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Compound:
    """
    Represent a chemical compound as an RDKit molecule.

    A ``Compound`` stores a canonicalized RDKit molecule together with
    derived properties such as molecular formula, exact mass, and charge.
    Atom map numbers are assigned during initialization.

    Notes
    -----
    This class is frozen. Atom-map assignment is an internal initialization
    detail and is not exposed as a public mutation method.
    """

    _mol: Chem.Mol = field(init=False, repr=False)
    _formula: Formula = field(init=False, repr=False)

    def __init__(
        self,
        mol: Chem.Mol,
        overwrite_atom_map: bool = False,
    ) -> None:
        """
        Initialize a compound from an RDKit molecule.

        Parameters
        ----------
        mol:
            RDKit molecule object.

        overwrite_atom_map:
            Whether to overwrite existing atom map numbers during initialization.

        Raises
        ------
        TypeError
            If ``mol`` is not an RDKit molecule.

        ValueError
            If ``mol`` cannot be converted into a valid canonical molecule.
        """
        if not isinstance(mol, Chem.Mol):
            raise TypeError("mol must be an RDKit Mol object.")

        # Atom map numbers participate in RDKit's canonical atom ordering. Keep
        # their association with the input atom indices, but canonicalize a copy
        # from which they have been removed.
        mol_without_atom_map = Chem.Mol(mol)
        original_atom_maps = [
            atom.GetAtomMapNum()
            for atom in mol_without_atom_map.GetAtoms()
        ]

        for atom in mol_without_atom_map.GetAtoms():
            atom.SetAtomMapNum(0)

        smiles = Chem.MolToSmiles(mol_without_atom_map, canonical=True)
        atom_output_order = self._smiles_atom_output_order(mol_without_atom_map)
        canonical_mol = Chem.MolFromSmiles(smiles)

        if canonical_mol is None:
            raise ValueError("Failed to construct a canonical molecule from the input.")

        if canonical_mol.GetNumAtoms() != len(atom_output_order):
            raise ValueError("Atom count mismatch after canonicalization.")

        if not overwrite_atom_map:
            for new_idx, old_idx in enumerate(atom_output_order):
                canonical_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(
                    original_atom_maps[old_idx]
                )

        mapped_mol = self._assign_atom_map(
            canonical_mol,
            overwrite=overwrite_atom_map,
        )

        formula = Formula.from_mol(mapped_mol)

        object.__setattr__(self, "_mol", mapped_mol)
        object.__setattr__(self, "_formula", formula)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        overwrite_atom_map: bool = False,
    ) -> Compound:
        """
        Create a compound from a SMILES string.

        Parameters
        ----------
        smiles:
            Input SMILES string.

        overwrite_atom_map:
            Whether to overwrite existing atom map numbers during initialization.

        Returns
        -------
        Compound
            Compound created from the SMILES string.

        Raises
        ------
        ValueError
            If the SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        return cls(
            mol=mol,
            overwrite_atom_map=overwrite_atom_map,
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """
        Convert the compound to debug representation.
        """
        return f"Compound(smiles={self.smiles})"

    def __str__(self) -> str:
        """
        Convert the compound to canonical SMILES.
        """
        return self.smiles

    # ------------------------------------------------------------------
    # Comparison / hashing
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        """
        Compare compounds by canonical SMILES without atom map numbers.
        """
        if not isinstance(other, Compound):
            return False

        return self.smiles == other.smiles

    def __hash__(self) -> int:
        """
        Hash by canonical SMILES without atom map numbers.
        """
        return hash(self.smiles)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def mol(self) -> Chem.Mol:
        """
        RDKit molecule without atom map numbers.

        Returns
        -------
        Chem.Mol
            Copy of the internal molecule with atom map numbers cleared.
        """
        mol = Chem.Mol(self._mol)

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

        return mol

    @property
    def mapped_mol(self) -> Chem.Mol:
        """
        RDKit molecule with atom map numbers.

        Returns
        -------
        Chem.Mol
            Copy of the internal molecule including atom map numbers.
        """
        return Chem.Mol(self._mol)

    @property
    def smiles(self) -> str:
        """
        Canonical SMILES string without atom map numbers.
        """
        return Chem.MolToSmiles(
            self.mol,
            canonical=True,
        )

    @property
    def mapped_smiles(self) -> str:
        """
        Canonical SMILES string with atom map numbers.
        """
        return Chem.MolToSmiles(
            self._mol,
            canonical=True,
        )

    @property
    def atom_index_mapped_smiles(self) -> str:
        """
        SMILES string with atom indices used as atom map numbers.

        Returns
        -------
        str
            SMILES string in which each atom map number equals its atom index.
        """
        mol = Chem.MolFromSmiles(self.smiles)

        if mol is None:
            raise ValueError("Failed to reconstruct molecule from canonical SMILES.")

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        return Chem.MolToSmiles(
            mol,
            canonical=True,
        )

    @property
    def formula(self) -> Formula:
        """
        Molecular formula of the compound.
        """
        return self._formula.copy()

    @property
    def atom_map_to_index(self) -> bidict[int, int]:
        """
        Mapping from atom map number to canonical atom index.

        Returns
        -------
        bidict[int, int]
            Bidirectional mapping from atom map number to atom index.
        """
        mol = Chem.Mol(self._mol)

        old_atom_map_num = {
            atom.GetIdx(): atom.GetAtomMapNum()
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

        Chem.MolToSmiles(
            mol,
            canonical=True,
        )

        atom_order_text = mol.GetProp("_smilesAtomOutputOrder")
        atom_order = [
            int(value)
            for value in atom_order_text.replace("[", "").replace("]", "").split(",")
            if value != ""
        ]

        if len(atom_order) != len(old_atom_map_num):
            raise ValueError("Atom count mismatch after canonicalization.")

        atom_map_to_idx: dict[int, int] = {}

        for new_idx, old_idx in enumerate(atom_order):
            atom_map_num = old_atom_map_num.get(old_idx)

            if atom_map_num is not None:
                atom_map_to_idx[atom_map_num] = new_idx

        return bidict(atom_map_to_idx)

    @property
    def charge(self) -> int:
        """
        Formal charge of the compound.
        """
        return sum(
            atom.GetFormalCharge()
            for atom in self._mol.GetAtoms()
        )

    @property
    def exact_mass(self) -> float:
        """
        Monoisotopic exact mass of the compound.
        """
        return self.formula.exact_mass

    # ------------------------------------------------------------------
    # Copy utilities
    # ------------------------------------------------------------------
    def copy(self) -> Compound:
        """
        Create a copy of the compound.
        """
        return Compound(
            self._mol,
            overwrite_atom_map=True,
        )

    def get_atom_index_from_map(self, map_num: int) -> Optional[int]:
        """
        Get the atom index corresponding to an atom map number.

        Parameters
        ----------
        map_num:
            Atom map number.

        Returns
        -------
        int or None
            Atom index if found, otherwise ``None``.
        """
        for atom in self._mol.GetAtoms():
            if atom.GetAtomMapNum() == map_num:
                return atom.GetIdx()

        return None

    # ------------------------------------------------------------------
    # Internal atom-map utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _smiles_atom_output_order(mol: Chem.Mol) -> list[int]:
        """Return the input atom indices in the last SMILES output order."""
        atom_order_text = mol.GetProp("_smilesAtomOutputOrder")
        return [
            int(value)
            for value in atom_order_text.strip("[]").split(",")
            if value
        ]

    @staticmethod
    def _assign_atom_map(
        mol: Chem.Mol,
        *,
        overwrite: bool = False,
        atom_map_dict: Optional[Dict[int, int]] = None,
    ) -> Chem.Mol:
        """
        Return a copied molecule with atom map numbers assigned.

        This is an internal helper. It does not mutate ``Compound`` itself.

        Parameters
        ----------
        mol:
            Source molecule.

        overwrite:
            Whether to overwrite existing atom map numbers.

        atom_map_dict:
            Optional dictionary to store old-to-new atom map number mappings.

        Returns
        -------
        Chem.Mol
            Copied molecule with atom map numbers assigned.
        """
        if atom_map_dict is not None:
            if not isinstance(atom_map_dict, dict):
                raise TypeError("atom_map_dict must be a dict if provided.")

            if len(atom_map_dict) != 0:
                raise ValueError("atom_map_dict must be empty if provided.")

        mapped_mol = Chem.Mol(mol)
        n_atoms = mapped_mol.GetNumAtoms()

        if overwrite:
            next_map_nums = list(range(1, n_atoms + 1))
        else:
            used_map_nums = {
                atom.GetAtomMapNum()
                for atom in mapped_mol.GetAtoms()
                if atom.GetAtomMapNum() > 0
            }

            max_map_num = max(used_map_nums, default=0)

            missing_map_nums = sorted(
                set(range(1, max_map_num)).difference(used_map_nums)
            )

            next_map_nums = missing_map_nums + list(
                range(max_map_num + 1, max_map_num + n_atoms + 1)
            )

        for atom in mapped_mol.GetAtoms():
            if overwrite or atom.GetAtomMapNum() == 0:
                old_map_num = atom.GetAtomMapNum()
                new_map_num = next_map_nums.pop(0)

                atom.SetAtomMapNum(new_map_num)

                if atom_map_dict is not None and old_map_num > 0:
                    atom_map_dict[old_map_num] = new_map_num

        return mapped_mol
