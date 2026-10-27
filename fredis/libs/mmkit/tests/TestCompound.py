import unittest

from rdkit import Chem

from mmkit.Compound import Compound
from mmkit.Formula import Formula


class TestCompound(unittest.TestCase):
    """Unit tests for Compound."""

    def setUp(self) -> None:
        """Prepare commonly used Compound objects."""
        self.ethanol = Compound.from_smiles("CCO")
        self.ammonium = Compound.from_smiles("[NH4+]")
        self.mapped_ethanol = Compound.from_smiles("[CH3:1][CH2:2][OH:3]")
        self.benzene = Compound.from_smiles("c1ccccc1")

    def test_init_from_mol(self) -> None:
        """Construct a compound from an RDKit Mol object."""
        mol = Chem.MolFromSmiles("CCO")
        compound = Compound(mol)

        self.assertIsInstance(compound, Compound)
        self.assertEqual(compound.smiles, "CCO")

    def test_from_smiles(self) -> None:
        """Construct a compound from a SMILES string."""
        compound = Compound.from_smiles("CCO")

        self.assertIsInstance(compound, Compound)
        self.assertEqual(compound.smiles, "CCO")

    def test_from_smiles_invalid(self) -> None:
        """Raise ValueError for an invalid SMILES string."""
        with self.assertRaises(ValueError):
            Compound.from_smiles("invalid_smiles")

    def test_str(self) -> None:
        """Return the canonical SMILES string."""
        self.assertEqual(str(self.ethanol), "CCO")

    def test_repr(self) -> None:
        """Return the debug representation."""
        self.assertEqual(repr(self.ethanol), "Compound(smiles=CCO)")

    def test_mol_returns_copy_without_atom_map(self) -> None:
        """Return a copy of the molecule without atom map numbers."""
        mol = self.mapped_ethanol.mol

        self.assertIsInstance(mol, Chem.Mol)
        self.assertTrue(all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms()))

    def test_mapped_mol_returns_copy(self) -> None:
        """Return a copy of the molecule with atom map numbers."""
        mol = self.mapped_ethanol.mapped_mol

        self.assertIsInstance(mol, Chem.Mol)
        self.assertEqual(
            [atom.GetAtomMapNum() for atom in mol.GetAtoms()],
            [1, 2, 3],
        )

    def test_smiles(self) -> None:
        """Return the canonical SMILES string without atom map numbers."""
        self.assertEqual(self.mapped_ethanol.smiles, "CCO")

    def test_mapped_smiles(self) -> None:
        """Return the canonical SMILES string with atom map numbers."""
        mapped_smiles = self.mapped_ethanol.mapped_smiles

        self.assertIn(":1", mapped_smiles)
        self.assertIn(":2", mapped_smiles)
        self.assertIn(":3", mapped_smiles)

    def test_atom_index_mapped_smiles(self) -> None:
        """Return a SMILES string with atom indices as atom map numbers."""
        smiles = self.ethanol.atom_index_mapped_smiles

        self.assertIn(":1", smiles)
        self.assertIn(":2", smiles)

    def test_formula(self) -> None:
        """Return the molecular formula."""
        self.assertEqual(self.ethanol.formula, Formula.parse("C2H6O"))

    def test_charge_neutral(self) -> None:
        """Return zero for a neutral compound."""
        self.assertEqual(self.ethanol.charge, 0)

    def test_charge_charged(self) -> None:
        """Return the formal charge of a charged compound."""
        self.assertEqual(self.ammonium.charge, 1)

    def test_exact_mass(self) -> None:
        """Return the exact mass derived from the molecular formula."""
        self.assertAlmostEqual(
            self.ethanol.exact_mass,
            self.ethanol.formula.exact_mass,
            places=6,
        )

    def test_atom_map_to_index(self) -> None:
        """Return a bidirectional mapping from atom map number to atom index."""
        mapping = self.mapped_ethanol.atom_map_to_index

        self.assertEqual(set(mapping.keys()), {1, 2, 3})
        self.assertEqual(set(mapping.values()), {0, 1, 2})

    def test_atom_map_is_assigned_on_initialization(self) -> None:
        """Assign atom map numbers during Compound initialization."""
        compound = Compound.from_smiles("CCO")

        self.assertEqual(compound.smiles, "CCO")
        self.assertTrue(
            all(
                atom.GetAtomMapNum() > 0
                for atom in compound.mapped_mol.GetAtoms()
            )
        )

    def test_overwrite_atom_map_on_initialization(self) -> None:
        """Overwrite existing atom map numbers during Compound initialization."""
        compound = Compound.from_smiles(
            "[CH3:5][CH2:7][OH:9]",
            overwrite_atom_map=True,
        )

        atom_map_nums = {
            atom.GetAtomMapNum()
            for atom in compound.mapped_mol.GetAtoms()
        }

        self.assertEqual(atom_map_nums, {1, 2, 3})

    def test_canonicalization_ignores_atom_maps_and_restores_them(self) -> None:
        """Canonicalize without maps, then restore each map to the same atom."""
        order_sensitive_mol = Chem.MolFromSmiles("[CH3:99][CH2][OH:7]")
        self.assertIsNotNone(order_sensitive_mol)
        Chem.MolToSmiles(order_sensitive_mol, canonical=True)
        mapped_order = Compound._smiles_atom_output_order(order_sensitive_mol)
        for atom in order_sensitive_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.MolToSmiles(order_sensitive_mol, canonical=True)
        unmapped_order = Compound._smiles_atom_output_order(order_sensitive_mol)
        self.assertNotEqual(mapped_order, unmapped_order)

        smiles = "[OH:30][CH2:20][NH2:10]"
        input_mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(input_mol)

        for atom in input_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        unmapped_canonical = Chem.MolToSmiles(input_mol, canonical=True)

        compound = Compound.from_smiles(smiles)

        self.assertEqual(compound.smiles, unmapped_canonical)
        self.assertEqual(
            {
                atom.GetSymbol(): atom.GetAtomMapNum()
                for atom in compound.mapped_mol.GetAtoms()
            },
            {"O": 30, "C": 20, "N": 10},
        )

    def test_partial_atom_maps_are_restored_before_missing_maps_are_filled(self) -> None:
        """Preserve existing maps and assign unused numbers to unmapped atoms."""
        compound = Compound.from_smiles("[CH3:99][CH2][OH:7]")

        maps_by_symbol_and_hydrogens = {
            (atom.GetSymbol(), atom.GetTotalNumHs()): atom.GetAtomMapNum()
            for atom in compound.mapped_mol.GetAtoms()
        }
        self.assertEqual(maps_by_symbol_and_hydrogens[("C", 3)], 99)
        self.assertEqual(maps_by_symbol_and_hydrogens[("O", 1)], 7)
        self.assertEqual(maps_by_symbol_and_hydrogens[("C", 2)], 1)

    def test_overwrite_does_not_restore_original_atom_maps(self) -> None:
        """Assign fresh maps after map-independent canonicalization on overwrite."""
        compound = Compound.from_smiles(
            "[OH:30][CH2:20][NH2:10]",
            overwrite_atom_map=True,
        )

        atom_map_nums = [
            atom.GetAtomMapNum()
            for atom in compound.mapped_mol.GetAtoms()
        ]
        self.assertEqual(atom_map_nums, [1, 2, 3])
        self.assertTrue({10, 20, 30}.isdisjoint(atom_map_nums))

    def test_assign_atom_map_helper_with_mapping_dict(self) -> None:
        """Store old-to-new atom map number mappings in the internal helper."""
        mol = Chem.MolFromSmiles("[CH3:5][CH2:7][OH:9]")
        self.assertIsNotNone(mol)

        mapping_dict: dict[int, int] = {}

        mapped_mol = Compound._assign_atom_map(
            mol,
            overwrite=True,
            atom_map_dict=mapping_dict,
        )

        self.assertEqual(set(mapping_dict.keys()), {5, 7, 9})
        self.assertEqual(set(mapping_dict.values()), {1, 2, 3})
        self.assertTrue(
            all(
                atom.GetAtomMapNum() > 0
                for atom in mapped_mol.GetAtoms()
            )
        )

    def test_assign_atom_map_helper_rejects_nonempty_dict(self) -> None:
        """Raise ValueError when atom_map_dict is not empty."""
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)

        with self.assertRaises(ValueError):
            Compound._assign_atom_map(
                mol,
                atom_map_dict={1: 2},
            )

    def test_copy(self) -> None:
        """Create an independent copy of the compound."""
        copied = self.ethanol.copy()

        self.assertIsInstance(copied, Compound)
        self.assertIsNot(copied, self.ethanol)
        self.assertEqual(copied.smiles, self.ethanol.smiles)
        self.assertEqual(copied.formula, self.ethanol.formula)

    def test_get_atom_index_from_map_found(self) -> None:
        """Return the atom index for an existing atom map number."""
        idx = self.mapped_ethanol.get_atom_index_from_map(2)

        self.assertIsInstance(idx, int)
        self.assertIn(idx, {0, 1, 2})

    def test_get_atom_index_from_map_not_found(self) -> None:
        """Return None when the atom map number is absent."""
        self.assertIsNone(self.mapped_ethanol.get_atom_index_from_map(999))

    def test_benzene_formula(self) -> None:
        """Return the correct formula for an aromatic compound."""
        self.assertEqual(self.benzene.formula, Formula.parse("C6H6"))


if __name__ == "__main__":
    unittest.main()
