import unittest
from collections import OrderedDict

from rdkit import Chem

from mmkit.Formula import Formula


class TestFormula(unittest.TestCase):
    """Unit tests for Formula."""

    def setUp(self) -> None:
        """Prepare commonly used Formula objects."""
        self.glucose = Formula(
            OrderedDict([("C", 6), ("H", 12), ("O", 6)]),
            charge=0,
            raw_formula="C6H12O6",
        )
        self.water = Formula(
            OrderedDict([("H", 2), ("O", 1)]),
            charge=0,
            raw_formula="H2O",
        )
        self.sodium = Formula(
            OrderedDict([("Na", 1)]),
            charge=1,
            raw_formula="Na+",
        )
        self.negative_group = Formula(
            OrderedDict([("C", -1), ("H", -4), ("O", -1)]),
            charge=0,
            raw_formula="-(CH3OH)",
        )
        self.glucose_minus_na = Formula(
            OrderedDict([("C", 6), ("H", 12), ("O", 6), ("Na", -1)]),
            charge=-1,
            raw_formula="C6H12O6-Na-",
        )

    # ------------------------------------------------------------------
    # Construction / parsing
    # ------------------------------------------------------------------
    def test_parse(self) -> None:
        """Formula.parse should correctly parse formula strings."""
        cases = [
            ("C6H12O6", self.glucose),
            ("HOH", self.water),
            ("Na+", self.sodium),
            ("-H5-C-OH", self.negative_group),
            ("C6H12-NaO6-", self.glucose_minus_na),
        ]

        for formula_str, expected in cases:
            with self.subTest(formula_str=formula_str):
                parsed = Formula.parse(formula_str)
                self.assertEqual(parsed, expected)

    def test_parse_store_raw(self) -> None:
        """Formula.parse should optionally preserve the raw input string."""
        f1 = Formula.parse("C6O6H12", store_raw=True)
        f2 = Formula.parse("C6O6H12", store_raw=False)

        self.assertEqual(f1.raw_formula, "C6O6H12")
        self.assertEqual(f2.raw_formula, "C6H12O6")

    def test_from_mol(self) -> None:
        """Formula.from_mol should build a Formula from an RDKit Mol."""
        mol = Chem.MolFromSmiles("O")  # water
        formula = Formula.from_mol(mol)

        self.assertEqual(formula.value, "H2O")

    def test_empty(self) -> None:
        """Formula.empty should return an empty neutral formula."""
        f = Formula.empty()
        self.assertEqual(f.value, "")
        self.assertEqual(f.charge, 0)
        self.assertEqual(f.elements, OrderedDict())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def test_exact_mass(self) -> None:
        """exact_mass should match expected values."""
        cases = [
            (self.glucose, 180.06339),
            (self.water, 18.01056),
            (self.sodium, 22.98977),
            (self.negative_group, -32.02621),
            (self.glucose_minus_na, 157.07362),
        ]

        for formula, expected_mass in cases:
            with self.subTest(formula=str(formula)):
                self.assertAlmostEqual(formula.exact_mass, expected_mass, places=4)

    def test_string_properties(self) -> None:
        """value and plain_value should return correct strings."""
        cases = [
            (self.glucose, "C6H12O6", "C6H12O6"),
            (self.water, "H2O", "H2O"),
            (self.sodium, "Na+", "Na"),
            (self.negative_group, "-C-H4-O", "-C-H4-O"),
            (self.glucose_minus_na, "C6H12-NaO6-", "C6H12-NaO6"),
        ]

        for formula, expected_value, expected_plain in cases:
            with self.subTest(formula=str(formula)):
                self.assertEqual(formula.value, expected_value)
                self.assertEqual(formula.plain_value, expected_plain)
                self.assertEqual(formula.to_string(no_charge=False), expected_value)
                self.assertEqual(formula.to_string(no_charge=True), expected_plain)

    def test_is_nonnegative(self) -> None:
        """is_nonnegative should reflect whether all element counts are >= 0."""
        self.assertTrue(self.glucose.is_nonnegative)
        self.assertTrue(self.water.is_nonnegative)
        self.assertFalse(self.negative_group.is_nonnegative)
        self.assertFalse(self.glucose_minus_na.is_nonnegative)

    def test_reorder_elements(self) -> None:
        """Elements should be stored in Hill order."""
        f = Formula(
            OrderedDict([("O", 6), ("C", 6), ("H", 12)]),
            charge=0,
            raw_formula="unordered",
        )

        self.assertEqual(list(f.elements.keys()), ["C", "H", "O"])

    # ------------------------------------------------------------------
    # Copy / normalization
    # ------------------------------------------------------------------
    def test_copy(self) -> None:
        """copy should return an equal but independent instance."""
        copied = self.glucose.copy()

        self.assertEqual(copied, self.glucose)
        self.assertIsNot(copied, self.glucose)
        self.assertEqual(copied.elements, self.glucose.elements)
        self.assertEqual(copied.charge, self.glucose.charge)
        self.assertEqual(copied.raw_formula, self.glucose.raw_formula)

    def test_plain(self) -> None:
        """plain should remove charge only."""
        f = self.sodium.plain
        self.assertEqual(f.value, "Na")
        self.assertEqual(f.charge, 0)
        self.assertEqual(f.raw_formula, "Na+")

    def test_normalized(self) -> None:
        """normalized should clear raw_formula but keep charge."""
        f = self.sodium.normalized
        self.assertEqual(f.value, "Na+")
        self.assertEqual(f._raw_formula, "")

    def test_normalized_plain(self) -> None:
        """normalized_plain should clear both raw_formula and charge."""
        f = self.sodium.normalized_plain
        self.assertEqual(f.value, "Na")
        self.assertEqual(f._raw_formula, "")
        self.assertEqual(f.charge, 0)

    # ------------------------------------------------------------------
    # Equality / hash
    # ------------------------------------------------------------------
    def test_equality(self) -> None:
        """Equivalent formulas should compare equal."""
        f1 = Formula.parse("HOH")
        f2 = Formula.parse("H2O")

        self.assertEqual(f1, f2)

    def test_eq_ignore_raw(self) -> None:
        """eq should optionally compare raw_formula."""
        f1 = Formula.parse("HOH", store_raw=True)
        f2 = Formula.parse("H2O", store_raw=True)

        self.assertTrue(f1.eq(f2, ignore_raw=True))
        self.assertFalse(f1.eq(f2, ignore_raw=False))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def test_addition(self) -> None:
        """Addition should combine element counts and charges."""
        self.assertEqual((self.glucose + self.water).value, "C6H14O7")
        self.assertEqual((self.sodium + self.water).value, "H2NaO+")
        self.assertEqual((self.negative_group + self.water).value, "-C-H2")

    def test_subtraction(self) -> None:
        """Subtraction should subtract element counts and charges."""
        self.assertEqual((self.glucose - self.water).value, "C6H10O5")
        self.assertEqual((self.glucose_minus_na - self.sodium).value, "C6H12-Na2O6-2")
        self.assertEqual((self.water - self.negative_group).value, "CH6O2")

    def test_multiplication(self) -> None:
        """Multiplication should scale element counts and charge."""
        self.assertEqual((self.water * 2).value, "H4O2")
        self.assertEqual((3 * self.glucose).value, "C18H36O18")
        self.assertEqual((self.sodium * 2).value, "Na2+2")
        self.assertEqual((self.negative_group * 2).value, "-C2-H8-O2")
        self.assertEqual((self.glucose * 1).value, self.glucose.value)

    def test_multiplication_invalid_type(self) -> None:
        """Multiplication by non-int should raise TypeError."""
        with self.assertRaises(TypeError):
            _ = self.water * 1.5

    # ------------------------------------------------------------------
    # Containment
    # ------------------------------------------------------------------
    def test_contains_formula(self) -> None:
        """Containment should work for Formula objects."""
        self.assertIn(Formula.parse("H2"), self.glucose)
        self.assertIn(Formula.parse("O6"), self.glucose)
        self.assertNotIn(Formula.parse("Na"), self.glucose)

    def test_contains_string(self) -> None:
        """Containment should also accept strings."""
        self.assertIn("H2", self.glucose)
        self.assertNotIn("Na", self.glucose)

    def test_contains_negative_formula_raises(self) -> None:
        """Containment on negative formulas should raise ValueError."""
        with self.assertRaises(ValueError):
            _ = "H2" in self.negative_group

    def test_contains_invalid_type_raises(self) -> None:
        """Containment with unsupported types should raise TypeError."""
        with self.assertRaises(TypeError):
            _ = 123 in self.glucose


if __name__ == "__main__":
    unittest.main()