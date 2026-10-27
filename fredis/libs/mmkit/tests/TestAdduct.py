import unittest

from mmkit.Formula import Formula
from mmkit.Adduct import Adduct


class TestAdduct(unittest.TestCase):
    """Unit tests for Adduct."""

    def test_init_basic(self):
        adduct = Adduct.from_dict(
            {"H": 1},
            ion_type="M",
            n_molecules=1,
            charge=1,
        )

        self.assertEqual(adduct.ion_type, "M")
        self.assertEqual(adduct.n_molecules, 1)
        self.assertEqual(adduct.charge, 1)
        self.assertEqual(adduct.get_formula_count("H"), 1)

    def test_from_dict_positive(self):
        adduct = Adduct.from_dict(
            {"H": 1},
            charge=1,
        )

        self.assertEqual(str(adduct), "[M+H]+")
        self.assertEqual(adduct.get_formula_count("H"), 1)

    def test_from_dict_negative(self):
        adduct = Adduct.from_dict(
            {"H": -1},
            charge=-1,
        )

        self.assertEqual(str(adduct), "[M-H]-")
        self.assertEqual(adduct.get_formula_count("H"), -1)

    def test_from_dict_multiple_negative_hydrogen(self):
        adduct = Adduct.from_dict(
            {"H": -5},
            charge=-1,
        )

        self.assertEqual(str(adduct), "[M-5H]-")
        self.assertEqual(adduct.get_formula_count("H"), -5)

    def test_from_dict_with_formula_key(self):
        adduct = Adduct.from_dict(
            {Formula.parse("H"): -5},
            charge=-1,
        )

        self.assertEqual(str(adduct), "[M-5H]-")
        self.assertEqual(adduct.get_formula_count("H"), -5)

    def test_from_adducts_compatibility(self):
        adduct = Adduct.from_adducts(
            ion_type="M",
            n_molecules=1,
            adducts_in=(Formula.parse("H"),),
            charge=1,
        )

        self.assertEqual(str(adduct), "[M+H]+")
        self.assertEqual(adduct.get_formula_count("H"), 1)

    def test_str_positive(self):
        adduct = Adduct.from_dict(
            {"H": 1},
            ion_type="M",
            charge=1,
        )

        self.assertEqual(str(adduct), "[M+H]+")

    def test_str_negative(self):
        adduct = Adduct.from_dict(
            {"H": -1},
            ion_type="M",
            charge=-1,
        )

        self.assertEqual(str(adduct), "[M-H]-")

    def test_str_multiple_molecules(self):
        adduct = Adduct.from_dict(
            {"Na": 1},
            ion_type="M",
            n_molecules=2,
            charge=1,
        )

        self.assertEqual(str(adduct), "[2M+Na]+")

    def test_repr(self):
        adduct = Adduct.from_dict(
            {"H": 1},
            ion_type="M",
            charge=1,
        )

        self.assertEqual(repr(adduct), "Adduct([M+H]+)")

    def test_eq_and_hash(self):
        adduct1 = Adduct.parse("[M+H]+")
        adduct2 = Adduct.from_dict(
            {"H": 1},
            ion_type="M",
            charge=1,
        )

        self.assertEqual(adduct1, adduct2)
        self.assertEqual(hash(adduct1), hash(adduct2))

    def test_adduct_formulas_returns_copy(self):
        adduct = Adduct.parse("[M+H]+")
        d = adduct.adduct_formulas

        d[Formula.parse("Na")] = 100

        self.assertEqual(adduct.get_formula_count("Na"), 0)

    def test_to_dict(self):
        adduct = Adduct.parse("[M-5H]-")

        self.assertEqual(adduct.to_dict(), {"H": -5})

    def test_formula_diff(self):
        adduct = Adduct.parse("[M+H]+")

        self.assertEqual(adduct.formula_diff, Formula.parse("H"))

    def test_mass_shift(self):
        adduct = Adduct.parse("[M+H]+")

        self.assertAlmostEqual(
            adduct.mass_shift,
            Formula.parse("H").exact_mass,
            places=6,
        )

    def test_element_diff(self):
        adduct = Adduct.parse("[M+H]+")
        self.assertEqual(adduct.element_diff["H"], 1)

        adduct2 = Adduct.parse("[M-H]-")
        self.assertEqual(adduct2.element_diff["H"], -1)

    def test_get_formula_count_with_formula(self):
        adduct = Adduct.parse("[M+H]+")

        self.assertEqual(adduct.get_formula_count(Formula.parse("H")), 1)
        self.assertEqual(adduct.get_formula_count(Formula.parse("Na")), 0)

    def test_get_formula_count_with_charged_formula(self):
        adduct = Adduct.parse("[M+H]+")

        self.assertEqual(adduct.get_formula_count(Formula.parse("H+")), 1)
        self.assertEqual(adduct.get_formula_count(Formula.parse("Na+")), 0)

    def test_get_formula_count_with_string(self):
        adduct = Adduct.parse("[M+H-H2O]+")

        self.assertEqual(adduct.get_formula_count("H"), 1)
        self.assertEqual(adduct.get_formula_count("H+"), 1)
        self.assertEqual(adduct.get_formula_count("H2O"), -1)
        self.assertEqual(adduct.get_formula_count("Na"), 0)

    def test_copy(self):
        adduct1 = Adduct.parse("[2M+Na]+")
        adduct2 = adduct1.copy()

        self.assertEqual(adduct1, adduct2)
        self.assertIsNot(adduct1, adduct2)
        self.assertEqual(adduct1.charge, adduct2.charge)

    def test_with_charge(self):
        adduct1 = Adduct.from_dict({"H": 1})
        adduct2 = adduct1.with_charge(1)

        self.assertEqual(str(adduct1), "[M+H]")
        self.assertEqual(str(adduct2), "[M+H]+")
        self.assertEqual(adduct1.charge, 0)
        self.assertEqual(adduct2.charge, 1)

    def test_parse_positive(self):
        adduct = Adduct.parse("[M+H]+")

        self.assertEqual(adduct.ion_type, "M")
        self.assertEqual(adduct.n_molecules, 1)
        self.assertEqual(adduct.charge, 1)
        self.assertEqual(adduct.get_formula_count("H"), 1)

    def test_parse_negative(self):
        adduct = Adduct.parse("[M-H]-")

        self.assertEqual(adduct.ion_type, "M")
        self.assertEqual(adduct.n_molecules, 1)
        self.assertEqual(adduct.charge, -1)
        self.assertEqual(adduct.get_formula_count("H"), -1)
        self.assertEqual(adduct.get_formula_count("H+"), -1)

    def test_parse_multiple_molecules(self):
        adduct = Adduct.parse("[2M+Na]+")

        self.assertEqual(adduct.ion_type, "M")
        self.assertEqual(adduct.n_molecules, 2)
        self.assertEqual(adduct.charge, 1)
        self.assertEqual(adduct.get_formula_count("Na"), 1)

    def test_parse_complex(self):
        adduct = Adduct.parse("[M+HCOOH-H]-")

        self.assertEqual(adduct.charge, -1)
        self.assertEqual(adduct.get_formula_count("HCOOH"), 1)
        self.assertEqual(adduct.get_formula_count("H"), -1)

    def test_parse_invalid_format(self):
        invalid_cases = [
            "M+H+",
            "",
            "abc",
            "[+H]",
        ]

        for s in invalid_cases:
            with self.subTest(adduct_str=s):
                with self.assertRaises(ValueError):
                    Adduct.parse(s)

    def test_add_same_adduct_type(self):
        adduct1 = Adduct.parse("[M+H]+")
        adduct2 = Adduct.parse("[M+Na]+")
        adduct3 = adduct1.add(adduct2, prefer_charge=True)

        self.assertEqual(adduct3.ion_type, "M")
        self.assertEqual(adduct3.n_molecules, 1)
        self.assertEqual(adduct3.charge, 1)
        self.assertEqual(adduct3.get_formula_count("H"), 1)
        self.assertEqual(adduct3.get_formula_count("Na"), 1)

    def test_add_different_ion_type_raises(self):
        adduct1 = Adduct(ion_type="M")
        adduct2 = Adduct(ion_type="F")

        with self.assertRaises(ValueError):
            adduct1.add(adduct2)

    def test_add_different_n_molecules_raises(self):
        adduct1 = Adduct.parse("[M+H]+")
        adduct2 = Adduct.parse("[2M+H]+")

        with self.assertRaises(ValueError):
            adduct1.add(adduct2)

    def test_add_different_charge_raises(self):
        adduct1 = Adduct.parse("[M+H]+")
        adduct2 = Adduct.parse("[M-H]-")

        with self.assertRaises(ValueError):
            adduct1.add(adduct2)

    def test_add_prefer_self(self):
        adduct1 = Adduct.parse("[M+H]+")
        adduct2 = Adduct.parse("[2M+Na]+")

        adduct3 = adduct1.add_prefer_self(adduct2)

        self.assertEqual(adduct3.ion_type, "M")
        self.assertEqual(adduct3.n_molecules, 1)
        self.assertEqual(adduct3.charge, 1)
        self.assertEqual(adduct3.get_formula_count("H"), 1)
        self.assertEqual(adduct3.get_formula_count("Na"), 1)

    def test_apply_to_formula(self):
        adduct = Adduct.parse("[M+H]+")
        neutral = Formula.parse("C6H12O6")
        ion = adduct.apply_to_formula(neutral)

        expected = Formula.parse("C6H13O6+")
        self.assertEqual(ion, expected)
        self.assertEqual(ion.charge, 1)

    def test_apply_to_mass(self):
        adduct = Adduct.parse("[M+H]+")
        neutral_mass = 100.0

        self.assertAlmostEqual(
            adduct.apply_to_mass(neutral_mass),
            100.0 + Formula.parse("H").exact_mass,
            places=6,
        )

    def test_apply_to_mz(self):
        adduct = Adduct.parse("[M+H]+")
        neutral_mass = 100.0
        expected = 100.0 + Formula.parse("H").exact_mass

        self.assertAlmostEqual(adduct.apply_to_mz(neutral_mass), expected, places=6)

    def test_apply_to_mz_raises_for_zero_charge(self):
        adduct = Adduct.from_dict({"H": 1})

        with self.assertRaises(ValueError):
            adduct.apply_to_mz(100.0)

    def test_split_grouped(self):
        adduct = Adduct.parse("[M+H-H2O]+")
        parts = adduct.split(split_each=False)

        self.assertEqual(len(parts), 2)
        self.assertEqual(str(parts[0]), "[M+H]")
        self.assertEqual(str(parts[1]), "[M-H2O]")

    def test_split_each(self):
        adduct = Adduct.parse("[M+H+Na-H2O]+")
        parts = adduct.split(split_each=True)

        self.assertEqual(len(parts), 3)
        self.assertIn("[M+H]", [str(x) for x in parts])
        self.assertIn("[M+Na]", [str(x) for x in parts])
        self.assertIn("[M-H2O]", [str(x) for x in parts])

    def test_split_by_reference_adducts_positive_single_match(self):
        adduct = Adduct.parse("[M+H]+")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(len(matched_flags), len(reference_adducts))
        self.assertEqual(matched_flags, (True, False, False, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(str(residual_component), "[M]")

    def test_split_by_reference_adducts_positive_multiple_matches(self):
        adduct = Adduct.parse("[M+H+Na+NH4]+")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(len(matched_flags), len(reference_adducts))
        self.assertEqual(matched_flags, (True, True, True, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(str(residual_component), "[M]")

    def test_split_by_reference_adducts_positive_with_residual_loss(self):
        adduct = Adduct.parse("[M+H-H2O]+")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (True, False, False, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)
        self.assertEqual(str(residual_component), "[M-H2O]")

    def test_split_by_reference_adducts_positive_with_residual_addition(self):
        adduct = Adduct.parse("[M+H+H2O]+")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (True, False, False, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H2O"), 1)
        self.assertEqual(str(residual_component), "[M+H2O]")

    def test_split_by_reference_adducts_uses_same_ion_mode_reference_only(self):
        adduct = Adduct.parse("[M-H]-")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (False, False, False, True))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(str(residual_component), "[M]")

    def test_split_by_reference_adducts_does_not_use_opposite_ion_mode_reference(self):
        adduct = Adduct.parse("[M-H]-")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (False, False, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H"), -1)
        self.assertEqual(str(residual_component), "[M-H]")

    def test_split_by_reference_adducts_positive_does_not_use_negative_reference(self):
        adduct = Adduct.parse("[M+H]+")
        reference_adducts = (
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (False,))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H"), 1)
        self.assertEqual(str(residual_component), "[M+H]")

    def test_split_by_reference_adducts_does_not_decompose_repeated_formula_into_single_reference(self):
        adduct = Adduct.parse("[M+H+H]+")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (False, False, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H"), 2)
        self.assertEqual(str(residual_component), "[M+2H]")

    def test_split_by_reference_adducts_handles_mixed_matched_and_unmatched_formulas(self):
        adduct = Adduct.parse("[M+H+Na-H2O+CH3OH]+")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
            Adduct.parse("[M-H]-"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (True, True, False, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)
        self.assertEqual(residual_component.get_formula_count("CH3OH"), 1)

    def test_split_by_reference_adducts_keeps_reference_order_in_flags(self):
        adduct = Adduct.parse("[M+NH4+H+Na]+")
        reference_adducts = (
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+NH4]+"),
            Adduct.parse("[M+H]+"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(len(matched_flags), len(reference_adducts))
        self.assertEqual(matched_flags, (True, True, True))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(str(residual_component), "[M]")

    def test_split_by_reference_adducts_does_not_match_reference_with_different_n_molecules(self):
        adduct = Adduct.parse("[2M+H-H2O]+")
        reference_adducts = (
            Adduct.parse("[M+H]+"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (False,))
        self.assertEqual(residual_component.ion_type, "M")
        self.assertEqual(residual_component.n_molecules, 1)
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H"), 1)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)
        self.assertEqual(str(residual_component), "[M-H2O+H]")

    def test_split_by_reference_adducts_matches_reference_with_same_n_molecules(self):
        adduct = Adduct.parse("[2M+H-H2O]+")
        reference_adducts = (
            Adduct.parse("[2M+H]+"),
            Adduct.parse("[M+H]+"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (True, False))
        self.assertEqual(residual_component.ion_type, "M")
        self.assertEqual(residual_component.n_molecules, 1)
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)
        self.assertEqual(str(residual_component), "[M-H2O]")

    def test_split_by_reference_adducts_matches_reference_with_multiple_same_formula_count(self):
        adduct = Adduct.parse("[M+2Na-H2O]+")
        reference_adducts = (
            Adduct.parse("[M+2Na]+"),
            Adduct.parse("[M+Na]+"),
            Adduct.parse("[M+H]+"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (True, False, False))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)
        self.assertEqual(str(residual_component), "[M-H2O]")

    def test_split_by_reference_adducts_does_not_decompose_multiple_formula_into_single_reference_count(self):
        adduct = Adduct.parse("[M+2Na-H2O]+")
        reference_adducts = (
            Adduct.parse("[M+Na]+"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (False,))
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("Na"), 2)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)
        self.assertEqual(str(residual_component), "[M+2Na-H2O]")

    def test_split_by_reference_adducts_does_not_match_reference_with_larger_formula_count(self):
        adduct = Adduct.parse("[M+Na-H2O]+")
        reference_adducts = (
            Adduct.parse("[M+2Na]+"),
        )

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, (False,))
        self.assertEqual(residual_component.ion_type, "M")
        self.assertEqual(residual_component.n_molecules, 1)
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("Na"), 1)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)
        self.assertEqual(str(residual_component), "[M+Na-H2O]")

    def test_split_by_reference_adducts_with_no_reference_adducts(self):
        adduct = Adduct.parse("[M+H+Na-H2O]+")
        reference_adducts = tuple()

        matched_flags, residual_component = Adduct.split_by_reference_adducts(
            adduct=adduct,
            reference_adducts=reference_adducts,
        )

        self.assertEqual(matched_flags, tuple())
        self.assertEqual(residual_component.charge, 0)
        self.assertEqual(residual_component.get_formula_count("H"), 1)
        self.assertEqual(residual_component.get_formula_count("Na"), 1)
        self.assertEqual(residual_component.get_formula_count("H2O"), -1)


if __name__ == "__main__":
    unittest.main()