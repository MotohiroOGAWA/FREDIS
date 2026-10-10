from __future__ import annotations

import unittest

from fredis.domain.formula._assignment import max_neutral_precursor_formula
from fredis.domain.formula.utils import (
    assign_formulas_to_peaks,
    make_formula_assignment_candidate,
    parse_formula_assignment_label,
)
from fredis.domain.mass.tolerance import parse_mass_tolerance
from fredis.libs.mmkit.mmkit import Adduct, Formula


class TestFormulaAssignment(unittest.TestCase):
    def test_grouping_formula_uses_elementwise_max_for_positive_adduct(self) -> None:
        original = Formula.parse("C6H6")
        precursor = Adduct.parse("[M+H]+").apply_to_formula(original)

        self.assertEqual(
            str(max_neutral_precursor_formula(original, precursor)),
            "C6H7",
        )

    def test_grouping_formula_uses_original_for_negative_h_loss(self) -> None:
        original = Formula.parse("C6H6")
        precursor = Adduct.parse("[M-H]-").apply_to_formula(original)

        self.assertEqual(
            str(max_neutral_precursor_formula(original, precursor)),
            "C6H6",
        )

    def test_parse_neutral_loss_formula_label(self) -> None:
        candidate = parse_formula_assignment_label("C5H3(-CH3)")

        self.assertEqual(str(candidate.formula), "C5H3")
        self.assertEqual(candidate.label, "C5H3(-CH3)")

    def test_rejects_non_negative_neutral_loss(self) -> None:
        with self.assertRaises(ValueError):
            parse_formula_assignment_label("C5H3(CH3)")

    def test_make_candidate_labels_neutral_loss_from_original_formula(self) -> None:
        candidate = make_formula_assignment_candidate(
            Formula.parse("C5H3"),
            original_formula=Formula.parse("C6H6"),
        )

        self.assertEqual(candidate.label, "C5H3(-CH3)")

    def test_assign_formulas_to_peaks_defaults_to_plain_formula_label(self) -> None:
        formula = Formula.parse("C5H3")
        results = assign_formulas_to_peaks(
            peaks_mz=[formula.exact_mass],
            formula_candidates=[formula],
            mass_tolerance=parse_mass_tolerance("0.001Da"),
        )

        self.assertEqual(results[0]["matched_formulas"], ["C5H3"])

    def test_assign_formulas_to_peaks_outputs_neutral_loss_label_when_enabled(self) -> None:
        formula = Formula.parse("C5H3")
        results = assign_formulas_to_peaks(
            peaks_mz=[formula.exact_mass],
            formula_candidates=[formula],
            mass_tolerance=parse_mass_tolerance("0.001Da"),
            original_formula=Formula.parse("C6H6"),
        )

        self.assertEqual(results[0]["matched_formulas"], ["C5H3(-CH3)"])


if __name__ == "__main__":
    unittest.main()
