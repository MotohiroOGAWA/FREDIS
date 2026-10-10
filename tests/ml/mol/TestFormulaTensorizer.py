from __future__ import annotations

import unittest

import torch

from fredis.libs.mmkit.mmkit import Adduct, Formula
from fredis.ml.mol import FormulaTensorizer


class TestFormulaTensorizer(unittest.TestCase):
    def test_from_symbols_and_adducts_extends_symbol_order_with_adduct_elements(self) -> None:
        tensorizer = FormulaTensorizer.from_symbols_and_adducts(
            symbols=("C", "N", "O", "Cl"),
            adducts=(
                Adduct.parse("[M+H]+"),
                Adduct.parse("[M+Na]+"),
                Adduct.parse("[M+K]+"),
            ),
        )

        self.assertEqual(
            tensorizer.element_order,
            ("C", "N", "O", "Cl", "H", "Na", "K"),
        )

    def test_formula_round_trip_uses_configured_element_order(self) -> None:
        tensorizer = FormulaTensorizer.from_symbols_and_adducts(
            symbols=("C", "H", "O"),
        )

        formula = Formula.parse("C6H6O")
        encoded = tensorizer.formula_to_tensor(formula)

        self.assertTrue(
            torch.equal(
                encoded,
                torch.tensor([6.0, 6.0, 1.0, 0.0]),
            )
        )
        self.assertEqual(str(tensorizer.tensor_to_formula(encoded)), "C6H6O")

    def test_adduct_to_delta_tensor_includes_element_and_charge_delta(self) -> None:
        tensorizer = FormulaTensorizer.from_symbols_and_adducts(
            symbols=("C", "H"),
            adducts=(Adduct.parse("[M+Na]+"),),
        )

        delta = tensorizer.adduct_to_delta_tensor(Adduct.parse("[M+Na]+"))

        self.assertEqual(tensorizer.element_order, ("C", "H", "Na"))
        self.assertTrue(
            torch.equal(
                delta,
                torch.tensor([0.0, 0.0, 1.0, 1.0]),
            )
        )

    def test_formula_with_unknown_element_raises(self) -> None:
        tensorizer = FormulaTensorizer.from_symbols_and_adducts(symbols=("C", "H"))

        with self.assertRaises(ValueError):
            tensorizer.formula_to_tensor(Formula.parse("C6H6O"))

    def test_adduct_with_unknown_element_raises(self) -> None:
        tensorizer = FormulaTensorizer.from_symbols_and_adducts(symbols=("C", "H"))

        with self.assertRaises(ValueError):
            tensorizer.adduct_to_delta_tensor(Adduct.parse("[M+Na]+"))


if __name__ == "__main__":
    unittest.main()
