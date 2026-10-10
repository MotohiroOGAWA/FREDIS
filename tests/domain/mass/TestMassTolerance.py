from __future__ import annotations

import unittest

from fredis.domain.mass.tolerance import (
    DaOrPpmTolerance,
    DaTolerance,
    MassTolerance,
    PpmTolerance,
    format_mass_tolerance,
    parse_mass_tolerance,
)


class TestDaTolerance(unittest.TestCase):
    def test_init_converts_tolerance_to_float(self) -> None:
        tolerance = DaTolerance(1)

        self.assertEqual(tolerance.tolerance, 1.0)
        self.assertEqual(tolerance.unit, "Da")

    def test_init_rejects_zero_or_negative_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            DaTolerance(0)

        with self.assertRaises(ValueError):
            DaTolerance(-0.01)

    def test_init_rejects_non_numeric_tolerance(self) -> None:
        with self.assertRaises(TypeError):
            DaTolerance("0.01")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            DaTolerance(True)  # type: ignore[arg-type]

    def test_error_returns_signed_da_error(self) -> None:
        tolerance = DaTolerance(0.01)

        self.assertAlmostEqual(
            tolerance.error(
                observed=100.005,
                theoretical=100.000,
            ),
            0.005,
        )

        self.assertAlmostEqual(
            tolerance.error(
                observed=99.995,
                theoretical=100.000,
            ),
            -0.005,
        )

    def test_within_returns_true_when_error_is_within_da_tolerance(self) -> None:
        tolerance = DaTolerance(0.01)

        self.assertTrue(
            tolerance.within(
                observed=100.005,
                theoretical=100.000,
            )
        )

        self.assertTrue(
            tolerance.within(
                observed=99.991,
                theoretical=100.000,
            )
        )

    def test_within_returns_false_when_error_exceeds_da_tolerance(self) -> None:
        tolerance = DaTolerance(0.01)

        self.assertFalse(
            tolerance.within(
                observed=100.011,
                theoretical=100.000,
            )
        )

    def test_to_da_range_returns_absolute_range(self) -> None:
        tolerance = DaTolerance(0.01)

        lower, upper = tolerance.to_da_range(100.0)

        self.assertAlmostEqual(lower, 99.99)
        self.assertAlmostEqual(upper, 100.01)


class TestPpmTolerance(unittest.TestCase):
    def test_init_converts_tolerance_to_float(self) -> None:
        tolerance = PpmTolerance(10)

        self.assertEqual(tolerance.tolerance, 10.0)
        self.assertEqual(tolerance.unit, "ppm")

    def test_init_rejects_zero_or_negative_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            PpmTolerance(0)

        with self.assertRaises(ValueError):
            PpmTolerance(-10)

    def test_error_returns_signed_ppm_error(self) -> None:
        tolerance = PpmTolerance(10)

        self.assertAlmostEqual(
            tolerance.error(
                observed=100.001,
                theoretical=100.000,
            ),
            10.0,
        )

        self.assertAlmostEqual(
            tolerance.error(
                observed=99.999,
                theoretical=100.000,
            ),
            -10.0,
        )

    def test_error_rejects_non_positive_theoretical_mass(self) -> None:
        tolerance = PpmTolerance(10)

        with self.assertRaises(ValueError):
            tolerance.error(
                observed=100.0,
                theoretical=0.0,
            )

        with self.assertRaises(ValueError):
            tolerance.error(
                observed=100.0,
                theoretical=-1.0,
            )

    def test_within_returns_true_when_error_is_within_ppm_tolerance(self) -> None:
        tolerance = PpmTolerance(10)

        self.assertTrue(
            tolerance.within(
                observed=100.0009,
                theoretical=100.000,
            )
        )

    def test_within_returns_false_when_error_exceeds_ppm_tolerance(self) -> None:
        tolerance = PpmTolerance(10)

        self.assertFalse(
            tolerance.within(
                observed=100.002,
                theoretical=100.000,
            )
        )

    def test_to_da_range_returns_ppm_range_converted_to_da(self) -> None:
        tolerance = PpmTolerance(10)

        lower, upper = tolerance.to_da_range(100.0)

        self.assertAlmostEqual(lower, 99.999)
        self.assertAlmostEqual(upper, 100.001)

    def test_to_da_range_rejects_non_positive_theoretical_mass(self) -> None:
        tolerance = PpmTolerance(10)

        with self.assertRaises(ValueError):
            tolerance.to_da_range(0.0)

        with self.assertRaises(ValueError):
            tolerance.to_da_range(-1.0)


class TestDaOrPpmTolerance(unittest.TestCase):
    def test_init_uses_da_as_representative_tolerance(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.01,
            ppm_tolerance=10,
        )

        self.assertEqual(tolerance.tolerance, 0.01)
        self.assertEqual(tolerance.da_tolerance, 0.01)
        self.assertEqual(tolerance.ppm_tolerance, 10.0)
        self.assertEqual(tolerance.unit, "Da")

    def test_init_rejects_invalid_ppm_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            DaOrPpmTolerance(
                tolerance=0.01,
                ppm_tolerance=0,
            )

        with self.assertRaises(ValueError):
            DaOrPpmTolerance(
                tolerance=0.01,
                ppm_tolerance=-10,
            )

        with self.assertRaises(TypeError):
            DaOrPpmTolerance(
                tolerance=0.01,
                ppm_tolerance="10",  # type: ignore[arg-type]
            )

    def test_error_returns_da_error(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.01,
            ppm_tolerance=10,
        )

        self.assertAlmostEqual(
            tolerance.error(
                observed=100.005,
                theoretical=100.000,
            ),
            0.005,
        )

    def test_within_returns_true_when_da_condition_is_satisfied(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.01,
            ppm_tolerance=1,
        )

        self.assertTrue(
            tolerance.within(
                observed=100.005,
                theoretical=100.000,
            )
        )

    def test_within_returns_true_when_ppm_condition_is_satisfied(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.0001,
            ppm_tolerance=10,
        )

        self.assertTrue(
            tolerance.within(
                observed=100.0009,
                theoretical=100.000,
            )
        )

    def test_within_returns_false_when_both_conditions_are_not_satisfied(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.0001,
            ppm_tolerance=1,
        )

        self.assertFalse(
            tolerance.within(
                observed=100.001,
                theoretical=100.000,
            )
        )

    def test_to_da_range_returns_union_of_da_and_ppm_ranges(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.01,
            ppm_tolerance=10,
        )

        lower, upper = tolerance.to_da_range(100.0)

        self.assertAlmostEqual(lower, 99.99)
        self.assertAlmostEqual(upper, 100.01)

    def test_to_da_range_returns_wider_ppm_range_when_ppm_is_wider(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.0001,
            ppm_tolerance=10,
        )

        lower, upper = tolerance.to_da_range(100.0)

        self.assertAlmostEqual(lower, 99.999)
        self.assertAlmostEqual(upper, 100.001)


class TestParseMassTolerance(unittest.TestCase):
    def test_parse_da_tolerance(self) -> None:
        tolerance = parse_mass_tolerance("0.01Da")

        self.assertIsInstance(tolerance, DaTolerance)
        self.assertEqual(tolerance.tolerance, 0.01)
        self.assertEqual(tolerance.unit, "Da")

    def test_parse_ppm_tolerance(self) -> None:
        tolerance = parse_mass_tolerance("10ppm")

        self.assertIsInstance(tolerance, PpmTolerance)
        self.assertEqual(tolerance.tolerance, 10.0)
        self.assertEqual(tolerance.unit, "ppm")

    def test_parse_da_or_ppm_tolerance(self) -> None:
        tolerance = parse_mass_tolerance("0.01Da,10ppm")

        self.assertIsInstance(tolerance, DaOrPpmTolerance)
        self.assertEqual(tolerance.tolerance, 0.01)
        self.assertEqual(tolerance.da_tolerance, 0.01)
        self.assertEqual(tolerance.ppm_tolerance, 10.0)
        self.assertEqual(tolerance.unit, "Da")

    def test_parse_allows_spaces(self) -> None:
        tolerance = parse_mass_tolerance(" 0.01 Da , 10 ppm ")

        self.assertIsInstance(tolerance, DaOrPpmTolerance)
        self.assertEqual(tolerance.tolerance, 0.01)
        self.assertEqual(tolerance.ppm_tolerance, 10.0)

    def test_parse_is_case_insensitive(self) -> None:
        da_tolerance = parse_mass_tolerance("0.01DA")
        ppm_tolerance = parse_mass_tolerance("10PPM")
        da_or_ppm_tolerance = parse_mass_tolerance("0.01DA,10PPM")

        self.assertIsInstance(da_tolerance, DaTolerance)
        self.assertIsInstance(ppm_tolerance, PpmTolerance)
        self.assertIsInstance(da_or_ppm_tolerance, DaOrPpmTolerance)

    def test_parse_supports_scientific_notation(self) -> None:
        tolerance = parse_mass_tolerance("1e-2Da,1e1ppm")

        self.assertIsInstance(tolerance, DaOrPpmTolerance)
        self.assertEqual(tolerance.tolerance, 0.01)
        self.assertEqual(tolerance.ppm_tolerance, 10.0)

    def test_parse_rejects_invalid_format(self) -> None:
        invalid_values = [
            "",
            "0.01",
            "Da",
            "ppm",
            "0.01Da,",
            "10ppm,0.01Da",
            "0.01Da,10",
            "any:0.01Da,10ppm",
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_mass_tolerance(value)

    def test_parse_rejects_non_string_value(self) -> None:
        with self.assertRaises(TypeError):
            parse_mass_tolerance(0.01)  # type: ignore[arg-type]


class TestFormatMassTolerance(unittest.TestCase):
    def test_format_da_tolerance(self) -> None:
        tolerance = DaTolerance(0.01)

        self.assertEqual(
            format_mass_tolerance(tolerance),
            "0.01Da",
        )

    def test_format_ppm_tolerance(self) -> None:
        tolerance = PpmTolerance(10)

        self.assertEqual(
            format_mass_tolerance(tolerance),
            "10ppm",
        )

    def test_format_da_or_ppm_tolerance(self) -> None:
        tolerance = DaOrPpmTolerance(
            tolerance=0.01,
            ppm_tolerance=10,
        )

        self.assertEqual(
            format_mass_tolerance(tolerance),
            "0.01Da,10ppm",
        )

    def test_format_rejects_unknown_tolerance_type(self) -> None:
        class UnknownTolerance(MassTolerance):
            @property
            def unit(self) -> str:
                return "unknown"

            def error(self, observed: float, theoretical: float) -> float:
                return 0.0

            def within(self, observed: float, theoretical: float) -> bool:
                return False

            def to_da_range(self, theoretical: float) -> tuple[float, float]:
                return theoretical, theoretical

        tolerance = UnknownTolerance(1)

        with self.assertRaises(TypeError):
            format_mass_tolerance(tolerance)


if __name__ == "__main__":
    unittest.main()