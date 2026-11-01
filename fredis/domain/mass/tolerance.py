from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import msentity_similarity as mss


@dataclass(frozen=True, slots=True)
class MassTolerance(ABC):
    """Base class for mass tolerance.

    Parameters
    ----------
    tolerance:
        Positive tolerance value.
        The unit is defined by each subclass.
    """

    tolerance: float

    def __post_init__(self) -> None:
        tolerance = self._validate_number(
            self.tolerance,
            name="tolerance",
            allow_zero=False,
        )

        object.__setattr__(self, "tolerance", tolerance)

    @property
    @abstractmethod
    def unit(self) -> str:
        """Representative unit name."""
        raise NotImplementedError

    @abstractmethod
    def error(self, observed: float, theoretical: float) -> float:
        """Compute signed mass error."""
        raise NotImplementedError

    @abstractmethod
    def within(self, observed: float, theoretical: float) -> bool:
        """Return True if observed is within tolerance."""
        raise NotImplementedError

    @abstractmethod
    def to_da_range(self, theoretical: float) -> tuple[float, float]:
        """Return lower and upper bounds in Da."""
        raise NotImplementedError

    @staticmethod
    def _validate_number(
        value: float,
        *,
        name: str,
        allow_zero: bool = True,
    ) -> float:
        """Validate and convert a numeric value to float."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")

        value = float(value)

        if allow_zero:
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
        else:
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")

        return value

    def _new_with_tolerance(self, tolerance: float) -> MassTolerance:
        """Create a new instance of the same tolerance class."""
        return type(self)(tolerance)

    # def __add__(self, value: float) -> MassTolerance:
    #     """Return a new instance with increased tolerance."""
    #     value = self._validate_number(value, name="value")
    #     return self._new_with_tolerance(self.tolerance + value)

    # def __sub__(self, value: float) -> MassTolerance:
    #     """Return a new instance with decreased tolerance."""
    #     value = self._validate_number(value, name="value")
    #     return self._new_with_tolerance(self.tolerance - value)

    # def __mul__(self, value: float) -> MassTolerance:
    #     """Return a new instance with multiplied tolerance."""
    #     value = self._validate_number(value, name="value")
    #     return self._new_with_tolerance(self.tolerance * value)

    # def __truediv__(self, value: float) -> MassTolerance:
    #     """Return a new instance with divided tolerance."""
    #     value = self._validate_number(value, name="value")

    #     if value == 0:
    #         raise ZeroDivisionError("Cannot divide tolerance by zero.")

    #     return self._new_with_tolerance(self.tolerance / value)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tolerance={self.tolerance}, "
            f"unit='{self.unit}'"
            f")"
        )


@dataclass(frozen=True, slots=True)
class DaTolerance(MassTolerance):
    """Absolute mass tolerance in Daltons."""

    UNIT: ClassVar[str] = "Da"

    @property
    def unit(self) -> str:
        return self.UNIT

    def error(self, observed: float, theoretical: float) -> float:
        observed = self._validate_number(observed, name="observed")
        theoretical = self._validate_number(theoretical, name="theoretical")

        return observed - theoretical

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance

    def to_da_range(self, theoretical: float) -> tuple[float, float]:
        theoretical = self._validate_number(theoretical, name="theoretical")

        return (
            theoretical - self.tolerance,
            theoretical + self.tolerance,
        )


@dataclass(frozen=True, slots=True)
class PpmTolerance(MassTolerance):
    """Relative mass tolerance in parts-per-million."""

    UNIT: ClassVar[str] = "ppm"

    @property
    def unit(self) -> str:
        return self.UNIT

    def error(self, observed: float, theoretical: float) -> float:
        observed = self._validate_number(observed, name="observed")
        theoretical = self._validate_number(
            theoretical,
            name="theoretical",
            allow_zero=False,
        )

        return (observed - theoretical) / theoretical * 1e6

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance

    def to_da_range(self, theoretical: float) -> tuple[float, float]:
        theoretical = self._validate_number(
            theoretical,
            name="theoretical",
            allow_zero=False,
        )

        delta = theoretical * self.tolerance / 1e6

        return (
            theoretical - delta,
            theoretical + delta,
        )


@dataclass(frozen=True, slots=True)
class DaOrPpmTolerance(MassTolerance):
    """
    Mass tolerance with OR condition.

    This class uses Da as the representative unit.

    - tolerance:
        Absolute tolerance in Da.
    - ppm_tolerance:
        Additional relative tolerance in ppm.

    error() returns Da error.
    unit returns "Da".
    to_da_range() returns the union range of Da and ppm tolerance in Da.
    within() returns True if either Da or ppm tolerance is satisfied.
    """

    ppm_tolerance: float

    _da: DaTolerance = field(init=False, repr=False, compare=False)
    _ppm: PpmTolerance = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        MassTolerance.__post_init__(self)

        ppm_tolerance = self._validate_number(
            self.ppm_tolerance,
            name="ppm_tolerance",
            allow_zero=False,
        )

        object.__setattr__(self, "ppm_tolerance", ppm_tolerance)

        object.__setattr__(self, "_da", DaTolerance(self.tolerance))
        object.__setattr__(self, "_ppm", PpmTolerance(self.ppm_tolerance))

    @property
    def da_tolerance(self) -> float:
        """Absolute tolerance in Da."""
        return self.tolerance

    @property
    def unit(self) -> str:
        return DaTolerance.UNIT

    def error(self, observed: float, theoretical: float) -> float:
        return self._da.error(observed, theoretical)

    def within(self, observed: float, theoretical: float) -> bool:
        return self._da.within(observed, theoretical) or self._ppm.within(
            observed, theoretical
        )

    def to_da_range(self, theoretical: float) -> tuple[float, float]:
        da_lower, da_upper = self._da.to_da_range(theoretical)
        ppm_lower, ppm_upper = self._ppm.to_da_range(theoretical)

        return (
            min(da_lower, ppm_lower),
            max(da_upper, ppm_upper),
        )

    # def __add__(self, value: float) -> DaOrPpmTolerance:
    #     value = self._validate_number(value, name="value")

    #     return type(self)(
    #         tolerance=self.tolerance + value,
    #         ppm_tolerance=self.ppm_tolerance + value,
    #     )

    # def __sub__(self, value: float) -> DaOrPpmTolerance:
    #     value = self._validate_number(value, name="value")

    #     return type(self)(
    #         tolerance=self.tolerance - value,
    #         ppm_tolerance=self.ppm_tolerance - value,
    #     )

    # def __mul__(self, value: float) -> DaOrPpmTolerance:
    #     value = self._validate_number(value, name="value")

    #     return type(self)(
    #         tolerance=self.tolerance * value,
    #         ppm_tolerance=self.ppm_tolerance * value,
    #     )

    # def __truediv__(self, value: float) -> DaOrPpmTolerance:
    #     value = self._validate_number(value, name="value")

    #     if value == 0:
    #         raise ZeroDivisionError("Cannot divide tolerance by zero.")

    #     return type(self)(
    #         tolerance=self.tolerance / value,
    #         ppm_tolerance=self.ppm_tolerance / value,
    #     )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"da_tolerance={self.tolerance}, "
            f"ppm_tolerance={self.ppm_tolerance}"
            f")"
        )


def parse_mass_tolerance(value: str) -> MassTolerance:
    """Parse a string into a MassTolerance with msentity-similarity's parser.

    Supported formats
    -----------------
    0.01Da
    10ppm
    0.01Da,10ppm (either order; peaks match within either term)
    """
    if not isinstance(value, str):
        raise TypeError("value must be a string.")

    parsed = mss.parse_mz_tolerance(value)
    da_tolerance = parsed["mz_tolerance_da"]
    ppm_tolerance = parsed["mz_tolerance_ppm"]

    if da_tolerance > 0 and ppm_tolerance > 0:
        return DaOrPpmTolerance(
            tolerance=da_tolerance,
            ppm_tolerance=ppm_tolerance,
        )

    if da_tolerance > 0:
        return DaTolerance(tolerance=da_tolerance)

    return PpmTolerance(tolerance=ppm_tolerance)


def format_mass_tolerance(tolerance: MassTolerance) -> str:
    """Convert a MassTolerance object to msentity-similarity's canonical text."""
    if isinstance(tolerance, DaOrPpmTolerance):
        return mss.format_mz_tolerance(tolerance.da_tolerance, tolerance.ppm_tolerance)

    if isinstance(tolerance, DaTolerance):
        return mss.format_mz_tolerance(tolerance.tolerance, 0.0)

    if isinstance(tolerance, PpmTolerance):
        return mss.format_mz_tolerance(0.0, tolerance.tolerance)

    raise TypeError(f"Unsupported tolerance type: {type(tolerance).__name__}")
