"""Settings that control spectrum generation, independent of model architecture."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class GenerationSettings:
    """Search and output settings applied whenever spectra are generated.

    branch_path_threshold:
        Minimum path probability for keeping a reaction branch (0 keeps all).
    max_fragment_nodes:
        Fragment-node budget shared by one (compound, main adduct) branch group.
    ion_prediction_threshold:
        Minimum physical-ion probability for an output peak.
    peak_intensity_threshold:
        Minimum max-normalized intensity for an output peak.
    """
    branch_path_threshold: float = 0.0
    max_fragment_nodes: int = 100
    ion_prediction_threshold: float = 0.5
    peak_intensity_threshold: float = 0.0

    def __post_init__(self) -> None:
        for name in ("branch_path_threshold", "ion_prediction_threshold", "peak_intensity_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number in [0, 1]")
            object.__setattr__(self, name, float(value))
        if type(self.max_fragment_nodes) is not int or self.max_fragment_nodes < 1:
            raise ValueError("max_fragment_nodes must be a positive integer")

    @classmethod
    def from_dict(cls, data: dict | None) -> GenerationSettings:
        data = dict(data or {})
        unknown = sorted(set(data) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown generation parameters: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
