from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import torch
from torch import Tensor


@dataclass(frozen=True)
class ReactionPreference:
    """Optional soft teacher weight attached to one teacher path."""

    path_index: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.path_index < 0:raise ValueError("path_index must be non-negative")
        if not self.weight > 0:raise ValueError("reaction preference weight must be positive")


def reaction_preference_weights(
    path_indices: Tensor,
    preferences: Mapping[int,float] | None,
    *,
    enabled: bool = False,
) -> Tensor:
    """Return neutral weights unless preference is explicitly enabled."""
    result=torch.ones(path_indices.shape,device=path_indices.device,dtype=torch.float32)
    if not enabled or not preferences:return result
    for path,weight in preferences.items():
        if weight<=0:raise ValueError("reaction preference weights must be positive")
        result[path_indices==int(path)]=float(weight)
    return result


def combine_teacher_and_preference_weights(
    teacher_weight: Tensor,
    preference_weight: Tensor | None,
    *,
    enabled: bool = False,
) -> Tensor:
    if not enabled or preference_weight is None:return teacher_weight
    if preference_weight.shape!=teacher_weight.shape:raise ValueError("preference weights must align with teacher weights")
    return teacher_weight*preference_weight.to(teacher_weight)
