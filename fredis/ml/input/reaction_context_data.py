"""CE-independent candidate edge selection and optional branch supervision."""

from dataclasses import dataclass
from torch import Tensor
from .tensor_table import TensorTable


@dataclass(frozen=True)
class ReactionContextData(TensorTable):
    root_source_compound_index: Tensor
    edge_ptr: Tensor
    edge_index: Tensor
    # Aligned with edge_index: 1 positive, 0 weak negative, -1 ignored.
    # Prediction has no teacher labels.
    edge_label: Tensor | None = None
