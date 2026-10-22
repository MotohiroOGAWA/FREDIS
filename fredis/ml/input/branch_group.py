"""Unique SourceCompound / main-adduct contexts, independent of collision energy."""

from dataclasses import dataclass
import torch
from torch import Tensor
from .reaction_context_data import TensorTable


@dataclass(frozen=True)
class BranchGroup(TensorTable):
    source_compound_index: Tensor
    main_adduct_index: Tensor

    def __len__(self):
        return self.source_compound_index.numel()

    def validate(self, source_count, adducts):
        for values, size in (
            (self.source_compound_index, source_count),
            (self.main_adduct_index, len(adducts.main_adducts)),
        ):
            if (
                values.dtype != torch.long
                or values.shape != (len(self),)
                or (
                    values.numel()
                    and (int(values.min()) < 0 or int(values.max()) >= size)
                )
            ):
                raise ValueError("Invalid BranchGroup foreign key")
        if torch.unique(
            torch.stack((self.source_compound_index, self.main_adduct_index), 1), dim=0
        ).shape[0] != len(self):
            raise ValueError(
                "BranchGroup SourceCompound/main-adduct pairs must be unique"
            )
