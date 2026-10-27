"""Shared, ordered chemistry vocabularies and main-to-fragment adduct membership."""

from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor


@dataclass(frozen=True)
class AdductData:
    main_adducts: tuple[str, ...]
    fragment_ion_adducts: tuple[str, ...]
    fragment_ion_adduct_ptr: Tensor
    fragment_ion_adduct_index: Tensor
    hydrogen_shifts: Tensor

    @classmethod
    def from_fragmenter(cls, fragmenter):
        rules = fragmenter.adduct_rule_set.adduct_rules
        mains = tuple(str(rule.adduct_type) for rule in rules)
        fragments = tuple(
            dict.fromkeys(str(a) for rule in rules for a in rule.fragment_ion_adducts)
        )
        rows = [
            [fragments.index(str(a)) for a in rule.fragment_ion_adducts]
            for rule in rules
        ]
        lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
        result = cls(
            mains,
            fragments,
            torch.cat((lengths.new_zeros(1), lengths.cumsum(0))),
            torch.tensor([v for row in rows for v in row], dtype=torch.long),
            torch.tensor(fragmenter.adduct_rule_set.hydrogen_shifts, dtype=torch.long),
        )
        result.validate()
        return result

    def to(self, device):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )

    def validate(self):
        for labels in (self.main_adducts, self.fragment_ion_adducts):
            if len(set(labels)) != len(labels):
                raise ValueError("Adduct labels must be unique")
        ptr, index = self.fragment_ion_adduct_ptr, self.fragment_ion_adduct_index
        if (
            ptr.dtype != torch.long
            or ptr.shape != (len(self.main_adducts) + 1,)
            or int(ptr[0]) != 0
            or int(ptr[-1]) != index.numel()
            or bool((ptr[1:] < ptr[:-1]).any())
        ):
            raise ValueError("Invalid main-to-fragment adduct CSR")
        if (
            index.dtype != torch.long
            or index.ndim != 1
            or (
                index.numel()
                and (
                    int(index.min()) < 0
                    or int(index.max()) >= len(self.fragment_ion_adducts)
                )
            )
        ):
            raise ValueError("Invalid fragment ion adduct index")
        shifts = self.hydrogen_shifts
        if (
            shifts.dtype != torch.long
            or shifts.ndim != 1
            or shifts.unique().numel() != shifts.numel()
        ):
            raise ValueError("Hydrogen shifts must be unique int64 values")
        for a, b in zip(ptr[:-1], ptr[1:]):
            if index[a:b].unique().numel() != int(b - a):
                raise ValueError("Duplicate fragment ion adduct membership")

    def same_vocabulary(self, other):
        return (
            self.main_adducts == other.main_adducts
            and self.fragment_ion_adducts == other.fragment_ion_adducts
            and all(
                torch.equal(getattr(self, key).cpu(), getattr(other, key).cpu())
                for key in (
                    "fragment_ion_adduct_ptr",
                    "fragment_ion_adduct_index",
                    "hydrogen_shifts",
                )
            )
        )
