"""Independent, canonical CSR sets of batch-global concrete reaction rows."""

from dataclasses import dataclass
from collections.abc import Sequence
import torch
from torch import Tensor


@dataclass(frozen=True)
class ReactionSet:
    # [N_set + 1], [N_members]: one strictly ascending list per unique set.
    # No group, Compound, topology, or teacher ownership belongs in this table.
    reaction_ptr: Tensor
    reaction_index: Tensor

    def __len__(self):
        return self.reaction_ptr.numel() - 1

    def to(self, device):
        return ReactionSet(self.reaction_ptr.to(device), self.reaction_index.to(device))

    @classmethod
    def from_rows(cls, rows, *, device=None):
        """Intern reaction sets; return the table and input-row -> set-ID mapping."""
        lookup, unique, inverse = {}, [], []
        for row in rows:
            key = tuple(sorted(set(int(r) for r in row)))
            if any(r < 0 for r in key):
                raise ValueError("Reaction sets require nonnegative registry row IDs")
            if key not in lookup:
                lookup[key] = len(unique)
                unique.append(key)
            inverse.append(lookup[key])
        counts = torch.tensor(
            [len(row) for row in unique], dtype=torch.long, device=device
        )
        ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        values = torch.tensor(
            [r for row in unique for r in row], dtype=torch.long, device=device
        )
        return cls(ptr, values), torch.tensor(inverse, dtype=torch.long, device=device)

    def rows(self):
        return [
            self.reaction_index[a:b].tolist()
            for a, b in zip(self.reaction_ptr[:-1], self.reaction_ptr[1:])
        ]

    def gather(self, set_index: Tensor) -> tuple[Tensor, Tensor]:
        """Build a runtime CSR projection without storing a second membership table."""
        counts = self.reaction_ptr[set_index + 1] - self.reaction_ptr[set_index]
        ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        owner = torch.repeat_interleave(
            torch.arange(set_index.numel(), device=set_index.device), counts
        )
        offsets = torch.arange(int(ptr[-1]), device=ptr.device) - ptr[owner]
        return ptr, self.reaction_index[self.reaction_ptr[set_index[owner]] + offsets]

    @classmethod
    def collate(cls, tables: Sequence["ReactionSet"], reaction_offsets):
        """Remap registry IDs before interning; the empty set is shared globally."""
        if not tables:
            raise ValueError("Cannot collate empty ReactionSet tables")
        rows, sizes = [], []
        for table, offset in zip(tables, reaction_offsets, strict=True):
            rows.extend([[r + offset for r in row] for row in table.rows()])
            sizes.append(len(table))
        result, inverse = cls.from_rows(rows, device=tables[0].reaction_ptr.device)
        return result, inverse.split(sizes)

    def validate(self, reaction_source_compound_index: Tensor):
        ptr, values = self.reaction_ptr, self.reaction_index
        if (
            ptr.dtype != torch.long
            or values.dtype != torch.long
            or ptr.ndim != 1
            or values.ndim != 1
            or ptr.numel() == 0
            or int(ptr[0]) != 0
            or int(ptr[-1]) != values.numel()
            or bool((ptr[1:] < ptr[:-1]).any())
        ):
            raise ValueError("Invalid ReactionSet CSR")
        if values.numel() and (
            int(values.min()) < 0
            or int(values.max()) >= reaction_source_compound_index.numel()
        ):
            raise ValueError("ReactionSet references a missing concrete reaction")
        known = set()
        for row in self.rows():
            key = tuple(row)
            if key != tuple(sorted(set(row))) or key in known:
                raise ValueError("ReactionSet rows must be unique sorted sets")
            known.add(key)
            if (
                row
                and reaction_source_compound_index[values.new_tensor(row)]
                .unique()
                .numel()
                != 1
            ):
                raise ValueError("A reaction set cannot mix different SourceCompounds")
