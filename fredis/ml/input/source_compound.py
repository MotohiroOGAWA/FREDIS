"""Independent packed SourceCompound graphs and their structural identity."""

from dataclasses import dataclass
from torch_geometric.data import Batch


@dataclass(frozen=True)
class SourceCompound:
    source_graph: Batch
    source_smiles: tuple[str, ...]
    # Shared atom-feature vocabulary, including the explicit-H policy.
    symbols: tuple[str, ...] = ()

    def to(self, device):
        return SourceCompound(
            self.source_graph.to(device), self.source_smiles, self.symbols
        )

    @classmethod
    def collate(cls, items):
        if not items:
            raise ValueError("Cannot collate empty SourceCompound batches")
        if any(item.symbols != items[0].symbols for item in items):
            raise ValueError("SourceCompound atom feature vocabularies must match")
        return cls(
            Batch.from_data_list(
                [g for item in items for g in item.source_graph.to_data_list()]
            ),
            tuple(s for item in items for s in item.source_smiles),
            items[0].symbols,
        )
