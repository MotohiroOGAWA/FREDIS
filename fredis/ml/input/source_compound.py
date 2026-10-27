"""Independent packed SourceCompound graphs and their structural identity."""

from dataclasses import dataclass, field
import torch
from torch import Tensor
from torch_geometric.data import Batch


@dataclass(frozen=True)
class SourceCompound:
    source_graph: Batch
    source_smiles: tuple[str, ...]
    # Shared atom-feature vocabulary, including the explicit-H policy.
    symbols: tuple[str, ...] = ()
    compound_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )

    def to(self, device):
        return SourceCompound(
            self.source_graph.clone().to(device),
            self.source_smiles,
            self.symbols,
            self.compound_index.to(device),
        )

    @classmethod
    def collate(cls, items, *, compound_offsets=None):
        if not items:
            raise ValueError("Cannot collate empty SourceCompound batches")
        if any(item.symbols != items[0].symbols for item in items):
            raise ValueError("SourceCompound atom feature vocabularies must match")
        mappings = [item.compound_index for item in items]
        if any(value.numel() for value in mappings) and compound_offsets is None:
            raise ValueError(
                "Source mappings require explicit CompoundBatch offsets; use SpectrumPredictionData.from_structures for shared registries"
            )
        if compound_offsets is not None and len(compound_offsets) != len(items):
            raise ValueError("One CompoundBatch offset is required per input")
        mapping = torch.cat(
            [
                value + offset
                for value, offset in zip(mappings, compound_offsets or [0] * len(items))
            ]
        )
        return cls(
            Batch.from_data_list(
                [g for item in items for g in item.source_graph.to_data_list()]
            ),
            tuple(s for item in items for s in item.source_smiles),
            items[0].symbols,
            mapping,
        )
