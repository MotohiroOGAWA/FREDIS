"""Unique molecular structures and the independent sets that generate them."""

from dataclasses import dataclass, replace
from torch import Tensor
from torch_geometric.data import Batch
from fredis.libs.mmkit.mmkit import Compound


@dataclass(frozen=True)
class CompoundBatch:
    compounds: tuple[Compound, ...]
    graph: Batch
    # One CSR row per Compound, pointing to unique independent ReactionSet IDs.
    generating_reaction_sets_ptr: Tensor
    generating_reaction_sets_index: Tensor

    @property
    def smiles(self) -> tuple[str, ...]:
        """Derived view; structural identity is not stored twice."""
        return tuple(compound.smiles for compound in self.compounds)

    def to(self, device):
        return replace(
            self,
            graph=self.graph.clone().to(device),
            generating_reaction_sets_ptr=self.generating_reaction_sets_ptr.to(device),
            generating_reaction_sets_index=self.generating_reaction_sets_index.to(
                device
            ),
        )
