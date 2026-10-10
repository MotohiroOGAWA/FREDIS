from typing import Tuple
import torch
import torch.nn as nn
from rdkit import Chem


class BondFeatureLayer(nn.Module):
    """
    Encodes RDKit Bond objects into concatenated one-hot feature vectors.

    Features:
      - bond_type:  ["AROMATIC", "SINGLE", "DOUBLE", "TRIPLE"]
      - ring_type:  ["no-ring", "small-ring", "5-cycle", "6-cycle", "large-ring"]
    """

    def __init__(self):
        super(BondFeatureLayer, self).__init__()

        self.feature_sets = {
            "bond_type": ("AROMATIC", "SINGLE", "DOUBLE", "TRIPLE"),
            "ring_type": ("no-ring", "small-ring", "5-cycle", "6-cycle", "large-ring"),
        }

        self.base_bond_type_vector = nn.Parameter(
            torch.zeros(len(self.feature_sets["bond_type"]), dtype=torch.float32),
            requires_grad=False
        )
        self.base_ring_type_vector = nn.Parameter(
            torch.zeros(len(self.feature_sets["ring_type"]), dtype=torch.float32),
            requires_grad=False
        )

    @property
    def feature_dim(self) -> int:
        """
        Total length (dimension) of the concatenated bond feature vector.
        """
        return sum(len(v) for v in self.feature_sets.values())

    def encode(self, bond: Chem.rdchem.Bond) -> torch.Tensor:
        """Encode an RDKit Bond object into a one-hot feature vector."""
        bond_type_tensor = self.encode_bond_type(bond)
        ring_type_tensor = self.encode_ring_type(bond)
        return torch.cat([bond_type_tensor, ring_type_tensor], dim=0)

    def encode_bond_type(self, bond: Chem.rdchem.Bond) -> torch.Tensor:
        one_hot = self.base_bond_type_vector.clone()
        bond_types = self.feature_sets["bond_type"]

        if bond is None:
            # No bond (e.g., implicit hydrogen) → leave as zero vector
            return one_hot

        if bond.GetIsAromatic():
            one_hot[bond_types.index("AROMATIC")] = 1.0
        else:
            btype = bond.GetBondType().name.upper()  # e.g., 'SINGLE', 'DOUBLE', 'TRIPLE'
            if btype in bond_types:
                one_hot[bond_types.index(btype)] = 1.0
        return one_hot

    def encode_ring_type(self, bond: Chem.rdchem.Bond) -> torch.Tensor:
        one_hot = self.base_ring_type_vector.clone()
        ring_types = self.feature_sets["ring_type"]

        if bond is None or not bond.IsInRing():
            one_hot[ring_types.index("no-ring")] = 1.0
            return one_hot

        # Check ring size (use bond.GetOwningMol())
        mol = bond.GetOwningMol()
        bond_idx = bond.GetIdx()
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.BondRings() if bond_idx in r]

        if not ring_sizes:
            one_hot[ring_types.index("no-ring")] = 1.0
        else:
            size = min(ring_sizes)
            if size <= 4:
                one_hot[ring_types.index("small-ring")] = 1.0
            elif size == 5:
                one_hot[ring_types.index("5-cycle")] = 1.0
            elif size == 6:
                one_hot[ring_types.index("6-cycle")] = 1.0
            else:
                one_hot[ring_types.index("large-ring")] = 1.0

        return one_hot