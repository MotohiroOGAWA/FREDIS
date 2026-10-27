from typing import Optional, Tuple
import torch
from torch_geometric.data import Data

from ...libs.mmkit.mmkit import Compound
from .atom_feature import AtomFeatureLayer
from .bond_feature import BondFeatureLayer


class MolGraphBuilder:
    """
    Convert Compound into a PyTorch Geometric Data object
    with atom/bond features, without any neural embedding.

    Responsibility:
      Compound -> molecular graph (PyG Data)
    """

    def __init__(self, symbols):
        self.atom_layer = AtomFeatureLayer(symbols=symbols)
        self.bond_layer = BondFeatureLayer()

    @property
    def atom_dim(self) -> int:
        return self.atom_layer.feature_dim

    @property
    def bond_dim(self) -> int:
        return self.bond_layer.feature_dim

    @property
    def symbols(self) -> Tuple[str]:
        return self.atom_layer.symbols

    def graph_atoms(self, compound):
        """Atoms represented by this encoder; hydrogen counts remain features."""
        return [
            atom
            for atom in compound.mapped_mol.GetAtoms()
            if atom.GetAtomicNum() != 1 or "H" in self.symbols
        ]

    def build(
        self, compound: Compound, *, device: Optional[torch.device] = None
    ) -> Data:
        """
        Build a PyTorch Geometric Data object from a Compound.

        Returns:
            Data:
              x         [num_atoms, atom_dim]
              edge_index[2, E]
              edge_attr [E, bond_dim]
        """
        mol = compound.mol

        # Decide device explicitly (default: CPU)
        if device is None:
            device = torch.device("cpu")

        # --- atom features ---
        atoms = self.graph_atoms(compound)
        atom_indices = {atom.GetIdx(): index for index, atom in enumerate(atoms)}
        atom_features = [self.atom_layer.encode(atom) for atom in atoms]
        x = (
            torch.stack(atom_features, dim=0)
            if atom_features
            else torch.empty((0, self.atom_dim))
        ).to(
            device=device
        )  # [num_atoms, atom_dim]

        # --- bond features (bidirectional) ---
        edge_indices = []
        bond_features = []

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i not in atom_indices or j not in atom_indices:
                continue
            i, j = atom_indices[i], atom_indices[j]
            bond_vec = self.bond_layer.encode(bond).to(device=device)

            edge_indices.append((i, j))
            edge_indices.append((j, i))
            bond_features.append(bond_vec)
            bond_features.append(bond_vec.clone())

        if edge_indices:
            edge_index = torch.tensor(
                edge_indices, dtype=torch.long, device=device
            ).T  # [2, E]
            edge_attr = torch.stack(bond_features, dim=0)  # [E, bond_dim]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty(
                (0, self.bond_dim), dtype=torch.float32, device=device
            )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
