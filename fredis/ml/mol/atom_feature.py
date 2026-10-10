from typing import Tuple
import torch
import torch.nn as nn
from rdkit import Chem

# Disable RDKit logging
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Only show critical errors, suppress warnings and other messages


class AtomFeatureLayer(nn.Module):
    def __init__(
            self, 
            symbols:Tuple[str],
            ):
        super(AtomFeatureLayer, self).__init__()
        self.symbols = tuple(sorted(set(symbols)))

        self.feature_sets = {
            'symbol': tuple(self.symbols),
            'charge': (-1, 0, 1),
            'ring_type': ('no-ring', 'small-ring', '5-cycle', '6-cycle', 'large-ring'),
            'hybridization': ('sp', 'sp2', 'sp3', 'sp3d', 'sp3d2'),
            'num_hydrogens': (0, 1, 2, 3, 4),
            'valence_electrons': (1, 2, 3, 4, 5, 6, 7, 8),
            # 'oxidation_number': (1, 2, 3, 4, 5, 6, 7, 8, 9),
        }
        
        self.base_symbol_vector = nn.Parameter(
            torch.zeros(len(self.symbols), dtype=torch.float32),
            requires_grad=False
        )
        self.base_charge_vector = nn.Parameter(
            torch.zeros(len(self.feature_sets['charge']), dtype=torch.float32),
            requires_grad=False
        )
        self.base_ring_type_vector = nn.Parameter(
            torch.zeros(len(self.feature_sets['ring_type']), dtype=torch.float32),
            requires_grad=False
        )
        self.base_hybridization_vector = nn.Parameter(
            torch.zeros(len(self.feature_sets['hybridization']), dtype=torch.float32),
            requires_grad=False
        )
        self.base_num_hydrogens_vector = nn.Parameter(
            torch.zeros(len(self.feature_sets['num_hydrogens']), dtype=torch.float32),
            requires_grad=False
        )
        self.base_valence_electrons_vector = nn.Parameter(
            torch.zeros(len(self.feature_sets['valence_electrons']), dtype=torch.float32),
            requires_grad=False
        )
        # self.base_oxidation_number_vector = nn.Parameter(
        #     torch.zeros(len(self.feature_sets['oxidation_number']), dtype=torch.float32),
        #     requires_grad=False
        # )

    @property
    def feature_dim(self) -> int:
        """
        Total length (dimension) of the concatenated atom feature vector.
        """
        return sum(len(v) for v in self.feature_sets.values())

    def encode(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        symbol_tensor = self.encode_symbol(atom)
        charge_tensor = self.encode_charge(atom)
        ring_type_tensor = self.encode_ring_type(atom)
        hybridization_tensor = self.encode_hybridization(atom)
        num_hydrogens_tensor = self.encode_num_hydrogens(atom)
        valence_electrons_tensor = self.encode_valence_electrons(atom)
        # oxidation_number_tensor = self.encode_oxidation_number(atom)

        return torch.cat([symbol_tensor, charge_tensor, ring_type_tensor, hybridization_tensor, num_hydrogens_tensor, valence_electrons_tensor], dim=0)


    def encode_symbol(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        one_hot = self.base_symbol_vector.clone()
        symbol = atom.GetSymbol()
        idx = self.symbols.index(symbol) if symbol in self.symbols else None

        if idx is None:
            raise ValueError(
                f"Unsupported atom symbol: '{symbol}'. "
                f"Supported symbols are: {self.symbols}"
            )
        one_hot[idx] = 1.0
        return one_hot

    def encode_charge(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        charge = atom.GetFormalCharge()
        one_hot = self.base_charge_vector.clone()
        if charge in self.feature_sets['charge']:
            one_hot[self.feature_sets['charge'].index(charge)] = 1.0
        return one_hot

    def encode_ring_type(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        one_hot = self.base_ring_type_vector.clone()
        ring_types = self.feature_sets['ring_type']
        if not atom.IsInRing():
            one_hot[ring_types.index('no-ring')] = 1.0
        elif atom.IsInRingSize(3) or atom.IsInRingSize(4):
            one_hot[ring_types.index('small-ring')] = 1.0
        elif atom.IsInRingSize(5):
            one_hot[ring_types.index('5-cycle')] = 1.0
        elif atom.IsInRingSize(6):
            one_hot[ring_types.index('6-cycle')] = 1.0
        else:
            one_hot[ring_types.index('large-ring')] = 1.0
        return one_hot
    
    def encode_hybridization(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        one_hot = self.base_hybridization_vector.clone()
        hybridizations = self.feature_sets['hybridization']
        hyb = atom.GetHybridization().name.lower()
        if hyb in hybridizations:
            one_hot[hybridizations.index(hyb)] = 1.0
        return one_hot
    
    def encode_num_hydrogens(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        num_h = atom.GetTotalNumHs(includeNeighbors=True)
        one_hot = self.base_num_hydrogens_vector.clone()
        if num_h in self.feature_sets['num_hydrogens']:
            one_hot[self.feature_sets['num_hydrogens'].index(num_h)] = 1.0
        return one_hot
    
    def encode_valence_electrons(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
        num_valence = atom.GetExplicitValence()
        one_hot = self.base_valence_electrons_vector.clone()
        if num_valence in self.feature_sets['valence_electrons']:
            one_hot[self.feature_sets['valence_electrons'].index(num_valence)] = 1.0
        return one_hot
    
    # def encode_oxidation_number(self, atom: Chem.rdchem.Atom) -> torch.Tensor:
    #     ox_num = Chem.rdMolDescriptors.CalcOxidationNumbers(atom)
    #     one_hot = self.base_oxidation_number_vector.clone()
    #     if ox_num in self.feature_sets['oxidation_number']:
    #         one_hot[self.feature_sets['oxidation_number'].index(ox_num)] = 1.0
    #     return one_hot

    