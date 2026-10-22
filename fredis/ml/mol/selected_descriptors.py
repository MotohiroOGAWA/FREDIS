"""Optional model features computed only after a descriptor selection exists."""

from functools import lru_cache
from rdkit import Chem
from fredis.domain.molecule.descriptors import compute_descriptor_values


@lru_cache(maxsize=8192)
def selected_descriptor_values(
    smiles: str, names: tuple[str, ...]
) -> tuple[float, ...]:
    """Cache only requested features; never attach them to prepared graphs."""
    if not names:
        return ()
    return compute_descriptor_values(Chem.MolFromSmiles(smiles), names)
