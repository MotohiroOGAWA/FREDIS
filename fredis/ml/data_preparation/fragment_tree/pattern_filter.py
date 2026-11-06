"""Filter Sources before preparing tensors, using reaction feature ownership."""

from rdkit import Chem
from tqdm.auto import tqdm
from fredis.domain.fragment.reaction import ReactionPatternSet


def select_trainable_pattern_records(smiles_values, pattern_set, reaction_models=()):
    """Keep rows matching any dataset-owned (not imported/frozen) reactant.

    Definitions, rather than source checkpoint IDs, identify frozen patterns:
    imported models can use different IDs from the merged dataset pattern set.
    """
    frozen = {
        ReactionPatternSet.from_dict({'name': 'imported', 'patterns': [record['pattern']]}).patterns[0].key
        for record in reaction_models if record.get('source')
    }
    patterns = [pattern for pattern in pattern_set if pattern.key not in frozen]
    if not patterns:
        raise ValueError('No trainable reaction patterns remain; add a new reaction pattern before preparing fine-tuning data.')
    smiles_values = list(map(str, smiles_values))
    matched = {}
    for smiles in tqdm(dict.fromkeys(smiles_values), desc='Selecting trainable-pattern compounds', unit='compound'):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f'Invalid Source SMILES during pattern selection: {smiles}')
        matched[smiles] = any(molecule.HasSubstructMatch(pattern.reactant_query) for pattern in patterns)
    indices = [index for index, smiles in enumerate(smiles_values) if matched[smiles]]
    return indices, {
        'rule': 'source_matches_any_non_frozen_reactant',
        'trainable_pattern_ids': [pattern.pattern_id for pattern in patterns],
        'input_records': len(smiles_values),
        'retained_records': len(indices),
        'excluded_records': len(smiles_values) - len(indices),
        'input_compounds': len(matched),
        'retained_compounds': sum(matched.values()),
    }
