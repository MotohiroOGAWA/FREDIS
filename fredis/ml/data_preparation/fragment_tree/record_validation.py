"""Shared metadata checks for inspection and exclusion during preparation."""
import math
import sys
from rdkit import Chem
from tqdm import tqdm
from fredis.libs.mmkit.mmkit import Adduct
from fredis.domain.mass.parse_ce import parse_ce_to_ev

DEFAULT_MAPPING = {'smilesColumn': 'SMILES', 'adductTypeColumn': 'AdductType',
                   'collisionEnergyColumn': 'CollisionEnergy', 'precursorMzColumn': 'PrecursorMZ'}

def inspect_records(dataset, fragmenter, mapping=None, *, progress=None, invalid_detail_limit=None, symbols=None):
    mapping = {**DEFAULT_MAPPING, **(mapping or {})}
    missing = [name for name in mapping.values() if name not in dataset.columns]
    if missing:
        raise ValueError('Missing dataset columns: ' + ', '.join(missing))
    validation = {key: {'column': name, 'exists': True, 'valid': 0, 'total': len(dataset)}
                  for key, name in mapping.items()}
    invalid = []
    invalid_count = 0
    accepted = []
    # Reference libraries (e.g. NIST) repeat a handful of distinct adduct/CE
    # strings across hundreds of thousands of rows; caching by the raw string
    # (like smiles_cache already does) turns this loop from O(records) calls
    # into O(unique values) calls to Adduct.parse/parse_ce_to_ev, which is
    # where nearly all the wall-clock time otherwise goes.
    # A SMILES the molecule encoder cannot featurize (an element outside
    # `symbols`, e.g. As in NIST) must be excluded here: otherwise it raises
    # deep inside a preparation worker and aborts the whole run.
    allowed_symbols = set(symbols) | {'H'} if symbols is not None else None
    smiles_cache = {}
    adduct_cache = {}
    ce_cache = {}
    reasons = {'smilesColumn': 'Invalid RDKit SMILES',
               'adductTypeColumn': 'Invalid adduct or main adduct not registered in the fragmenter',
               'collisionEnergyColumn': 'Cannot parse collision energy to finite non-negative eV',
               'precursorMzColumn': 'Cannot parse precursor m/z to a finite positive number'}
    values = {key: dataset[name].tolist() for key, name in mapping.items()}
    ids = dataset['SpecID'].astype(str).tolist() if 'SpecID' in dataset.columns else None
    total = len(dataset)
    progress_step = max(1, total // 100)
    if progress:
        progress(0, total)
    for index in tqdm(range(total), total=total, desc='Inspecting records', file=sys.stderr):
        failures = []
        for key, item in validation.items():
            value = values[key][index]
            valid = False
            try:
                if key == 'smilesColumn':
                    text = str(value)
                    if text not in smiles_cache:
                        mol = Chem.MolFromSmiles(text) if text.strip() else None
                        if mol is None:
                            smiles_cache[text] = reasons['smilesColumn']
                        elif allowed_symbols is not None and any(atom.GetSymbol() not in allowed_symbols for atom in mol.GetAtoms()):
                            unsupported = sorted({atom.GetSymbol() for atom in mol.GetAtoms()} - allowed_symbols)
                            smiles_cache[text] = 'Unsupported element(s) for the molecule encoder: ' + ', '.join(unsupported)
                        else:
                            smiles_cache[text] = None
                    reason = smiles_cache[text]
                    valid = reason is None
                elif key == 'adductTypeColumn':
                    text = str(value)
                    if text not in adduct_cache:
                        try:
                            fragmenter.get_index_by_adduct_type(Adduct.parse(text))
                            adduct_cache[text] = True
                        except Exception:
                            adduct_cache[text] = False
                    valid = adduct_cache[text]
                elif key == 'precursorMzColumn':
                    valid = math.isfinite(float(value)) and float(value) > 0
                else:
                    try:
                        mz = float(values['precursorMzColumn'][index])
                    except (ValueError, TypeError):
                        mz = 0.0
                    text = str(value)
                    # parse_ce_to_ev only depends on precursor m/z for a "%"
                    # (NCE) value; every other form can be cached by string
                    # alone (this caller never passes an instrument, so the
                    # other mz-dependent branch is unreachable here).
                    if '%' in text:
                        energy = parse_ce_to_ev(value, mz)
                    else:
                        if text not in ce_cache:
                            ce_cache[text] = parse_ce_to_ev(value, mz)
                        energy = ce_cache[text]
                    valid = energy is not None and math.isfinite(float(energy)) and float(energy) >= 0
            except Exception:
                valid = False
            item['valid'] += int(valid)
            if not valid:
                failures.append({'field': key, 'column': item['column'],
                                 'reason': smiles_cache.get(str(value)) or reasons[key] if key == 'smilesColumn' else reasons[key]})
        if failures:
            invalid_count += 1
            if invalid_detail_limit is None or len(invalid) < invalid_detail_limit:
                invalid.append({'index': index, 'id': ids[index] if ids else str(index),
                                'values': {key: str(items[index]) for key, items in values.items()},
                                'issues': failures})
        else:
            accepted.append(index)
        if progress and ((index + 1) % progress_step == 0 or index + 1 == total):
            progress(index + 1, total)
    return {'validation': validation, 'invalidRecords': invalid, 'validIndexes': accepted,
            'invalidRecordCount': invalid_count, 'eligibleRecords': len(accepted), 'excludedRecords': invalid_count,
            'eligibleUniqueSmiles': len({str(values['smilesColumn'][index]) for index in accepted})}
