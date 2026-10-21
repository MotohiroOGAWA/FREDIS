"""Count SMARTS substructure matches in an MSDataset."""

from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase


def count_matches(values, smarts, include_compounds=False):
    multiple = not isinstance(smarts, str)
    patterns = [value.strip() for value in (smarts if multiple else [smarts])]
    if not patterns:
        raise ValueError("At least one reactant SMARTS is required.")
    queries = []
    for index, pattern in enumerate(patterns):
        with rdBase.BlockLogs():
            query = Chem.MolFromSmarts(pattern)
        if query is None or query.GetNumAtoms() == 0:
            raise ValueError(
                f"Invalid or empty reactant SMARTS #{index + 1}: {pattern}"
            )
        queries.append(query)
    pattern_counts = [0] * len(queries)
    total = valid = matched = missing = invalid = 0
    unique = {}
    cache = {}
    compounds = {}
    for value in values:
        total += 1
        if value is None or pd.isna(value) or not str(value).strip():
            missing += 1
            continue
        smiles = str(value).strip()
        if smiles not in cache:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(smiles)
            cache[smiles] = (
                None
                if mol is None or not mol.GetNumAtoms()
                else (
                    Chem.MolToSmiles(mol),
                    tuple(mol.HasSubstructMatch(query) for query in queries),
                )
            )
        entry = cache[smiles]
        if entry is None:
            invalid += 1
            continue
        canonical, hits = entry
        hit = any(hits)
        for index, pattern_hit in enumerate(hits):
            pattern_counts[index] += int(pattern_hit)
        valid += 1
        matched += int(hit)
        unique[canonical] = hits
        if include_compounds:
            if canonical not in compounds:
                compounds[canonical] = {
                    "smiles": canonical,
                    "matched": bool(hit),
                    "records": 0,
                }
            if multiple:
                compounds[canonical]["matchedPatterns"] = [
                    i for i, value in enumerate(hits) if value
                ]
            compounds[canonical]["records"] += 1
    unique_matched = sum(any(hits) for hits in unique.values())
    result = dict(
        total=total,
        valid=valid,
        matched=matched,
        missing=missing,
        invalid=invalid,
        unique=len(unique),
        uniqueMatched=unique_matched,
        percent=100 * matched / valid if valid else None,
        uniquePercent=100 * unique_matched / len(unique) if unique else None,
    )

    if multiple:
        result["patterns"] = []
        for index, pattern in enumerate(patterns):
            unique_count = sum(hits[index] for hits in unique.values())
            result["patterns"].append(
                dict(
                    index=index,
                    smarts=pattern,
                    matched=pattern_counts[index],
                    valid=valid,
                    unique=len(unique),
                    uniqueMatched=unique_count,
                    percent=100 * pattern_counts[index] / valid if valid else None,
                    uniquePercent=100 * unique_count / len(unique) if unique else None,
                )
            )
    if include_compounds:
        result["compounds"] = list(compounds.values())
    return result


def search(file, smarts, column="SMILES", include_compounds=False):
    from fredis.libs.msentity.msentity import MSDataset

    filename = Path(file).expanduser()
    if filename.suffix.lower() != ".msds" or not filename.is_file():
        raise ValueError("Select an existing .msds file.")
    smarts = (
        smarts.strip()
        if isinstance(smarts, str)
        else [value.strip() for value in smarts]
    )
    # Validate before loading a potentially large dataset.
    count_matches([], smarts)
    column = column.strip()
    dataset = MSDataset.load(str(filename), load_peak_metadata=False)
    metadata = dataset.metadata
    if column not in metadata.columns:
        raise ValueError(
            f"SMILES column {column!r} not found. Available: {', '.join(map(str, metadata.columns))}"
        )
    return {
        "file": str(filename),
        "smarts": smarts,
        "column": column,
        **count_matches(metadata[column], smarts, include_compounds=include_compounds),
    }
