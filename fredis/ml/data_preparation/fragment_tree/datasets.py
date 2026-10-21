"""Dataset loading shared by preparation and Workbench inspection."""

from __future__ import annotations
import csv
import io
import random
from pathlib import Path
from fredis.libs.msentity.msentity import MSDataset

SUPPORTED_SUFFIXES = (".msds", ".msp", ".mgf", ".tsv", ".csv", ".parquet")


def load_spectrum_dataset(path: str | Path) -> MSDataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".msds":
        return MSDataset.load(str(path), load_peak_metadata=False)
    if suffix == ".msp":
        from fredis.libs.msentity.msentity.io.msp import read_msp

        return read_msp(str(path), show_progress=False, normalize_intensity=False)
    if suffix == ".mgf":
        from fredis.libs.msentity.msentity.io.mgf import read_mgf

        return read_mgf(str(path), show_progress=False, normalize_intensity=False)
    from fredis.libs.msentity.msentity.io.tsv import read_tsv, read_tsv_text

    if suffix == ".tsv":
        return read_tsv(path, show_progress=False)
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t")
        writer.writerows(rows)
        return read_tsv_text(output.getvalue(), source_name=str(path))
    if suffix == ".parquet":
        import pandas as pd

        return read_tsv_text(
            pd.read_parquet(path).to_csv(sep="\t", index=False), source_name=str(path)
        )
    raise ValueError(
        "Unsupported dataset format. Use MSDataset, MSP, MGF or a TSV / CSV / Parquet spectrum table with a Peak column."
    )


def canonical_smiles(value: str) -> str:
    """RDKit canonical isomeric SMILES; retain stereochemistry and charge."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {value}")
    # Atom-map numbers are annotations, not molecular identity.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_keys(values):
    cache = {}
    result = []
    for value in values:
        value = str(value)
        if value not in cache:
            cache[value] = canonical_smiles(value)
        result.append(cache[value])
    return result


def canonical_column(dataset, column):
    values = dataset[column].astype(str)
    return values.map(dict(zip(values, canonical_keys(values))))


def validate_split_ratios(validation_ratio, test_ratio):
    import math

    ratios = (validation_ratio, test_ratio)
    if any(not math.isfinite(r) or not 0 <= r < 1 for r in ratios) or sum(ratios) >= 1:
        raise ValueError("Validation and test ratios must be finite, non-negative and sum to less than 1.")


def split_assignments(values, validation_ratio=0.0, test_ratio=0.0, seed=0):
    """Return canonical keys and their deterministic, molecule-disjoint assignment.

    Fractions use the original molecule count. Round each held-out count, with
    at least one molecule for each enabled split, and reserve one for training.
    """
    validate_split_ratios(validation_ratio, test_ratio)
    ratios = (validation_ratio, test_ratio)
    keys = canonical_keys(values)
    molecules = sorted(set(keys))
    counts = [max(1, round(len(molecules) * r)) if r else 0 for r in ratios]
    if len(molecules) < 1 + sum(r > 0 for r in ratios):
        raise ValueError("Not enough canonical molecules for the requested non-empty splits.")
    # Rounding can over-allocate very small datasets; reduce the largest holdout.
    while sum(counts) >= len(molecules):
        index = max(range(2), key=lambda i: counts[i])
        counts[index] -= 1
    random.Random(seed).shuffle(molecules)
    nval, ntest = counts
    assignments = {key: ("validation" if i < nval else "test" if i < nval + ntest else "train")
                   for i, key in enumerate(molecules)}
    return keys, assignments


def split_dataset(dataset, column, validation_ratio=0.0, test_ratio=0.0, seed=0):
    import numpy as np

    keys, assignments = split_assignments(dataset[column].astype(str), validation_ratio, test_ratio, seed)
    labels = np.array([assignments[key] for key in keys])
    return {name: dataset[np.flatnonzero(labels == name)] for name in ("train", "validation", "test")}, assignments


def split_by_smiles(dataset: MSDataset, column: str, ratio: float, seed: int) -> tuple[MSDataset, MSDataset]:
    if not 0 < ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1.")
    splits, _ = split_dataset(dataset, column, ratio, seed=seed)
    return splits["train"], splits["validation"]


def dedupe_validation(
    train: MSDataset,
    validation: MSDataset,
    train_column: str,
    validation_column: str,
    ratio: float | None = None,
    seed: int = 0,
) -> tuple[MSDataset, dict]:
    """Remove validation records whose SMILES also appears in training, then
    optionally cap validation to `ratio` of training's unique SMILES count.

    A model must not be scored on a molecule it trained on, so the overlap is
    always removed rather than raising. Likewise, a validation set smaller
    than the requested ratio (or empty) is reported back for the caller to
    warn about, never a reason to abort an otherwise valid run.
    """
    train_smiles = set(canonical_keys(train[train_column].astype(str)))
    overlap_mask = canonical_column(validation, validation_column).isin(train_smiles)
    report = {"removedOverlapRecords": int(overlap_mask.sum())}
    validation = validation[~overlap_mask]
    if ratio is not None:
        if not 0 < ratio < 1:
            raise ValueError("Validation ratio must be between 0 and 1.")
        target_smiles = list(dict.fromkeys(canonical_keys(train[train_column].astype(str))))
        target_count = max(1, round(len(target_smiles) * ratio))
        available = list(
            dict.fromkeys(canonical_keys(validation[validation_column].astype(str)))
        )
        report["targetSmiles"] = target_count
        report["availableSmiles"] = len(available)
        if len(available) > target_count:
            random.Random(seed).shuffle(available)
            kept = set(available[:target_count])
            validation = validation[
                canonical_column(validation, validation_column).isin(kept)
            ]
    report["remainingRecords"] = len(validation)
    report["remainingSmiles"] = len(set(canonical_keys(validation[validation_column].astype(str))))
    return validation, report
