from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from fredis.libs.msentity.msentity import MSDataset


@dataclass(frozen=True)
class ValidationSmilesSample:
    smiles: str
    max_tanimoto_index: float
    bin_index: int


def _fingerprints(smiles_values: Sequence[str], *, radius: int, n_bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(n_bits)
    )
    valid_smiles: list[str] = []
    fingerprints = []
    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            continue
        valid_smiles.append(str(smiles))
        fingerprints.append(generator.GetFingerprint(molecule))
    return valid_smiles, fingerprints


def sample_validation_smiles_by_max_tanimoto(
    train_smiles: Sequence[str],
    validation_smiles: Sequence[str],
    *,
    ratio: float = 0.1,
    num_bins: int = 10,
    radius: int = 2,
    n_bits: int = 2048,
    seed: int = 0,
) -> list[ValidationSmilesSample]:
    """Select validation SMILES approximately uniformly over max train similarity."""
    if ratio <= 0:
        raise ValueError("validation_smiles_ratio must be positive.")
    if num_bins <= 0:
        raise ValueError("tanimoto_num_bins must be positive.")

    unique_train = list(dict.fromkeys(str(value) for value in train_smiles))
    unique_validation = list(dict.fromkeys(str(value) for value in validation_smiles))
    valid_train_smiles, train_fingerprints = _fingerprints(
        unique_train, radius=radius, n_bits=n_bits
    )
    valid_smiles, validation_fingerprints = _fingerprints(
        unique_validation, radius=radius, n_bits=n_bits
    )
    if not train_fingerprints:
        raise ValueError("No valid training SMILES were available for Tanimoto sampling.")
    if not validation_fingerprints:
        raise ValueError("No valid validation SMILES were available for Tanimoto sampling.")

    target_count = min(
        len(valid_smiles), max(1, int(math.ceil(len(valid_train_smiles) * float(ratio))))
    )
    rng = np.random.default_rng(seed)
    bins: list[list[ValidationSmilesSample]] = [[] for _ in range(num_bins)]
    for smiles, fingerprint in zip(valid_smiles, validation_fingerprints):
        maximum = float(max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints)))
        bin_index = min(int(maximum * num_bins), num_bins - 1)
        bins[bin_index].append(ValidationSmilesSample(smiles, maximum, bin_index))

    for values in bins:
        rng.shuffle(values)

    # Taking one item from every non-empty similarity interval per pass gives sparse
    # intervals equal priority; extra capacity naturally goes to denser intervals.
    selected: list[ValidationSmilesSample] = []
    while len(selected) < target_count:
        active = [values for values in bins if values]
        if not active:
            break
        remaining = target_count - len(selected)
        if remaining < len(active):
            positions = np.linspace(0, len(active) - 1, remaining).round().astype(int)
            active = [active[position] for position in positions]
        for values in active:
            selected.append(values.pop())
    return selected


def sample_validation_dataset(
    *,
    train_dataset: MSDataset | None = None,
    train_smiles: Sequence[str] | None = None,
    validation_dataset: MSDataset,
    smiles_column: str,
    ratio: float,
    num_bins: int,
    radius: int,
    n_bits: int,
    seed: int,
    output_file: str | Path,
    report_file: str | Path,
) -> Path:
    if train_smiles is None:
        if train_dataset is None:
            raise ValueError("train_dataset or train_smiles is required.")
        train_smiles = train_dataset.metadata[smiles_column].dropna().astype(str).tolist()
    validation_smiles = validation_dataset.metadata[smiles_column].dropna().astype(str).tolist()
    selected = sample_validation_smiles_by_max_tanimoto(
        train_smiles,
        validation_smiles,
        ratio=ratio,
        num_bins=num_bins,
        radius=radius,
        n_bits=n_bits,
        seed=seed,
    )
    selected_smiles = {item.smiles for item in selected}
    indexes = [
        index
        for index, value in enumerate(validation_dataset.metadata[smiles_column])
        if not pd.isna(value) and str(value) in selected_smiles
    ]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_dataset[indexes].save(str(output_path))

    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "smiles": item.smiles,
                "max_tanimoto_index": item.max_tanimoto_index,
                "tanimoto_bin": item.bin_index,
            }
            for item in selected
        ]
    ).sort_values("max_tanimoto_index").to_csv(report_path, sep="\t", index=False)
    return output_path
