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


def split_by_smiles(
    dataset: MSDataset, column: str, ratio: float, seed: int
) -> tuple[MSDataset, MSDataset]:
    if not 0 < ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1.")
    smiles = list(dict.fromkeys(dataset[column].astype(str).tolist()))
    if len(smiles) < 2:
        raise ValueError(
            "Automatic validation split requires at least two unique SMILES. Supply a separate validation dataset."
        )
    random.Random(seed).shuffle(smiles)
    count = min(len(smiles) - 1, max(1, round(len(smiles) * ratio)))
    held_out = set(smiles[:count])
    mask = dataset[column].astype(str).isin(held_out)
    return dataset[~mask], dataset[mask]


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
    train_smiles = set(train[train_column].astype(str))
    overlap_mask = validation[validation_column].astype(str).isin(train_smiles)
    report = {"removedOverlapRecords": int(overlap_mask.sum())}
    validation = validation[~overlap_mask]
    if ratio is not None:
        if not 0 < ratio < 1:
            raise ValueError("Validation ratio must be between 0 and 1.")
        target_smiles = list(dict.fromkeys(train[train_column].astype(str).tolist()))
        target_count = max(1, round(len(target_smiles) * ratio))
        available = list(
            dict.fromkeys(validation[validation_column].astype(str).tolist())
        )
        report["targetSmiles"] = target_count
        report["availableSmiles"] = len(available)
        if len(available) > target_count:
            random.Random(seed).shuffle(available)
            kept = set(available[:target_count])
            validation = validation[
                validation[validation_column].astype(str).isin(kept)
            ]
    report["remainingRecords"] = len(validation)
    report["remainingSmiles"] = len(set(validation[validation_column].astype(str)))
    return validation, report
