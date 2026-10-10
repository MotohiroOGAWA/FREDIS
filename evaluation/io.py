from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".mssim")


def _plain_table(path: Path) -> pd.DataFrame:
    """Read a table that is not an msentity container."""
    suffix = path.suffix.lower()
    if suffix == ".msds":
        try:
            from fredis.libs.msentity.msentity import MSDataset
        except ImportError:
            from msentity import MSDataset
        return MSDataset.load(str(path), load_peak_metadata=False).metadata
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, dict):
            for key in ("rows", "records", "data", "results"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError("JSON input must be an array, or contain a rows/records/data/results array.")
        return pd.DataFrame(value)
    raise ValueError(f"Unsupported table type {suffix!r}.")


def _similarity_table(path: Path) -> pd.DataFrame:
    """Load the public msentity SimilarityDataset schema without requiring its version."""
    with h5py.File(path, "r") as handle:
        file_format = handle.attrs.get("format")
        if isinstance(file_format, bytes):
            file_format = file_format.decode("utf-8")
        if file_format != "msentity.similarity":
            raise ValueError("Not an msentity similarity file.")
        # msentity.SimilarityDataset.load() itself accepts schema versions 1 and 2
        # (2 only adds an optional embedded-matched-data group; table.parquet's
        # own columns are unchanged), so this reader must too.
        if int(handle.attrs.get("schema_version", -1)) not in (1, 2):
            raise ValueError("Unsupported msentity similarity schema version.")
        if "table.parquet" not in handle:
            raise ValueError("Similarity file does not contain table.parquet.")
        frame = pd.read_parquet(io.BytesIO(handle["table.parquet"][()].tobytes()))
    required = {"index1", "index2", "cosine_similarity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Similarity table is missing required columns: {sorted(missing)}")
    return frame


def _attach_metadata(
    frame: pd.DataFrame, metadata_path: str, join_column: str | None,
    metadata_join_column: str | None,
) -> tuple[pd.DataFrame, str, str]:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Metadata file does not exist: {path}")
    metadata = _plain_table(path).reset_index(drop=True)
    if join_column:
        result_key = join_column
        metadata_key = metadata_join_column or join_column
    elif metadata_join_column:
        metadata_key = metadata_join_column
        result_key = (
            "SourceSpecID"
            if metadata_key == "SpecID" and "SourceSpecID" in frame.columns
            else metadata_key
        )
    else:
        pair = next((candidate for candidate in (
            ("SourceSpecID", "SourceSpecID"),
            ("SourceSpecID", "SpecID"), ("SpecID", "SpecID"),
            ("IDENTIFIER", "IDENTIFIER"),
        ) if candidate[0] in frame.columns and candidate[1] in metadata.columns), None)
        if pair:
            result_key, metadata_key = pair
        else:
            result_key = metadata_key = None
    if not result_key or not metadata_key:
        common = [str(column) for column in frame.columns if column in metadata.columns]
        raise ValueError(
            "Select the result and metadata join columns. They may have different names. "
            f"Same-name columns: {common or 'none'}"
        )
    if result_key not in frame.columns:
        raise ValueError(f"Result join column {result_key!r} was not found in the .mssim result.")
    if metadata_key not in metadata.columns:
        raise ValueError(f"Metadata join column {metadata_key!r} was not found in {path}.")
    non_missing = metadata[metadata_key].notna()
    duplicate = metadata.loc[non_missing, metadata_key].duplicated(keep=False)
    if duplicate.any():
        examples = metadata.loc[non_missing].loc[duplicate, metadata_key].drop_duplicates().tolist()[:5]
        raise ValueError(
            f"Metadata join column {metadata_key!r} must be unique; duplicate examples: {examples}"
        )
    # Missing right-hand keys can never match a real source id. Pandas also
    # considers repeated null keys non-unique during many-to-one validation,
    # so discard them before the merge rather than rejecting otherwise valid
    # prediction datasets that contain additional rows without SourceSpecID.
    metadata = metadata.loc[non_missing].copy()
    temporary_key = "__fredis_metadata_join_key__"
    while temporary_key in frame.columns or temporary_key in metadata.columns:
        temporary_key += "_"
    metadata = metadata.rename(columns={metadata_key: temporary_key})
    renamed = {
        column: f"metadata.{column}"
        for column in metadata.columns
        if column != temporary_key and column in frame.columns
    }
    metadata = metadata.rename(columns=renamed)
    combined = frame.merge(
        metadata, left_on=result_key, right_on=temporary_key,
        how="left", sort=False, validate="many_to_one",
    ).drop(columns=temporary_key)
    return combined, result_key, metadata_key


def read_table(
    source: str | Path, *, metadata: str | None = None,
    join_column: str | None = None, metadata_join_column: str | None = None,
) -> pd.DataFrame:
    """Read a supported tabular result file without changing its columns."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".mssim":
        frame = _similarity_table(path)
        if metadata:
            frame, _, _ = _attach_metadata(
                frame, metadata, join_column, metadata_join_column
            )
        return frame
    if metadata:
        raise ValueError("Metadata sources can only be attached to an .mssim input.")
    try:
        return _plain_table(path)
    except ValueError as error:
        raise ValueError(f"Unsupported input type {path.suffix.lower()!r}. Supported: {', '.join(SUPPORTED_SUFFIXES)}") from error
