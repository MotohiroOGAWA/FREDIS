"""Construct native msentity datasets from FREDIS's computed arrays.

All datasets returned here are msentity.Dataset instances. This module owns no
spectrum storage or alternative dataset model.
"""

import msentity as ms
import numpy as np
import pandas as pd


def _values(values):
    return [
        None
        if value is None or (np.isscalar(value) and pd.isna(value))
        else value.item() if isinstance(value, np.generic) else value
        for value in values
    ]


def dataset_from_arrays(
    data, offsets, metadata=None, metadata_columns=None, sort_by_mz=False
):
    data = np.asarray(data, dtype=float).reshape(-1, 2)
    offsets = np.asarray(offsets, dtype=np.int64)
    if (
        len(offsets) == 0
        or offsets[0] != 0
        or offsets[-1] != len(data)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError("Invalid spectrum offsets")
    dataset = ms.from_peaks(
        data[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])
    )
    if metadata is not None:
        columns = metadata_columns if metadata_columns is not None else metadata.columns
        for column in columns:
            dataset.set_peak_column(str(column), _values(metadata[column]))
    if sort_by_mz:
        dataset.sort_peaks()
    return dataset


def with_metadata(
    metadata, dataset, columns=None, description="", attributes=None, tags=None
):
    if len(metadata) != len(dataset):
        raise ValueError("Metadata rows must match spectra count")
    dataset = dataset.copy()
    for column in columns if columns is not None else metadata.columns:
        dataset[str(column)] = _values(metadata[column])
    if description:
        dataset.description = description
    for key, value in (attributes or {}).items():
        dataset.set_attribute(key, str(value))
    for tag in tags or []:
        dataset.add_tag(tag)
    return dataset


def set_record_peak_column(record, column, values):
    """Write one spectrum's annotations through the native peak-column API."""
    dataset = record._dataset
    offsets = dataset.offsets
    start, stop = map(int, offsets[record.position : record.position + 2])
    values = list(values)
    if len(values) != stop - start:
        raise ValueError("Peak metadata length differs from spectrum length")
    column_values = (
        dataset.peak_column(column)
        if column in dataset.peak_columns
        else [None] * dataset.n_peaks_total
    )
    column_values[start:stop] = _values(values)
    dataset.set_peak_column(column, column_values)
