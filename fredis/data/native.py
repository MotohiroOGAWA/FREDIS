"""Conversion between FREDIS's pandas model and the native msentity bindings."""
import numpy as np
from pathlib import Path


def to_native(dataset):
    import msentity
    native = msentity.from_peaks(record.peaks.data for record in dataset)
    try:
        for column in dataset.columns:
            native[column] = dataset[column].astype(object).where(dataset[column].notna(), None).tolist()
        for column in dataset.peaks.metadata_columns:
            values = dataset.peaks.metadata[column]
            native.set_peak_column(column, values.astype(object).where(values.notna(), None).tolist())
        native.description = dataset.description
        for key, value in dataset.attributes.items():
            native.set_attribute(key, value)
        for tag in dataset.tags:
            native.add_tag(tag)
        return native
    except BaseException:
        native.close()
        raise


def from_native(native, load_peak_metadata=True):
    from . import MSDataset, PeakSeries
    metadata = native.to_pandas()
    peaks = PeakSeries(np.column_stack((native.mz, native.intensity)), native.offsets,
        native.peak_metadata_to_arrow().to_pandas()
        if load_peak_metadata and native.peak_columns else None)
    return MSDataset(metadata, peaks, description=native.description,
                     attributes=native.attributes, tags=native.tags)


def load_ms_dataset(input_file, *, file_type=None, spec_id_prefix=None,
                    show_progress=True, progress_callback=None):
    import msentity
    with msentity.load(input_file, file_type=file_type, spec_id_prefix=spec_id_prefix) as native:
        if (file_type or Path(input_file).suffix.lstrip(".")).lower() in {"msp", "mgf"}:
            native.normalize()
        result = from_native(native)
    if progress_callback:
        progress_callback(len(result), len(result), result.n_peaks_total, result.n_peaks_total)
    return result
