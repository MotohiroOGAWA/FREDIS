"""Python binding of the msentity C++ library.

    import msentity as ms

    ds = ms.load("example.msp")
    len(ds)                     # 2
    ds["Name"]                  # ['MassSpecID0000001', 'MassSpecID0000002']
    ds[0].mz                    # array([ 91.0542, 125.0233, 185.0961, 246.1125])
    ds[[1]].save("one.msds")

All work is done by the shared C library (libmsentity_c); this module only
converts arguments and results.
"""

from __future__ import annotations

import ctypes
import math
import os
import weakref
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from . import _native as _n

__all__ = [
    "Dataset",
    "Record",
    "MsEntityError",
    "InvalidArgumentError",
    "PositionError",
    "ColumnError",
    "FormatError",
    "FileError",
    "load",
    "read_text",
    "from_peaks",
    "concat",
    "from_bytes",
    "__version__",
]

__version__ = _n.lib.msentity_version().decode()


# ---- errors ----------------------------------------------------------------------------

class MsEntityError(Exception):
    """Base class of all errors raised by msentity."""


class InvalidArgumentError(MsEntityError, ValueError):
    """Invalid argument or invalid data (C status MSENTITY_ERR_INVALID_ARGUMENT)."""


class PositionError(MsEntityError, IndexError):
    """Spectrum or peak position out of range (MSENTITY_ERR_OUT_OF_RANGE)."""


class ColumnError(MsEntityError, KeyError):
    """Unknown column (MSENTITY_ERR_KEY)."""

    def __str__(self) -> str:  # KeyError would show the message in quotes
        return str(self.args[0]) if self.args else ""


class FormatError(MsEntityError):
    """Malformed or unreadable file (MSENTITY_ERR_FORMAT)."""


class FileError(MsEntityError, OSError):
    """File cannot be opened or written (MSENTITY_ERR_IO)."""


_ERRORS = {1: InvalidArgumentError, 2: PositionError, 3: ColumnError, 4: FormatError, 5: FileError}


def _check(status: int) -> None:
    if status != 0:
        message = _n.lib.msentity_last_error().decode("utf-8", "replace")
        raise _ERRORS.get(status, MsEntityError)(message)


# ---- conversions ---------------------------------------------------------------------------

def _encode(text: str | os.PathLike | None) -> bytes | None:
    if text is None:
        return None
    return os.fspath(text).encode("utf-8")


def _take_string(pointer: ctypes.c_void_p) -> str:
    try:
        return ctypes.string_at(pointer.value).decode("utf-8")
    finally:
        _n.lib.msentity_string_free(pointer)


def _from_value(value: _n.Value) -> Any:
    kind = value.type
    if kind == _n.TYPE_BOOL:
        return bool(value.bool_value)
    if kind == _n.TYPE_INT:
        return value.int_value
    if kind == _n.TYPE_FLOAT:
        return value.float_value
    if kind == _n.TYPE_STRING:
        return ctypes.string_at(value.string_value).decode("utf-8")
    return None


def _to_value(item: Any, out: _n.Value, keep: list) -> None:
    """Fill a msentity_value; `keep` holds encoded strings alive during the call."""
    if item is None or (isinstance(item, float) and math.isnan(item)):
        out.type = _n.TYPE_NULL
    elif isinstance(item, (bool, np.bool_)):
        out.type, out.bool_value = _n.TYPE_BOOL, int(bool(item))
    elif isinstance(item, (int, np.integer)):
        out.type, out.int_value = _n.TYPE_INT, int(item)
    elif isinstance(item, (float, np.floating)):
        out.type, out.float_value = _n.TYPE_FLOAT, float(item)
    elif isinstance(item, str):
        encoded = ctypes.create_string_buffer(item.encode("utf-8"))
        keep.append(encoded)
        out.type, out.string_value = _n.TYPE_STRING, ctypes.addressof(encoded)
    else:
        raise TypeError(f"unsupported value type: {type(item).__name__}")


def _value_array(items: Sequence[Any]) -> tuple[ctypes.Array, list]:
    array = (_n.Value * len(items))()
    keep: list = []
    for i, item in enumerate(items):
        _to_value(item, array[i], keep)
    return array, keep


def _read_values(fill, n: int) -> list:
    array = (_n.Value * n)()
    _check(fill(array, n))
    try:
        return [_from_value(array[i]) for i in range(n)]
    finally:
        _n.lib.msentity_values_clear(array, n)


def _arrow_table(export) -> "Any":
    import pyarrow as pa

    stream = _n.ArrowArrayStream()
    _check(export(ctypes.byref(stream)))
    reader = pa.RecordBatchReader._import_from_c(ctypes.addressof(stream))
    return reader.read_all()


def _new_handle(call) -> "Dataset":
    handle = ctypes.c_void_p()
    _check(call(ctypes.byref(handle)))
    return Dataset._wrap(handle)


# ---- module functions --------------------------------------------------------------------------

def load(path: str | os.PathLike, file_type: str | None = None, *,
         spec_id_prefix: str | None = None, infer_types: bool = False) -> "Dataset":
    """Load MSP, MGF, MSDS, TSV, or CSV (format from the extension unless
    ``file_type`` is given). ``infer_types`` converts numeric TSV/CSV columns."""
    return _new_handle(lambda out: _n.lib.msentity_load(
        _encode(path), _encode(file_type), _encode(spec_id_prefix), int(infer_types), out))


def read_text(text: str, format: str, *, infer_types: bool = False) -> "Dataset":
    """Parse ``text`` in "msp", "mgf", "tsv", or "csv" format."""
    return _new_handle(lambda out: _n.lib.msentity_read_text(
        text.encode("utf-8"), format.encode(), int(infer_types), out))


def from_peaks(spectra: Iterable[Any]) -> "Dataset":
    """Build a dataset without metadata. Each spectrum is an (n, 2) array-like
    of (m/z, intensity) rows."""
    arrays = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in spectra]
    data = np.ascontiguousarray(np.concatenate(arrays) if arrays else np.empty((0, 2)))
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(a) for a in arrays])
    return _new_handle(lambda out: _n.lib.msentity_from_peaks(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(data),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(offsets), out))


def from_bytes(data: bytes) -> "Dataset":
    """Dataset from the bytes of an .msds file (see Dataset.to_bytes)."""
    data = bytes(data)
    return _new_handle(lambda out: _n.lib.msentity_from_msds_bytes(data, len(data), out))


def concat(datasets: Sequence["Dataset"]) -> "Dataset":
    """Join datasets; columns are the union, missing cells are None."""
    handles = (ctypes.c_void_p * len(datasets))(*[d._handle for d in datasets])
    return _new_handle(lambda out: _n.lib.msentity_concat(handles, len(datasets), out))


# ---- Dataset --------------------------------------------------------------------------------------

class Dataset:
    """Spectra with their metadata and peaks.

    Indexing returns views that share data with this dataset:
    ``ds[1:3]``, ``ds[[2, 0]]``, ``ds[np.array([True, False])]``. Use
    :meth:`copy` for an independent dataset.
    """

    __slots__ = ("_handle", "_finalizer", "__weakref__")

    def __init__(self) -> None:
        raise TypeError("use msentity.load(), read_text(), or from_peaks()")

    @classmethod
    def _wrap(cls, handle: ctypes.c_void_p) -> "Dataset":
        self = object.__new__(cls)
        self._handle = handle
        self._finalizer = weakref.finalize(self, _n.lib.msentity_free, handle)
        return self

    def close(self) -> None:
        """Release the native dataset now (views of it stay valid)."""
        self._finalizer()

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- shape --------------------------------------------------------------------------
    def __len__(self) -> int:
        return _n.lib.msentity_size(self._handle)

    @property
    def n_peaks_total(self) -> int:
        return _n.lib.msentity_n_peaks_total(self._handle)

    def __repr__(self) -> str:
        return (f"Dataset(n_spectra={len(self)}, n_peaks={self.n_peaks_total}, "
                f"columns={self.columns})")

    # -- columns ------------------------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        """Visible metadata columns, in display order."""
        names = []
        for i in range(_n.lib.msentity_n_columns(self._handle)):
            out = ctypes.c_void_p()
            _check(_n.lib.msentity_column_name(self._handle, i, ctypes.byref(out)))
            names.append(_take_string(out))
        return names

    @columns.setter
    def columns(self, names: Sequence[str]) -> None:
        encoded = (ctypes.c_char_p * len(names))(*[n.encode() for n in names])
        _check(_n.lib.msentity_set_columns(self._handle, encoded, len(names)))

    def __contains__(self, column: str) -> bool:
        return bool(_n.lib.msentity_has_column(self._handle, column.encode()))

    # -- values ---------------------------------------------------------------------------
    def column(self, name: str) -> list:
        """Values of a column for every spectrum (None = missing)."""
        return _read_values(
            lambda out, n: _n.lib.msentity_get_column(self._handle, name.encode(), out, n),
            len(self))

    def value(self, position: int, column: str) -> Any:
        out = _n.Value()
        _check(_n.lib.msentity_get_value(self._handle, position, column.encode(), ctypes.byref(out)))
        try:
            return _from_value(out)
        finally:
            _n.lib.msentity_values_clear(ctypes.byref(out), 1)

    def set_column(self, name: str, values: Any) -> None:
        """Set a column: one value per spectrum, or one value for all."""
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            single = _n.Value()
            keep: list = []
            _to_value(values, single, keep)
            _check(_n.lib.msentity_set_column_scalar(self._handle, name.encode(), ctypes.byref(single)))
            return
        items = list(values)
        array, keep = _value_array(items)
        _check(_n.lib.msentity_set_column(self._handle, name.encode(), array, len(items)))

    def set_value(self, position: int, column: str, value: Any) -> None:
        single = _n.Value()
        keep: list = []
        _to_value(value, single, keep)
        _check(_n.lib.msentity_set_value(self._handle, position, column.encode(), ctypes.byref(single)))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.column(key)
        if isinstance(key, (int, np.integer)):
            position = int(key)
            if position < 0:
                position += len(self)
            if not 0 <= position < len(self):
                raise PositionError(f"spectrum position out of range: {key}")
            return Record(self, position)
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step == 1:
                return self.slice(start, stop)
            return self.select(range(start, stop, step))
        array = np.asarray(key)
        if array.dtype == bool:
            return self.filter(array)
        return self.select(array)

    def __setitem__(self, column: str, values: Any) -> None:
        self.set_column(column, values)

    def __iter__(self) -> Iterator["Record"]:
        for position in range(len(self)):
            yield Record(self, position)

    # -- views --------------------------------------------------------------------------------
    def select(self, positions: Iterable[int]) -> "Dataset":
        array = np.ascontiguousarray(np.fromiter(positions, dtype=np.int64))
        return _new_handle(lambda out: _n.lib.msentity_select(
            self._handle, array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(array), out))

    def slice(self, start: int, stop: int) -> "Dataset":
        return _new_handle(lambda out: _n.lib.msentity_slice(self._handle, start, stop, out))

    def filter(self, mask: Sequence[bool]) -> "Dataset":
        array = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
        return _new_handle(lambda out: _n.lib.msentity_filter(
            self._handle, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(array), out))

    def sort_by(self, column: str, ascending: bool = True) -> "Dataset":
        """Stable sort by a column; missing values last."""
        return _new_handle(lambda out: _n.lib.msentity_sort_by(
            self._handle, column.encode(), int(ascending), out))

    def copy(self) -> "Dataset":
        return _new_handle(lambda out: _n.lib.msentity_copy(self._handle, out))

    def full_view(self, reset_columns: bool = False) -> "Dataset":
        return _new_handle(lambda out: _n.lib.msentity_full_view(self._handle, int(reset_columns), out))

    # -- peaks -----------------------------------------------------------------------------------
    @property
    def offsets(self) -> np.ndarray:
        """Spectrum i has peaks offsets[i] … offsets[i + 1] - 1."""
        out = np.empty(len(self) + 1, dtype=np.int64)
        _check(_n.lib.msentity_offsets(self._handle, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(out)))
        return out

    def _peak_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_peaks_total
        mz, intensity = np.empty(n), np.empty(n)
        pointer = ctypes.POINTER(ctypes.c_double)
        _check(_n.lib.msentity_peaks(self._handle, mz.ctypes.data_as(pointer),
                                     intensity.ctypes.data_as(pointer), n))
        return mz, intensity

    @property
    def mz(self) -> np.ndarray:
        """m/z of every peak of every spectrum (a copy)."""
        return self._peak_arrays()[0]

    @property
    def intensity(self) -> np.ndarray:
        return self._peak_arrays()[1]

    def spectrum(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        """(m/z, intensity) of one spectrum."""
        size = ctypes.c_int64()
        _check(_n.lib.msentity_spectrum_size(self._handle, position, ctypes.byref(size)))
        mz, intensity = np.empty(size.value), np.empty(size.value)
        pointer = ctypes.POINTER(ctypes.c_double)
        _check(_n.lib.msentity_spectrum_peaks(self._handle, position, mz.ctypes.data_as(pointer),
                                              intensity.ctypes.data_as(pointer), size.value))
        return mz, intensity

    def normalize(self, scale: float = 1.0) -> None:
        """In place: the highest peak of each spectrum becomes ``scale``."""
        _check(_n.lib.msentity_normalize(self._handle, scale))

    def sort_peaks(self, by: str = "mz", ascending: bool | None = None) -> None:
        """In place: sort the peaks of each spectrum by "mz" (default ascending)
        or "intensity" (default descending)."""
        if by not in ("mz", "intensity"):
            raise InvalidArgumentError('by must be "mz" or "intensity"')
        if ascending is None:
            ascending = by == "mz"
        _check(_n.lib.msentity_sort_peaks(self._handle, 0 if by == "mz" else 1, int(ascending)))

    # -- peak metadata -------------------------------------------------------------------------------
    @property
    def peak_columns(self) -> list[str]:
        names = []
        for i in range(_n.lib.msentity_n_peak_columns(self._handle)):
            out = ctypes.c_void_p()
            _check(_n.lib.msentity_peak_column_name(self._handle, i, ctypes.byref(out)))
            names.append(_take_string(out))
        return names

    def peak_column(self, name: str) -> list:
        """Values of a peak column for every peak."""
        return _read_values(
            lambda out, n: _n.lib.msentity_get_peak_column(self._handle, name.encode(), out, n),
            self.n_peaks_total)

    def set_peak_column(self, name: str, values: Sequence[Any]) -> None:
        items = list(values)
        array, keep = _value_array(items)
        _check(_n.lib.msentity_set_peak_column(self._handle, name.encode(), array, len(items)))

    # -- dataset-level metadata -------------------------------------------------------------------------
    @property
    def description(self) -> str:
        out = ctypes.c_void_p()
        _check(_n.lib.msentity_get_description(self._handle, ctypes.byref(out)))
        return _take_string(out)

    @description.setter
    def description(self, text: str) -> None:
        _check(_n.lib.msentity_set_description(self._handle, text.encode()))

    @property
    def attributes(self) -> dict[str, str]:
        """A copy of the attributes (use set_attribute / remove_attribute to change them)."""
        result = {}
        for i in range(_n.lib.msentity_n_attributes(self._handle)):
            key, value = ctypes.c_void_p(), ctypes.c_void_p()
            _check(_n.lib.msentity_attribute(self._handle, i, ctypes.byref(key), ctypes.byref(value)))
            result[_take_string(key)] = _take_string(value)
        return result

    def set_attribute(self, key: str, value: str) -> None:
        _check(_n.lib.msentity_set_attribute(self._handle, key.encode(), value.encode()))

    def remove_attribute(self, key: str) -> bool:
        removed = ctypes.c_int32()
        _check(_n.lib.msentity_remove_attribute(self._handle, key.encode(), ctypes.byref(removed)))
        return bool(removed.value)

    @property
    def tags(self) -> list[str]:
        result = []
        for i in range(_n.lib.msentity_n_tags(self._handle)):
            out = ctypes.c_void_p()
            _check(_n.lib.msentity_tag(self._handle, i, ctypes.byref(out)))
            result.append(_take_string(out))
        return result

    def add_tag(self, tag: str) -> bool:
        added = ctypes.c_int32()
        _check(_n.lib.msentity_add_tag(self._handle, tag.encode(), ctypes.byref(added)))
        return bool(added.value)

    def remove_tag(self, tag: str) -> bool:
        removed = ctypes.c_int32()
        _check(_n.lib.msentity_remove_tag(self._handle, tag.encode(), ctypes.byref(removed)))
        return bool(removed.value)

    def set_spec_id(self, column: str = "SpecID", prefix: str = "", *,
                    overwrite: bool = False, start: int = 1) -> bool:
        """Sequential zero-padded IDs (SP01 … SP12). False if the column exists."""
        assigned = ctypes.c_int32()
        _check(_n.lib.msentity_set_spec_id(self._handle, column.encode(), prefix.encode(),
                                           int(overwrite), start, ctypes.byref(assigned)))
        return bool(assigned.value)

    # -- saving and exporting ------------------------------------------------------------------------------
    def save(self, path: str | os.PathLike, format: str | None = None, *,
             save_view: bool = True, append: bool = False) -> None:
        """Write MSDS, MSP, MGF, TSV, or CSV (format from the extension unless given)."""
        _check(_n.lib.msentity_save(self._handle, _encode(path), _encode(format),
                                    int(save_view), int(append)))

    def to_bytes(self) -> bytes:
        """The visible data as the bytes of an .msds file. Other msentity
        libraries (such as msentity-similarity) exchange datasets this way."""
        pointer, size = ctypes.c_void_p(), ctypes.c_int64()
        _check(_n.lib.msentity_to_msds_bytes(self._handle, ctypes.byref(pointer), ctypes.byref(size)))
        try:
            return ctypes.string_at(pointer.value, size.value)
        finally:
            _n.lib.msentity_bytes_free(pointer)

    def to_arrow(self):
        """Visible metadata as a pyarrow.Table (requires pyarrow)."""
        return _arrow_table(lambda s: _n.lib.msentity_metadata_arrow(self._handle, s))

    def peak_metadata_to_arrow(self):
        """Peak metadata (one row per peak) as a pyarrow.Table."""
        return _arrow_table(lambda s: _n.lib.msentity_peak_metadata_arrow(self._handle, s))

    def to_pandas(self):
        """Visible metadata as a pandas.DataFrame (requires pyarrow and pandas)."""
        return self.to_arrow().to_pandas()


class Record:
    """One spectrum of a dataset: ``rec["Name"]``, ``rec.mz``, ``rec.intensity``."""

    __slots__ = ("_dataset", "position")

    def __init__(self, dataset: Dataset, position: int) -> None:
        self._dataset = dataset
        self.position = position

    def __getitem__(self, column: str) -> Any:
        return self._dataset.value(self.position, column)

    def __setitem__(self, column: str, value: Any) -> None:
        self._dataset.set_value(self.position, column, value)

    def __contains__(self, column: str) -> bool:
        return column in self._dataset

    @property
    def peaks(self) -> tuple[np.ndarray, np.ndarray]:
        return self._dataset.spectrum(self.position)

    @property
    def mz(self) -> np.ndarray:
        return self.peaks[0]

    @property
    def intensity(self) -> np.ndarray:
        return self.peaks[1]

    @property
    def n_peaks(self) -> int:
        size = ctypes.c_int64()
        _check(_n.lib.msentity_spectrum_size(self._dataset._handle, self.position, ctypes.byref(size)))
        return size.value

    def to_dict(self) -> dict[str, Any]:
        return {column: self[column] for column in self._dataset.columns}

    def __repr__(self) -> str:
        return f"Record(position={self.position}, n_peaks={self.n_peaks})"
