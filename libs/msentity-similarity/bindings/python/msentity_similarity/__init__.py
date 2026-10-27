"""Python binding of msentity-similarity: spectrum similarity and .mssim files.

    import msentity as ms
    import msentity_similarity as mss

    query = ms.load("query.msp")
    library = ms.load("library.msp")
    result = mss.library_search(query, library, threshold=0.8)
    result["cosine_similarity"]          # [0.93, 0.88, ...]
    q, ref = result.match_records(0)     # the two spectra of row 0
    result.save("hits.mssim")

Datasets are msentity.Dataset objects; they are handed to the C library as
.msds bytes (Dataset.to_bytes). All work is done by libmsentity_similarity_c.
"""

from __future__ import annotations

import ctypes
import json
import os
import weakref
from typing import Any, Callable, Sequence

import msentity as ms
import numpy as np
from msentity import (ColumnError, FileError, FormatError, InvalidArgumentError, MsEntityError,
                      PositionError)

from . import _native as _n

__all__ = [
    "SimilarityDataset",
    "pair_scores",
    "all_pairs",
    "library_search",
    "similarity_by_key",
    "load",
    "MsEntityError",
    "InvalidArgumentError",
    "PositionError",
    "ColumnError",
    "FormatError",
    "FileError",
    "__version__",
]

__version__ = _n.lib.mssim_version().decode()

_ERRORS = {1: InvalidArgumentError, 2: PositionError, 3: ColumnError, 4: FormatError, 5: FileError}


def _check(status: int) -> None:
    if status != 0:
        message = _n.lib.mssim_last_error().decode("utf-8", "replace")
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
        _n.lib.mssim_string_free(pointer)


def _dataset_bytes(dataset: "ms.Dataset | bytes") -> bytes:
    if isinstance(dataset, (bytes, bytearray)):
        return bytes(dataset)
    if isinstance(dataset, ms.Dataset):
        return dataset.to_bytes()
    raise TypeError(f"expected msentity.Dataset or .msds bytes, got {type(dataset).__name__}")


def _options(method: str, bin_width: float, intensity_exponent: float, threads: int) -> _n.Options:
    if method not in _n.METHODS:
        raise InvalidArgumentError(f"method must be one of {sorted(_n.METHODS)}, got {method!r}")
    return _n.Options(_n.METHODS[method], int(threads), float(bin_width), float(intensity_exponent))


def _new_result(call) -> "SimilarityDataset":
    handle = ctypes.c_void_p()
    _check(call(ctypes.byref(handle)))
    return SimilarityDataset._wrap(handle)


def _i64_array(values: Sequence[int]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))


def _i64_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


# ---- calculations --------------------------------------------------------------------------

def pair_scores(ds1: "ms.Dataset", index1: Sequence[int], ds2: "ms.Dataset",
                index2: Sequence[int], *, method: str = "cosine", bin_width: float = 0.01,
                intensity_exponent: float = 1.0, threads: int = 0) -> np.ndarray:
    """Scores of ds1[index1[k]] vs ds2[index2[k]] as a float32 array."""
    a, b = _dataset_bytes(ds1), _dataset_bytes(ds2)
    i1, i2 = _i64_array(index1), _i64_array(index2)
    if len(i1) != len(i2):
        raise InvalidArgumentError("index1 and index2 must have the same length")
    out = np.empty(len(i1), dtype=np.float32)
    options = _options(method, bin_width, intensity_exponent, threads)
    _check(_n.lib.mssim_pair_scores(a, len(a), _i64_ptr(i1), b, len(b), _i64_ptr(i2), len(i1),
                                    ctypes.byref(options),
                                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    return out


def all_pairs(ds1: "ms.Dataset", ds2: "ms.Dataset", *, method: str = "cosine",
              bin_width: float = 0.01, intensity_exponent: float = 1.0,
              threads: int = 0) -> np.ndarray:
    """len(ds1) × len(ds2) float32 score matrix."""
    a, b = _dataset_bytes(ds1), _dataset_bytes(ds2)
    n1 = len(ms.from_bytes(a)) if isinstance(ds1, (bytes, bytearray)) else len(ds1)
    n2 = len(ms.from_bytes(b)) if isinstance(ds2, (bytes, bytearray)) else len(ds2)
    out = np.empty((n1, n2), dtype=np.float32)
    options = _options(method, bin_width, intensity_exponent, threads)
    _check(_n.lib.mssim_all_pairs(a, len(a), b, len(b), ctypes.byref(options),
                                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size))
    return out


def library_search(query: "ms.Dataset", reference: "ms.Dataset", threshold: float = 0.8, *,
                   method: str = "cosine", bin_width: float = 0.01,
                   intensity_exponent: float = 1.0, threads: int = 0,
                   include_matched_data: bool = True, query_source: str | None = None,
                   reference_source: str | None = None,
                   progress: Callable[[int, int], None] | None = None) -> "SimilarityDataset":
    """All (query, reference) pairs scoring at least ``threshold``.

    ``progress(done, total)`` is called as pairs are scored (from worker
    threads; exceptions raised in it are printed and ignored).
    ``query_source`` / ``reference_source`` are recorded in the metadata.
    """
    a, b = _dataset_bytes(query), _dataset_bytes(reference)
    options = _options(method, bin_width, intensity_exponent, threads)
    callback = _n.ProgressFn(lambda done, total, _: progress(done, total)) if progress else \
        _n.ProgressFn()
    return _new_result(lambda out: _n.lib.mssim_library_search(
        a, len(a), b, len(b), float(threshold), ctypes.byref(options), int(include_matched_data),
        _encode(query_source), _encode(reference_source), callback, None, out))


def similarity_by_key(ds1: "ms.Dataset", ds2: "ms.Dataset", key1: str = "SpecID",
                      key2: str = "SpecID", *, method: str = "cosine", bin_width: float = 0.01,
                      intensity_exponent: float = 1.0, threads: int = 0,
                      include_matched_data: bool = False, source1: str | None = None,
                      source2: str | None = None) -> "SimilarityDataset":
    """Scores of the spectrum pairs whose ``key1`` / ``key2`` values are equal
    (each key value must be unique within its dataset)."""
    a, b = _dataset_bytes(ds1), _dataset_bytes(ds2)
    options = _options(method, bin_width, intensity_exponent, threads)
    return _new_result(lambda out: _n.lib.mssim_similarity_by_key(
        a, len(a), b, len(b), _encode(key1), _encode(key2), ctypes.byref(options),
        int(include_matched_data), _encode(source1), _encode(source2), out))


def load(path: str | os.PathLike) -> "SimilarityDataset":
    """Read an .mssim file."""
    return _new_result(lambda out: _n.lib.mssim_load(_encode(path), out))


# ---- SimilarityDataset ---------------------------------------------------------------------

class SimilarityDataset:
    """A similarity result (one row per spectrum pair), as stored in .mssim.

    Columns: key columns (by-key results), ``index1``, ``index2`` (positions in
    the input datasets), ``cosine_similarity`` (the score of any method; the
    method is in ``metadata["parameters"]``), and ``data_index1`` /
    ``data_index2`` when the matched spectra are embedded.
    """

    __slots__ = ("_handle", "_finalizer", "_matched", "__weakref__")

    def __init__(self) -> None:
        raise TypeError("use msentity_similarity.library_search(), similarity_by_key(), or load()")

    @classmethod
    def _wrap(cls, handle: ctypes.c_void_p) -> "SimilarityDataset":
        self = object.__new__(cls)
        self._handle = handle
        self._finalizer = weakref.finalize(self, _n.lib.mssim_free, handle)
        self._matched = None
        return self

    def close(self) -> None:
        """Release the native result now."""
        self._finalizer()

    def __enter__(self) -> "SimilarityDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return _n.lib.mssim_size(self._handle)

    def __repr__(self) -> str:
        matched = "embedded" if self.has_matched_data else "lightweight"
        return f"<SimilarityDataset rows={len(self)} columns={self.columns} matched_data={matched}>"

    # -- table ------------------------------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        names = []
        for i in range(_n.lib.mssim_n_columns(self._handle)):
            pointer = ctypes.c_void_p()
            _check(_n.lib.mssim_column_name(self._handle, i, ctypes.byref(pointer)))
            names.append(_take_string(pointer))
        return names

    def column(self, name: str) -> list:
        n = len(self)
        array = (_n.Value * n)()
        _check(_n.lib.mssim_get_column(self._handle, _encode(name), array, n))
        try:
            return [ms._from_value(array[i]) for i in range(n)]
        finally:
            _n.lib.mssim_values_clear(array, n)

    def __getitem__(self, column: str) -> list:
        return self.column(column)

    def __contains__(self, column: str) -> bool:
        return column in self.columns

    def to_arrow(self):
        """The result table as a pyarrow.Table (requires pyarrow)."""
        import pyarrow as pa

        stream = _n.ArrowArrayStream()
        _check(_n.lib.mssim_table_arrow(self._handle, ctypes.byref(stream)))
        return pa.RecordBatchReader._import_from_c(ctypes.addressof(stream)).read_all()

    def to_pandas(self):
        """The result table as a pandas.DataFrame (requires pyarrow and pandas)."""
        return self.to_arrow().to_pandas()

    # -- metadata ---------------------------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        """Calculation metadata (parameters, sources, created_at, ...)."""
        pointer = ctypes.c_void_p()
        _check(_n.lib.mssim_get_metadata_json(self._handle, ctypes.byref(pointer)))
        return json.loads(_take_string(pointer))

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        _check(_n.lib.mssim_set_metadata_json(self._handle, _encode(json.dumps(value))))

    # -- matched data -----------------------------------------------------------------------
    @property
    def has_matched_data(self) -> bool:
        return bool(_n.lib.mssim_has_matched_data(self._handle))

    @property
    def matched_datasets(self) -> tuple["ms.Dataset", "ms.Dataset"]:
        """The embedded spectra of both sides (only those the rows refer to)."""
        if self._matched is None:
            sides = []
            for side in (1, 2):
                pointer, size = ctypes.c_void_p(), ctypes.c_int64()
                _check(_n.lib.mssim_matched_dataset(self._handle, side, ctypes.byref(pointer),
                                                    ctypes.byref(size)))
                try:
                    sides.append(ms.from_bytes(ctypes.string_at(pointer.value, size.value)))
                finally:
                    _n.lib.mssim_bytes_free(pointer)
            self._matched = tuple(sides)
        return self._matched

    def source_indices(self, side: int) -> np.ndarray:
        """For each embedded spectrum of ``side`` (1 or 2), its position in the
        original input dataset."""
        out = np.empty(_n.lib.mssim_matched_size(self._handle, side), dtype=np.int64)
        _check(_n.lib.mssim_source_indices(self._handle, side, _i64_ptr(out), len(out)))
        return out

    def match_records(self, row: int) -> tuple["ms.Record", "ms.Record"]:
        """The two spectra compared in ``row``."""
        d1, d2 = ctypes.c_int64(), ctypes.c_int64()
        _check(_n.lib.mssim_match_positions(self._handle, row, ctypes.byref(d1), ctypes.byref(d2)))
        first, second = self.matched_datasets
        return first[d1.value], second[d2.value]

    # -- selection --------------------------------------------------------------------------
    def filter(self, mask: Sequence[bool]) -> "SimilarityDataset":
        """Rows where ``mask`` is true."""
        array = np.ascontiguousarray(np.asarray(mask, dtype=bool).astype(np.uint8))
        return _new_result(lambda out: _n.lib.mssim_filter(
            self._handle, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(array), out))

    def sort_by(self, columns: str | Sequence[str],
                ascending: bool | Sequence[bool] = True) -> "SimilarityDataset":
        """Stable sort; missing values last."""
        names = [columns] if isinstance(columns, str) else list(columns)
        orders = [ascending] * len(names) if isinstance(ascending, bool) else list(ascending)
        if len(orders) != len(names):
            raise InvalidArgumentError("ascending must have one entry per column")
        c_names = (ctypes.c_char_p * len(names))(*[_encode(n) for n in names])
        c_orders = (ctypes.c_int32 * len(names))(*[int(bool(o)) for o in orders])
        return _new_result(lambda out: _n.lib.mssim_sort_by(
            self._handle, c_names, c_orders, len(names), out))

    # -- statistics -------------------------------------------------------------------------
    def describe_scores(self) -> dict[str, float]:
        """count, mean, q1, median, q3, min, max of the score column."""
        out = (ctypes.c_double * 7)()
        _check(_n.lib.mssim_describe_scores(self._handle, out))
        keys = ("count", "mean", "q1", "median", "q3", "min", "max")
        summary: dict[str, float] = dict(zip(keys, out))
        summary["count"] = int(summary["count"])
        return summary

    def histogram(self, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """(counts, edges) of the scores over [0, 1]."""
        if not 1 <= bins <= 200:
            raise InvalidArgumentError("bins must be between 1 and 200")
        counts = np.empty(bins, dtype=np.int64)
        edges = np.empty(bins + 1, dtype=np.float64)
        _check(_n.lib.mssim_histogram(self._handle, bins, _i64_ptr(counts),
                                      edges.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        return counts, edges

    # -- files ------------------------------------------------------------------------------
    def save(self, path: str | os.PathLike) -> None:
        """Write an .mssim file (embedded spectra are reduced to those still referenced)."""
        _check(_n.lib.mssim_save(self._handle, _encode(path)))

    def export(self, path: str | os.PathLike) -> None:
        """Write the table only, as .parquet, .csv, or .tsv."""
        _check(_n.lib.mssim_export_table(self._handle, _encode(path)))
