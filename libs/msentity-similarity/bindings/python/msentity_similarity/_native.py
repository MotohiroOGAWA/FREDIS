"""ctypes declarations for the msentity-similarity C API
(include/msentity/similarity/c/msentity_similarity_c.h)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from msentity import _native as _ms

# ---- locating the shared library -------------------------------------------------

_LIBRARY_NAMES = {
    "win32": ["msentity_similarity_c.dll"],
    "darwin": ["libmsentity_similarity_c.dylib"],
}.get(sys.platform, ["libmsentity_similarity_c.so"])


def _candidates() -> list[Path]:
    """Places searched for the C library, in order."""
    paths: list[Path] = []
    explicit = os.environ.get("MSSIM_C_LIBRARY")
    if explicit:
        paths.append(Path(explicit))
    here = Path(__file__).resolve().parent
    for name in _LIBRARY_NAMES:
        paths.append(here / "lib" / name)          # bundled in a wheel
        paths.append(here.parents[2] / "build" / name)  # repository build directory
    return paths


def _load() -> ctypes.CDLL:
    tried = []
    for path in _candidates():
        if path.exists():
            return ctypes.CDLL(str(path))
        tried.append(str(path))
    for name in _LIBRARY_NAMES:                      # system library path
        try:
            return ctypes.CDLL(name)
        except OSError:
            tried.append(name)
    raise OSError(
        "msentity-similarity C library not found; build it (cmake --build build) or set "
        "MSSIM_C_LIBRARY. Tried: " + ", ".join(tried)
    )


lib = _load()

# ---- types -----------------------------------------------------------------------------

METHODS = {"cosine": 0, "reverse_cosine": 1}


class Options(ctypes.Structure):
    """struct mssim_options"""

    _fields_ = [
        ("method", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("bin_width", ctypes.c_double),
        ("intensity_exponent", ctypes.c_double),
    ]


Value = _ms.Value
ArrowArrayStream = _ms.ArrowArrayStream
Handle = ctypes.c_void_p
HandlePtr = ctypes.POINTER(ctypes.c_void_p)
i32, i64, f64, cstr = ctypes.c_int32, ctypes.c_int64, ctypes.c_double, ctypes.c_char_p
ValuePtr = ctypes.POINTER(Value)
StrOut = ctypes.POINTER(ctypes.c_void_p)
OptionsPtr = ctypes.POINTER(Options)
Bytes = ctypes.c_char_p
ProgressFn = ctypes.CFUNCTYPE(None, i64, i64, ctypes.c_void_p)


def _declare(name: str, restype, *argtypes) -> None:
    function = getattr(lib, name)
    function.restype = restype
    function.argtypes = list(argtypes)


_declare("mssim_version", cstr)
_declare("mssim_last_error", cstr)
_declare("mssim_string_free", None, ctypes.c_void_p)
_declare("mssim_bytes_free", None, ctypes.c_void_p)
_declare("mssim_values_clear", None, ValuePtr, i64)
_declare("mssim_default_options", None, OptionsPtr)

_declare("mssim_pair_scores", i32, Bytes, i64, ctypes.POINTER(i64), Bytes, i64, ctypes.POINTER(i64),
         i64, OptionsPtr, ctypes.POINTER(ctypes.c_float))
_declare("mssim_all_pairs", i32, Bytes, i64, Bytes, i64, OptionsPtr,
         ctypes.POINTER(ctypes.c_float), i64)
_declare("mssim_library_search", i32, Bytes, i64, Bytes, i64, f64, OptionsPtr, i32, cstr, cstr,
         ProgressFn, ctypes.c_void_p, HandlePtr)
_declare("mssim_similarity_by_key", i32, Bytes, i64, Bytes, i64, cstr, cstr, OptionsPtr, i32,
         cstr, cstr, HandlePtr)

_declare("mssim_load", i32, cstr, HandlePtr)
_declare("mssim_save", i32, Handle, cstr)
_declare("mssim_free", None, Handle)

_declare("mssim_size", i64, Handle)
_declare("mssim_n_columns", i64, Handle)
_declare("mssim_column_name", i32, Handle, i64, StrOut)
_declare("mssim_get_column", i32, Handle, cstr, ValuePtr, i64)
_declare("mssim_get_metadata_json", i32, Handle, StrOut)
_declare("mssim_set_metadata_json", i32, Handle, cstr)

_declare("mssim_has_matched_data", i32, Handle)
_declare("mssim_matched_dataset", i32, Handle, i32, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(i64))
_declare("mssim_matched_size", i64, Handle, i32)
_declare("mssim_source_indices", i32, Handle, i32, ctypes.POINTER(i64), i64)
_declare("mssim_match_positions", i32, Handle, i64, ctypes.POINTER(i64), ctypes.POINTER(i64))

_declare("mssim_filter", i32, Handle, ctypes.POINTER(ctypes.c_uint8), i64, HandlePtr)
_declare("mssim_sort_by", i32, Handle, ctypes.POINTER(cstr), ctypes.POINTER(i32), i64, HandlePtr)
_declare("mssim_describe_scores", i32, Handle, ctypes.POINTER(f64))
_declare("mssim_histogram", i32, Handle, i32, ctypes.POINTER(i64), ctypes.POINTER(f64))
_declare("mssim_export_table", i32, Handle, cstr)
_declare("mssim_table_arrow", i32, Handle, ctypes.POINTER(ArrowArrayStream))
