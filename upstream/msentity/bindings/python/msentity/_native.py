"""ctypes declarations for the msentity C API (include/msentity/c/msentity_c.h)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# ---- locating the shared library -------------------------------------------------

_LIBRARY_NAMES = {
    "win32": ["msentity_c.dll"],
    "darwin": ["libmsentity_c.dylib"],
}.get(sys.platform, ["libmsentity_c.so"])


def _candidates() -> list[Path]:
    """Places searched for the C library, in order."""
    paths: list[Path] = []
    explicit = os.environ.get("MSENTITY_C_LIBRARY")
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
        "msentity C library not found; build it (cmake --build build) or set "
        "MSENTITY_C_LIBRARY. Tried: " + ", ".join(tried)
    )


lib = _load()

# ---- types -----------------------------------------------------------------------------

TYPE_NULL, TYPE_BOOL, TYPE_INT, TYPE_FLOAT, TYPE_STRING = range(5)


class Value(ctypes.Structure):
    """struct msentity_value"""

    _fields_ = [
        ("type", ctypes.c_int32),
        ("bool_value", ctypes.c_int32),
        ("int_value", ctypes.c_int64),
        ("float_value", ctypes.c_double),
        ("string_value", ctypes.c_void_p),
    ]


class ArrowArrayStream(ctypes.Structure):
    """struct ArrowArrayStream (Arrow C stream interface); filled by the library."""

    _fields_ = [
        ("get_schema", ctypes.c_void_p),
        ("get_next", ctypes.c_void_p),
        ("get_last_error", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


Handle = ctypes.c_void_p
HandlePtr = ctypes.POINTER(ctypes.c_void_p)
i32, i64, f64, cstr = ctypes.c_int32, ctypes.c_int64, ctypes.c_double, ctypes.c_char_p
ValuePtr = ctypes.POINTER(Value)
StrOut = ctypes.POINTER(ctypes.c_void_p)


def _declare(name: str, restype, *argtypes) -> None:
    function = getattr(lib, name)
    function.restype = restype
    function.argtypes = list(argtypes)


_declare("msentity_version", cstr)
_declare("msentity_last_error", cstr)
_declare("msentity_string_free", None, ctypes.c_void_p)
_declare("msentity_values_clear", None, ValuePtr, i64)

_declare("msentity_load", i32, cstr, cstr, cstr, i32, HandlePtr)
_declare("msentity_read_text", i32, cstr, cstr, i32, HandlePtr)
_declare("msentity_from_peaks", i32, ctypes.POINTER(f64), i64, ctypes.POINTER(i64), i64, HandlePtr)
_declare("msentity_concat", i32, ctypes.POINTER(Handle), i64, HandlePtr)
_declare("msentity_free", None, Handle)
_declare("msentity_save", i32, Handle, cstr, cstr, i32, i32)

_declare("msentity_size", i64, Handle)
_declare("msentity_n_peaks_total", i64, Handle)
_declare("msentity_n_columns", i64, Handle)
_declare("msentity_column_name", i32, Handle, i64, StrOut)
_declare("msentity_has_column", i32, Handle, cstr)
_declare("msentity_set_columns", i32, Handle, ctypes.POINTER(cstr), i64)

_declare("msentity_get_value", i32, Handle, i64, cstr, ValuePtr)
_declare("msentity_get_column", i32, Handle, cstr, ValuePtr, i64)
_declare("msentity_set_column", i32, Handle, cstr, ValuePtr, i64)
_declare("msentity_set_column_scalar", i32, Handle, cstr, ValuePtr)
_declare("msentity_set_value", i32, Handle, i64, cstr, ValuePtr)

_declare("msentity_select", i32, Handle, ctypes.POINTER(i64), i64, HandlePtr)
_declare("msentity_slice", i32, Handle, i64, i64, HandlePtr)
_declare("msentity_filter", i32, Handle, ctypes.POINTER(ctypes.c_uint8), i64, HandlePtr)
_declare("msentity_sort_by", i32, Handle, cstr, i32, HandlePtr)
_declare("msentity_copy", i32, Handle, HandlePtr)
_declare("msentity_full_view", i32, Handle, i32, HandlePtr)

_declare("msentity_offsets", i32, Handle, ctypes.POINTER(i64), i64)
_declare("msentity_peaks", i32, Handle, ctypes.POINTER(f64), ctypes.POINTER(f64), i64)
_declare("msentity_spectrum_size", i32, Handle, i64, ctypes.POINTER(i64))
_declare("msentity_spectrum_peaks", i32, Handle, i64, ctypes.POINTER(f64), ctypes.POINTER(f64), i64)
_declare("msentity_normalize", i32, Handle, f64)
_declare("msentity_sort_peaks", i32, Handle, i32, i32)

_declare("msentity_n_peak_columns", i64, Handle)
_declare("msentity_peak_column_name", i32, Handle, i64, StrOut)
_declare("msentity_get_peak_column", i32, Handle, cstr, ValuePtr, i64)
_declare("msentity_set_peak_column", i32, Handle, cstr, ValuePtr, i64)

_declare("msentity_get_description", i32, Handle, StrOut)
_declare("msentity_set_description", i32, Handle, cstr)
_declare("msentity_n_attributes", i64, Handle)
_declare("msentity_attribute", i32, Handle, i64, StrOut, StrOut)
_declare("msentity_set_attribute", i32, Handle, cstr, cstr)
_declare("msentity_remove_attribute", i32, Handle, cstr, ctypes.POINTER(i32))
_declare("msentity_n_tags", i64, Handle)
_declare("msentity_tag", i32, Handle, i64, StrOut)
_declare("msentity_add_tag", i32, Handle, cstr, ctypes.POINTER(i32))
_declare("msentity_remove_tag", i32, Handle, cstr, ctypes.POINTER(i32))
_declare("msentity_set_spec_id", i32, Handle, cstr, cstr, i32, i64, ctypes.POINTER(i32))

_declare("msentity_to_msds_bytes", i32, Handle, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(i64))
_declare("msentity_from_msds_bytes", i32, ctypes.c_char_p, i64, HandlePtr)
_declare("msentity_bytes_free", None, ctypes.c_void_p)

_declare("msentity_metadata_arrow", i32, Handle, ctypes.POINTER(ArrowArrayStream))
_declare("msentity_peak_metadata_arrow", i32, Handle, ctypes.POINTER(ArrowArrayStream))
