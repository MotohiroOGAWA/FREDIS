# Architecture

msentity is a C++ library that holds mass-spectrometry datasets in memory
and reads/writes them in the MSDS, MSP, MGF, TSV, and CSV formats. It is
meant to be the one implementation behind packages for several languages, so
a file written from any of them can be read from any other.

The library is deliberately limited to **data and files**. Analysis
(similarity scoring, peak-based filtering, …) belongs in separate modules
built on top of it.

---

## 1. Source layout

```text
include/msentity/                  public headers
├── msentity.hpp                   includes everything below
├── value.hpp                      Value (one metadata cell), KeyError, FormatError,
│                                  format_float / to_display_string
├── core/
│   ├── peak.hpp                   Peak: m/z, intensity, metadata of one peak
│   ├── peak_series.hpp            PeakSeries: all peaks (CSR layout) + view index
│   │                              Spectrum: the peaks of one spectrum
│   └── ms_dataset.hpp             MSDataset: metadata table + PeakSeries + columns,
│                                  description, attributes, tags
│                                  SpectrumRecord: one spectrum of a dataset
├── c/
│   └── msentity_c.h               C API for the language bindings (libmsentity_c)
├── io/
│   ├── msds.hpp                   read_msds / write_msds / read_msds_meta (+ in-memory)
│   ├── text_formats.hpp           read_/write_ msp, mgf, tsv, csv and their options
│   └── load.hpp                   load_ms_dataset: any format by file extension
└── processing/
    └── id.hpp                     set_spec_id
src/
├── core/  io/  processing/        implementations of the headers above
└── internal/                      private helpers (not installed)
    ├── table.*                    Arrow tables: cell access, row selection, concatenation
    ├── hdf5.*                     HDF5 C API wrapper (types as in the MSDS spec)
    ├── parquet.*                  Arrow table ⇄ Parquet bytes
    ├── json.*                     JSON attributes (canonical output)
    └── text.*                     strip/split, number parsing, line splitting
src/c/msentity_c.cpp               C API implementation
tools/msentity_tool.cpp            command-line tool (dump / convert / meta)
bindings/python, java, csharp      language bindings over the C API
tests/                             unit tests, reference data, conformance tests
docs/                              this documentation
```

---

## 2. Data model in memory

The in-memory structures mirror the [MSDS file layout](format/msds.md), so
reading and writing are mostly copies. With
[`docs/examples/example.msp`](examples/example.msp) loaded:

```text
MSDataset d
├── metadata_  ──► TableRef ──► arrow::Table  (2 rows: one per stored spectrum)
│                                ┌───┬───────────────────┬─────────────┬─────┬──────────┐
│                                │   │ Name              │ PrecursorMZ │ ... │ NumPeaks │
│                                │ 0 │ MassSpecID0000001 │ 288.1225    │     │ 4        │
│                                │ 1 │ MassSpecID0000002 │ 288.1225    │     │ 3        │
│                                └───┴───────────────────┴─────────────┴─────┴──────────┘
├── columns_      {"Name", "PrecursorMZ", "AdductType", "Formula", "CollisionEnergy", "NumPeaks"}
├── description_  ""          attributes_  {}          tags_  {}
└── peaks_ : PeakSeries
    ├── storage_ ──► PeakStorage
    │                ├── data      {91.0542, 0.245,  125.0233, 1.0,  185.0961, 0.349,
    │                │              246.1125, 0.735,  91.0542, 0.099,  125.0233, 0.281,
    │                │              246.1125, 1.0}            (m/z, intensity) × 7
    │                ├── offsets   {0, 4, 7}                  spectrum i = peaks offsets[i] … offsets[i+1]-1
    │                └── metadata  arrow::Table (7 rows)      formula: "C7H7+", "C6H5O3+", "", …
    ├── metadata_columns_  {"formula"}
    └── index_             {0, 1}                             the view: which spectra, in which order
```

### Views share storage

`d.select({1})` creates a new `MSDataset` whose `index_` is `{1}`; the
`TableRef` and `PeakStorage` are the same objects (shared through
`std::shared_ptr`):

```text
MSDataset d            MSDataset v = d.select({1})
  index_ {0, 1}          index_ {1}
  columns_ {...}         columns_ {...}           ← own copy of the column list
       │                      │
       └───────┬──────────────┘
               ▼
   TableRef / PeakStorage  (one copy of the metadata and peaks)
```

Consequences:

| Operation | Cost | Effect |
|---|---|---|
| `select`, `slice`, `filter`, `sort_by` | O(size of the view) | new view, no data copied |
| `set_column`, `set_value` through a view | rebuilds one column | visible to every view of the same data |
| `PeakSeries::normalize`, `sort_by_mz` (in place) | O(peaks of the view) | changes the shared peaks |
| `copy()` | copies visible rows, columns, peaks | independent dataset with `index_ = {0, 1, …}` |

### Values

A metadata cell is a `Value`:

| Alternative | Arrow column types that produce it |
|---|---|
| `std::monostate` (missing) | any null cell, float NaN, `null` type |
| `bool` | `bool` |
| `int64_t` | `int8` … `int64`, `uint8` … `uint32` (and `uint64` in range) |
| `double` | `float`, `double` |
| `std::string` (UTF-8) | `string`, `large_string`, `string_view`; other types as text |

`Value` is also the unit that language bindings will exchange.

---

## 3. Dependencies

| Library | Used for | In public headers? |
|---|---|---|
| Apache Arrow (core `libarrow`) | metadata tables | yes — `arrow::Table` |
| Apache Parquet | tables inside `.msds` | no |
| HDF5 (C + high-level) | the `.msds` container | no |
| nlohmann/json | JSON attributes | no (build only) |

Only core Arrow is used, not `arrow_compute`: row selection, concatenation,
and type unification are implemented in `src/internal/table.cpp` with Arrow
builders. This keeps the shared-library footprint of every future binding
small.

---

## 4. Errors

| Exception | Meaning |
|---|---|
| `std::invalid_argument` | invalid argument or invalid input data |
| `std::out_of_range` | spectrum or peak position out of range |
| `msentity::KeyError` (derives from `std::out_of_range`) | unknown column |
| `msentity::FormatError` (derives from `std::runtime_error`) | malformed or unreadable file |
| `std::runtime_error` | other I/O failures |

---

## 5. Language bindings

```text
 Python (ctypes + NumPy)   Java (Foreign Function & Memory API)   C# (P/Invoke)
          └──────────────────────────────┼───────────────────────────────┘
                                         ▼
               libmsentity_c   (C API, include/msentity/c/msentity_c.h)
                                         ▼
               libmsentity     (this C++ core, linked statically)
                                         ▼
               HDF5 · Arrow · Parquet
```

All three languages use the same C API rather than wrapping C++ classes
directly: there is one native library to build and ship, the behavior is
identical in every language, and no binding needs its own compiled glue
code (ctypes, `java.lang.foreign`, and P/Invoke are part of each language's
standard runtime).

| | C API rule | Python | Java | C# |
|---|---|---|---|---|
| handle | `msentity_dataset*` + `msentity_free` | `Dataset` (freed by `weakref.finalize`) | `Dataset` (`AutoCloseable`, `Cleaner`) | `Dataset` (`IDisposable`, `SafeHandle`) |
| errors | status code + `msentity_last_error()` | exception classes (`ValueError`, `KeyError`, …) | `MsEntityException` subclasses | `MsEntityException` subclasses |
| values | `msentity_value` struct | `None/bool/int/float/str` | `null/Boolean/Long/Double/String` | `null/bool/long/double/string` |
| arrays | copied into caller buffers | NumPy arrays | `long[]`, `double[]` | `long[]`, `double[]` |
| tables | Arrow C stream | `pyarrow.Table`, pandas | — | — |

`libmsentity_c` is built with hidden visibility and exports only the
`msentity_*` functions. Usage of every binding is described in
[bindings.md](bindings.md).

## 6. Roadmap

1. ✅ Data model, MSDS/MSP/MGF/TSV/CSV, specifications, tests.
2. ✅ C API: load/save, dataset access, Arrow stream export.
3. ✅ Python, Java, and C# bindings with tests (`ctest` runs all of them).
4. Binary packages: Python wheels, Maven JARs with natives, NuGet — each
   bundling `libmsentity_c` and its dependencies per platform.
5. Analysis modules on top of the core, each in its own namespace and
   library target.
