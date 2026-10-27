# MSSIM file format (`.mssim`)

Status: **schema version 2**. This document is the normative definition of the
format; any implementation that follows it can exchange files with any other.
The key words MUST, SHOULD, and MAY are used as in RFC 2119.

An MSSIM file stores one **similarity result**:

* a **result table** — one row per compared spectrum pair (which spectra, and their score),
* **calculation metadata** — how the scores were computed and from which inputs,
* optionally, the **matched spectra** themselves — so the file can be opened
  without the original input files ("embedded" mode).

The file is an [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file. The table
is a complete [Apache Parquet](https://parquet.apache.org/) file stored as a
byte array, and each set of matched spectra is a complete
[MSDS file](https://github.com/MotohiroOGAWA/msentity/blob/main/docs/format/msds.md)
stored the same way. Nothing in the format is language-specific.

---

## 1. A complete example

[`docs/examples/example.msp`](../examples/example.msp) holds two spectra
(the same file as in msentity):

```text
NAME: MassSpecID0000001            ← spectrum 0: 4 peaks
...
mz intensity formula
91.0542	245	C7H7+
125.0233	1000	C6H5O3+
185.0961	349
246.1125	735

NAME: MassSpecID0000002            ← spectrum 1: 3 peaks
...
mz intensity formula
91.0542	99	C7H7+
125.0233	281
246.1125	1000
```

Searching the file against itself and keeping every pair with a score of at
least 0.7:

```console
$ mssim-tool search example.msp example.msp example.mssim --threshold 0.7
4 pairs written to example.mssim
```

produces [`docs/examples/example.mssim`](../examples/example.mssim). This is its
complete content; every value below was read from that file.

```text
example.mssim
  @format         = "msentity.similarity"      ← identifies the file type
  @schema_version = 2                          ← this document
  @data_mode      = "embedded"                 ← matched spectra are included
│
├── table.parquet       uint8[1775]            ← a Parquet file: the result table (A)
│
├── metadata.json       string (scalar)        ← calculation metadata as JSON (B)
│
└── matched_data/                              ← the compared spectra (group)
      │
      ├── 1/                                   ← side 1: the query spectra
      │     ├── dataset.msds    uint8[16976]   ← an .msds file (gzip-compressed dataset)
      │     └── source_indices  int64[2] = [0, 1]
      │                                         ← row k of dataset.msds was spectrum
      │                                           source_indices[k] of the query input
      │
      └── 2/                                   ← side 2: the reference spectra
            ├── dataset.msds    uint8[16976]
            └── source_indices  int64[2] = [0, 1]
```

### A. The result table (`table.parquet`)

```text
 index1  index2  cosine_similarity  data_index1  data_index2
      0       0  1.0                          0            0
      0       1  0.7597143650054932           0            1
      1       0  0.7597143650054932           1            0
      1       1  1.0                          1            1
```

| Column | Type | Meaning |
|---|---|---|
| `index1` | int64 | position (0-based) of the spectrum in the **first / query** input dataset |
| `index2` | int64 | position of the spectrum in the **second / reference** input dataset |
| `cosine_similarity` | float32 | the score, in [0, 1]. The column has this name for every method (the method is in the metadata) |
| `data_index1` | int64 | row of `matched_data/1/dataset.msds` holding spectrum `index1` (embedded mode only) |
| `data_index2` | int64 | row of `matched_data/2/dataset.msds` holding spectrum `index2` (embedded mode only) |

Row 1 says: query spectrum 0 (`MassSpecID0000001`) and reference spectrum 1
(`MassSpecID0000002`) have a cosine similarity of 0.7597. Their peaks are in
row 0 of side 1 and row 1 of side 2 of `matched_data`. How the score is computed
is described in [../similarity.md](../similarity.md).

Results of a **by-key** calculation start with the key columns. With key
`SpecID` on both sides there is one key column:

```text
 SpecID  index1  index2  cosine_similarity
 SP1          0       1  1.0
 SP2          1       0  1.0
```

With different key names (for example `ID` and `Key`) there are two, `ID` and
`Key`, in that order. Key columns keep the type they had in the input.

### B. The metadata (`metadata.json`)

```jsonc
{
  "created_at": "2026-10-27T07:47:15.039020+00:00",  // when the result was calculated (UTC)
  "mode": "library_search",                // "library_search" or "by_key"
  "algorithm": "binned_cosine_similarity_library_search",
                                           // binned_<method>_similarity_<mode name>
  "parameters": {
    "threshold": 0.7,                      // library search: minimum score kept
    "method": "cosine",                    // "cosine" or "reverse_cosine"
    "bin_width": 0.01,                     // m/z bin width
    "intensity_exponent": 1.0              // intensities were raised to this power
  },                                       // (by_key: "key1" and "key2" instead of "threshold")
  "matching": "all_pairs_at_or_above_threshold",
                                           // by_key: "unique_intersection"
  "sources": [                             // the two inputs, query/first then reference/second
    {
      "role": "query",                     // "query", "reference", "first", or "second"
      "path": "example.msp",               // as given by the caller; may be ""
      "n_rows": 2,                         // number of spectra in that input
      "description": "",                   // the input dataset's description, attributes, tags
      "attributes": {},
      "tags": []
    },
    { "role": "reference", "path": "example.msp", "n_rows": 2,
      "description": "", "attributes": {}, "tags": [] }
  ],
  "row_count": 4,                          // rows in the table when it was calculated
  "candidate_pair_count": 4,               // library search: n_query × n_reference
  "index_basis": "zero-based input dataset position",
  "input_scope": "entire in-memory datasets (including unsaved edits)",
  "matched_data": "embedded"               // or "lightweight"
}
```

After filtering a result, `row_count` is the new number of rows and
`source_row_count` the number before filtering:

```jsonc
  "row_count": 2,
  "source_row_count": 4
```

The metadata is free to extend: readers MUST accept unknown keys, and tools
MAY add their own.

### Lightweight mode

Without matched spectra (`--lightweight` for `search`; the default for
`by-key`), the file has `@data_mode = "lightweight"`, no `matched_data` group,
and no `data_index1` / `data_index2` columns. `index1` / `index2` then refer to
the original input files, which must be kept to look the spectra up.

---

## 2. Rules

### File

1. The file MUST be HDF5 and SHOULD use the extension `.mssim`.
2. The root group MUST carry the attributes
   `format = "msentity.similarity"` (string) and `schema_version = 2` (integer).
   It SHOULD carry `data_mode` = `"embedded"` or `"lightweight"`.
3. Readers MUST reject files whose `format` differs and MUST reject schema
   versions they do not know. Readers SHOULD also accept schema version 1,
   which is version 2 without `matched_data`.

### Table

4. `table.parquet` MUST be a one-dimensional `uint8` dataset holding a
   complete Parquet file.
5. The table MUST have the columns `index1`, `index2` (integers), and
   `cosine_similarity` (numbers). Writers SHOULD store the indices as int64 and
   scores as float32.
6. Every score MUST be finite and in [0, 1].
7. `index1` / `index2` are 0-based positions in the input datasets as they were
   at calculation time (after any view selection, before saving).
8. Other columns MAY be present (key columns, user-added columns); readers
   MUST keep them.

### Metadata

9. `metadata.json` MUST be a scalar variable-length UTF-8 string dataset whose
   content is a JSON **object**. The keys listed in §1 B SHOULD be written.

### Matched data (embedded mode)

10. When present, `matched_data` MUST contain the groups `1` and `2`, each with
    * `dataset.msds` — a one-dimensional `uint8` dataset holding a complete MSDS
      file (it MAY be chunked and gzip-compressed);
    * `source_indices` — an int64 array with one entry per spectrum of that
      dataset: the spectrum's position in the original input.
11. The table MUST then have `data_index1` and `data_index2`, and every value
    MUST be a valid row of the corresponding `dataset.msds`.
12. Writers SHOULD store only the spectra the table refers to, in order of first
    appearance in `data_index1` / `data_index2`. (Saving a filtered result
    therefore shrinks `matched_data`; the reference implementation renumbers
    `data_index1/2` accordingly.)
13. For every row `r` and side `s`, `source_indices[data_index<s>[r]]` equals
    `index<s>[r]`.

### Writing

14. Writers SHOULD write to a temporary file and rename it over the target, so a
    failed write never leaves a partial file.

---

## 3. Reading a file without this library

Any HDF5 + Parquet reader works. In Python with `h5py` and `pyarrow`:

```python
import io, json
import h5py, pyarrow.parquet as pq

with h5py.File("example.mssim", "r") as f:
    assert f.attrs["format"] == "msentity.similarity"
    table = pq.read_table(io.BytesIO(f["table.parquet"][()].tobytes()))
    metadata = json.loads(f["metadata.json"][()])
    query_msds = f["matched_data/1/dataset.msds"][()].tobytes()   # bytes of an .msds file
    query_sources = f["matched_data/1/source_indices"][()]        # array([0, 1])
```
