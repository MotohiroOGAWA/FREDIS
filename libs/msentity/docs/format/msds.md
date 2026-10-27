# MSDS file format (`.msds`)

Status: **version 1**. This document is the normative definition of the
format; any implementation that follows it can exchange files with any other.
The key words MUST, SHOULD, and MAY are used as in RFC 2119.

An MSDS file stores one mass-spectrometry **dataset**:

* a **spectrum metadata table** — one row per spectrum (name, precursor m/z, …),
* the **peaks** of every spectrum — (m/z, intensity) pairs,
* an optional **peak metadata table** — one row per peak (fragment formula, …),
* **dataset-level metadata** — a description, key/value attributes, tags.

The file is an [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file. Peaks are
plain HDF5 numeric arrays; the two tables are complete
[Apache Parquet](https://parquet.apache.org/) files stored as byte arrays
inside it.

---

## 1. A complete example

[`docs/examples/example.msp`](../examples/example.msp) holds two spectra; the
first peaks carry a fragment formula:

```text
NAME: MassSpecID0000001
PRECURSORMZ: 288.1225
PRECURSORTYPE: [M+H]1+
FORMULA: C16H17NO4
COLLISIONENERGY: 30.0
Num Peaks: 4
mz intensity formula
91.0542	245	C7H7+
125.0233	1000	C6H5O3+
185.0961	349
246.1125	735

NAME: MassSpecID0000002
...
Num Peaks: 3
mz intensity formula
91.0542	99	C7H7+
125.0233	281
246.1125	1000
```

Converting it (`msentity-tool convert example.msp example.msds`) produces
[`docs/examples/example.msds`](../examples/example.msds). This is its complete
content; every value below was read from that file.

```text
example.msds
│
├── metadata/                                  ← dataset-level metadata (group)
│     @description     = ""                    ← free text
│     @attributes_json = "{}"                  ← key/value pairs as a JSON object
│     @tags_json       = "[]"                  ← labels as a JSON array
│
└── dataset_0/                                 ← the dataset (group)
      @columns_json = "[\"Name\", \"PrecursorMZ\", \"AdductType\", \"Formula\",
                        \"CollisionEnergy\", \"NumPeaks\"]"
                                               ← visible spectrum columns, in display order
      @spectrum_metadata_parquet__chunked   = FALSE   ← the table is stored in one piece
      @spectrum_metadata_parquet__num_parts = 1
      │
      ├── spectrum_metadata_parquet            ← uint8[1965]: a Parquet file (table A below)
      │     @bytes_format = "uint8_1d"
      │     @nbytes       = 1965
      │
      └── peaks/                               ← all peaks (group)
            @metadata_parquet__chunked   = FALSE
            @metadata_parquet__num_parts = 1
            │
            ├── data             float64[7, 2] ← one row per peak: (m/z, intensity)
            ├── offsets          int64[3]      = [0, 4, 7]
            ├── index            int64[2]      = [0, 1]
            ├── metadata_parquet uint8[479]    ← a Parquet file (table B below)
            │     @bytes_format = "uint8_1d"
            │     @nbytes       = 479
            └── metadata_columns string[1]     = ["formula"]
```

`@name` marks an HDF5 attribute; everything else is a group or a dataset.

### Peaks: `data` and `offsets`

The peaks of *all* spectra are stacked into one array; `offsets` says where
each spectrum starts and ends.

```text
            data[row]                    offsets = [0, 4, 7]
 row   m/z         intensity
  0    91.0542     0.245      ┐
  1   125.0233     1.0        │ spectrum 0 = rows offsets[0]..offsets[1]-1 = 0..3
  2   185.0961     0.349      │
  3   246.1125     0.735      ┘
  4    91.0542     0.099      ┐
  5   125.0233     0.281      │ spectrum 1 = rows offsets[1]..offsets[2]-1 = 4..6
  6   246.1125     1.0        ┘
```

Intensities were scaled while reading the MSP so that each spectrum's highest
peak is 1.0 (245/1000 = 0.245, …).

### Peaks: `index`

`index = [0, 1]` lists which stored spectra form the dataset and in which
order. A file written from a filtered or reordered dataset in *raw* mode
(§5) could contain, for example, `index = [1]` (only the second spectrum) or
`index = [1, 0]` (both, reversed) while `data` and `offsets` stay as above.

### Table A: `spectrum_metadata_parquet` (one row per spectrum)

| row | Name | PrecursorMZ | AdductType | Formula | CollisionEnergy | NumPeaks |
|---|---|---|---|---|---|---|
| 0 | MassSpecID0000001 | 288.1225 | [M+H]+ | C16H17NO4 | 30.0 | 4 |
| 1 | MassSpecID0000002 | 288.1225 | [M+H]+ | C16H17NO4 | 20.0 | 3 |

All six columns have type `string`, because MSP values are text. Header
names were canonicalized (`PRECURSORTYPE` → `AdductType`, `Num Peaks` →
`NumPeaks`) and `[M+H]1+` became `[M+H]+`; see
[text-formats.md](text-formats.md).

### Table B: `peaks/metadata_parquet` (one row per peak)

| row (= `data` row) | formula |
|---|---|
| 0 | C7H7+ |
| 1 | C6H5O3+ |
| 2 | *(empty)* |
| 3 | *(empty)* |
| 4 | C7H7+ |
| 5 | *(empty)* |
| 6 | *(empty)* |

---

## 2. HDF5 types

| What | Writers MUST use | Readers MUST accept |
|---|---|---|
| string attribute, string dataset element | variable-length string with UTF-8 character set; attributes use a scalar dataspace | variable- or fixed-length strings, ASCII or UTF-8 (fixed-length: cut at the first NUL; space-padded: trailing spaces removed) |
| bool attribute (`…__chunked`) | enumeration on `int8` with members `FALSE = 0`, `TRUE = 1` | that enumeration, or any integer type (non-zero = true) |
| integer attribute (`…__num_parts`, `@nbytes`) | `H5T_STD_I64LE`, scalar | any integer type |
| `peaks/data` | `H5T_IEEE_F64LE`, shape `(n_peaks, 2)` | any floating-point type |
| `peaks/offsets`, `peaks/index` | `H5T_STD_I64LE`, one-dimensional | any integer type |
| Parquet byte array | one-dimensional `H5T_STD_U8LE` | one-dimensional 8-bit unsigned integers; opaque types as raw bytes |

`data`, `offsets`, and `index` are written contiguous and uncompressed.
Readers MUST accept any storage layout and any filter their HDF5 library
supports.

---

## 3. Data model rules

Notation: `n_spectra` = number of rows of table A = `len(offsets) − 1`;
`n_peaks` = number of rows of `data`.

### 3.1 `peaks/data` and `peaks/offsets`

| Rule | Example above |
|---|---|
| `data` has shape `(n_peaks, 2)`, row-major: column 0 = m/z, column 1 = intensity | `(7, 2)` |
| `offsets` has `n_spectra + 1` values and never decreases | `[0, 4, 7]` |
| Stored spectrum `i` owns the rows `offsets[i] ≤ r < offsets[i + 1]` | spectrum 1 → rows 4, 5, 6 |
| `offsets[0]` SHOULD be 0 | `0` |
| `offsets[n_spectra]` MUST NOT exceed `n_peaks` | `7 ≤ 7` |
| An empty spectrum has two equal offsets | `[0, 4, 4, 7]` would add an empty spectrum |

### 3.2 `peaks/index` — the view

| Rule | |
|---|---|
| Every value is a stored-spectrum number in `[0, n_spectra)` | |
| Its order is the order of spectra in the dataset; values MAY repeat | `[1, 0]` reverses, `[1, 1]` repeats |
| A materialized dataset (the default when writing) has `index = [0, 1, …, n_spectra − 1]` | `[0, 1]` |

### 3.3 Spectrum metadata (table A, `@columns_json`)

| Rule | |
|---|---|
| Table A has exactly `n_spectra` rows | 2 |
| Row `i` describes **stored** spectrum `i` (not position `i` of `index`) | |
| `@columns_json` lists the visible columns in display order; each MUST exist in table A | the six column names |
| Columns not listed are kept in the file but hidden | |
| Missing or empty `@columns_json` = all columns visible in table order | |

### 3.4 Peak metadata (table B, `metadata_columns`) — optional

| Rule | |
|---|---|
| Table B has exactly `n_peaks` rows; row `p` describes `data` row `p` | 7 rows |
| `metadata_columns` lists the visible peak columns | `["formula"]` |
| `mz` and `intensity` are reserved names and MUST NOT be peak columns | |
| If `metadata_columns` is missing, all columns of table B are visible | |
| If there is no peak metadata, neither table B, its attributes, nor `metadata_columns` exist | |

### 3.5 Dataset-level metadata (`/metadata`)

| Attribute | Content | Example |
|---|---|---|
| `@description` | free text | `"MoNA positive mode, ClassyFire annotated"` |
| `@attributes_json` | JSON object, values are strings; key order is kept | `{"source": "MoNA-export-LC-MS-MS_Positive_Mode.msp"}` |
| `@tags_json` | JSON array of distinct strings | `["ClassyFire"]` |

A missing group or attribute means an empty value. Readers MUST accept any
valid JSON. Writers SHOULD use this canonical form, so equal metadata gives
byte-identical files:

* `", "` between items and `": "` between key and value, no other whitespace;
* `"` → `\"`, `\` → `\\`, newline/CR/tab/backspace/form feed → `\n` `\r` `\t` `\b` `\f`;
* every other character outside U+0020–U+007E → `\uXXXX` (lower-case hex;
  characters above U+FFFF as two `\uXXXX` surrogates).

| Value | Canonical JSON |
|---|---|
| attributes `source = MoNA`, `méthode = LC` | `{"source": "MoNA", "méthode": "LC"}` |
| tags `ClassyFire`, `タグ` | `["ClassyFire", "タグ"]` |

### 3.6 Several datasets in one file

Appending (§5) adds groups `dataset_1`, `dataset_2`, …. A reader MUST load all
groups named `dataset_<n>` (`<n>` a non-negative decimal integer) in
ascending order of `<n>` and join them into one dataset:

| Aspect | Result |
|---|---|
| spectra | those of `dataset_0`, then `dataset_1`, … |
| visible columns | union, in order of first appearance |
| cells of a column a group does not have | missing |
| dataset-level metadata | the single `/metadata` group |

A group name starting with `dataset_` with any other suffix makes the file
invalid.

---

## 4. Tables inside the file (Parquet)

### 4.1 One piece or several parts

A table stored under the name `<name>` (`spectrum_metadata_parquet` in
`dataset_<n>/`, `metadata_parquet` in `peaks/`) is stored in one of two ways:

| `@<name>__chunked` | Datasets | Table |
|---|---|---|
| `FALSE` or missing | `<name>` | the Parquet file in `<name>` (MUST be smaller than 2 147 483 647 bytes) |
| `TRUE` | `<name>__part_000`, `<name>__part_001`, … (`@<name>__num_parts` of them) | the rows of all parts, in part order |

Example of a split table:

```text
dataset_0/
  @spectrum_metadata_parquet__chunked   = TRUE
  @spectrum_metadata_parquet__num_parts = 3
  spectrum_metadata_parquet__part_000   ← rows 0 … 1 999 999
  spectrum_metadata_parquet__part_001   ← rows 2 000 000 … 3 999 999
  spectrum_metadata_parquet__part_002   ← rows 4 000 000 … end
```

When writers split: if the encoded table — the larger of the Parquet file
size and the sum of its row groups' uncompressed sizes — exceeds the part
limit (default 1 000 000 000 bytes), they start with
`min(2 000 000, number of rows)` rows per part and halve that number until
every part fits. The part index has at least three digits.

### 4.2 Byte-array storage details (informative for readers)

Each Parquet byte array carries `@bytes_format = "uint8_1d"` and
`@nbytes = <length>` and is written gzip-compressed (level 4) in chunks of
`c` bytes, where for a byte length `L`:

```text
target = 16384 × 2^(log10(L / 1048576)), limited to 8192 … 1048576
c = L
repeat:
    stop if (c < target or |c − target| / target < 0.5) and c < 1048576
    stop if c = 1
    c = ceil(c / 2)
```

`L = 1965` (table A above) gives `c = 1965`; `L = 28 647 517` gives
`c = 55 953`. Readers MUST NOT depend on these attributes or settings.

### 4.3 Column types and cell values

Each cell is one of five values: **missing**, **boolean**, **integer**
(64-bit), **float** (double), **text** (UTF-8).

| Parquet / Arrow column type | Cell value | Typical source |
|---|---|---|
| `string`, `large_string`, `string_view` | text | MSP/MGF values, names, SMILES |
| `int8` … `int64`, `uint8` … `uint32` | integer | counts, IDs |
| `uint64` | integer (float if beyond int64) | |
| `float`, `double` | float | computed values |
| `bool` | boolean | flags |
| `null` | missing | column whose cells are all missing |
| dictionary-encoded | the decoded value | categorical columns |
| any other type | implementation-defined (reference implementation: text) | |

* A null cell is missing; a float NaN is also treated as missing.
* Column names are UTF-8 and unique within a table.
* Writers SHOULD use only `string`, `int64`, `double`, `bool`, one type per column.
* A table without columns cannot record its row count in Parquet. Readers
  MUST treat a column-less table A as having `n_spectra` rows and a
  column-less table B as having `n_peaks` rows. (Example: an MGF file whose
  records have peaks but no `KEY=value` lines.)

### 4.4 Parquet schema metadata

Readers MUST ignore key/value metadata in the Parquet schema, except:
if the key `pandas` exists and its JSON value has an `index_columns` array,
the columns named by its string entries are row labels, not data, and MUST
be dropped.

```json
{"index_columns": ["__index_level_0__"], ...}   → drop column "__index_level_0__"
{"index_columns": [{"kind": "range", ...}], ...} → nothing to drop
```

A writer that rewrites a table it has read MUST drop such metadata unless the
row labels are still valid. The reference implementation writes none.

---

## 5. Writing

| Mode | What is written |
|---|---|
| overwrite (default) | truncate the file; write `/metadata` and `dataset_0` |
| append | open or create the file; rewrite `/metadata`; add `dataset_<largest n + 1>` |
| materialized (default) | only visible spectra and columns: table A has just those rows/columns, table B just the visible peak columns, `index = [0, 1, …]` |
| raw | complete tables and peak arrays, the current `index`, `@columns_json`, and `metadata_columns` as they are |

### 5.1 What a file preserves

A dataset in memory is a *view*: which stored spectra, in which order
(**index**), and which columns, in which order (**columns**). Both are stored:

| In memory | In the file |
|---|---|
| view index (selected spectra and their order) | `peaks/index` |
| visible spectrum columns and their order | `@columns_json` |
| all spectrum columns, names and types | table A (Parquet schema) |
| visible peak columns | `peaks/metadata_columns` |
| hidden spectra and hidden columns | kept in raw mode only |

One index serves both tables and the peaks: view position `k` is stored
spectrum `index[k]`, whose metadata is row `index[k]` of table A and whose
peaks are rows `offsets[index[k]]` … `offsets[index[k] + 1] − 1` of `data`.

Measured example: the two spectra of `example.msp`, reversed and reduced to
two columns (`d.select({1, 0})` then `set_columns({"Name", "CollisionEnergy"})`):

| | raw (`save_view = false`) | materialized (default) |
|---|---|---|
| `peaks/index` | **`[1, 0]`** | `[0, 1]` |
| `peaks/offsets` | `[0, 4, 7]` (unchanged) | `[0, 3, 7]` (spectrum 1's 3 peaks first) |
| `@columns_json` | `["Name", "CollisionEnergy"]` | `["Name", "CollisionEnergy"]` |
| table A columns | all 6 | `Name`, `CollisionEnergy` |
| table A `Name` (stored order) | `MassSpecID0000001`, `MassSpecID0000002` | `MassSpecID0000002`, `MassSpecID0000001` |
| dataset after reading | `MassSpecID0000002`, `MassSpecID0000001`; 2 visible columns | the same |
| can the hidden data be restored? | yes (`reset_view(true)` → 2 spectra × 6 columns) | no |

### 5.2 What a file does not preserve

* **Row labels.** Rows of table A and table B are identified only by
  position (stored spectrum number, `data` row). Labels kept by other tools
  (for example a DataFrame index) are not part of the format (§4.4).
* **Original numbers after materializing.** A materialized file renumbers
  its spectra `0, 1, …`; which spectrum of the source dataset each one was is
  lost. Keep that information either by saving in raw mode or by storing it
  in a column (for example `SpecID`, see `processing::set_spec_id`) before
  saving.

---

## 6. Conformance

`tests/data` and `docs/examples` contain reference files. An implementation
in another language — whether a binding of this library or a native reader
built on its own HDF5 and Parquet libraries (Java: jHDF or the HDF5 Java
bindings with parquet-java / Arrow Java; C#: PureHDF or HDF.PInvoke with
Parquet.Net / Arrow C#) — should read these files to the same datasets and
write files this library reads back unchanged.
