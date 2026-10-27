# C++ API guide

Every example on this page uses
[`docs/examples/example.msp`](examples/example.msp) (two spectra, 4 + 3 peaks,
and a `formula` value on some peaks). The comment after each line shows the actual
result.

```cpp
#include <msentity/msentity.hpp>   // everything below
using namespace msentity;          // types; file I/O is in msentity::io
```

---

## 1. Loading files

```cpp
MSDataset d = io::load_ms_dataset("example.msp");     // format chosen by extension
//   .msp → MSP   .mgf → MGF   .msds/.h5/.hdf5 → MSDS   .tsv → TSV   .csv → CSV

MSDataset m = io::load_ms_dataset("data.txt", std::string("msp"));   // explicit format
MSDataset s = io::load_ms_dataset("example.msp", std::nullopt,
                                  std::string("SP"));  // also add SpecID = SP1, SP2
```

Format-specific readers give access to all options:

```cpp
io::TextReadOptions opt;                     // MSP / MGF options (defaults shown)
opt.canonicalize_column = true;              // "PRECURSORTYPE" → "AdductType"
opt.canonicalize_adduct_type = true;         // "[M+H]1+" → "[M+H]+"
opt.normalize_intensity = true;              // highest peak of each spectrum → 1.0
opt.allow_duplicate_cols = false;            // duplicate key → record skipped
opt.error_log_level = io::ErrorLogLevel::None;   // Basic / Detail write a log file

io::TextReadResult r = io::read_msp_with_header_map("example.msp", opt);
r.dataset.size();                            // 2
r.header_map;                                // {"Name","NAME"}, {"PrecursorMZ","PRECURSORMZ"},
                                             // {"AdductType","PRECURSORTYPE"}, …,
                                             // {"NumPeaks","Num Peaks"}  (canonical ← original)

MSDataset g = io::read_mgf_text("BEGIN IONS\nTITLE=x\n100 1\nEND IONS\n");   // from a string

io::DelimitedReadOptions tsv_opt;            // TSV / CSV
io::read_tsv_text("Name\tID\tPeak\nA\t007\t1,2\n").value(0, "ID");            // "007" (text)
tsv_opt.infer_types = true;
io::read_tsv_text("Name\tID\tPeak\nA\t007\t1,2\n", tsv_opt).value(0, "ID");   // 7 (integer)
```

`MSDS` files:

```cpp
MSDataset a = io::read_msds("example.msds");                     // everything
MSDataset b = io::read_msds("example.msds", false);              // skip peak metadata (faster)
io::MSDatasetMeta meta = io::read_msds_meta("example.msds");     // only /metadata
meta.description;                                                // ""
meta.attributes.size();                                          // 0
```

---

## 2. Looking at a dataset

```cpp
d.size();             // 2   spectra
d.n_columns();        // 6
d.n_peaks_total();    // 7
d.columns();          // {"Name", "PrecursorMZ", "AdductType", "Formula",
                      //  "CollisionEnergy", "NumPeaks"}
```

### Metadata values

A cell is a `Value`: missing, `bool`, `int64_t`, `double`, or `std::string`.

```cpp
Value v = d.value(1, "CollisionEnergy");      // std::string "20.0"  (MSP values are text)
std::get<std::string>(v);                     // "20.0"
to_display_string(v);                         // "20.0"   (any Value → text)
is_null(v);                                   // false

d.column_values("Name");                      // {"MassSpecID0000001", "MassSpecID0000002"}
d.value(0, "Missing");                        // throws msentity::KeyError

std::shared_ptr<arrow::Table> t = d.metadata();   // visible rows × visible columns
t->num_rows();                                    // 2
t->schema()->field(0)->type()->ToString();        // "string"
```

### One spectrum

```cpp
SpectrumRecord rec = d[0];          // spectrum 0 of the dataset
rec["Name"];                        // "MassSpecID0000001"
rec.n_peaks();                      // 4

Spectrum sp = rec.peaks();          // its peaks
sp.size();                          // 4
sp.mz(0);                           // 91.0542
sp.intensity(0);                    // 0.245   (245 / 1000)

Peak p = sp.peak(1);                // one peak with its metadata
p.mz;                               // 125.0233
p.intensity;                        // 1.0
p.metadata.at("formula");           // "C6H5O3+"
```

### All peaks at once

```cpp
PeakSeries& ps = d.peaks();
ps.offsets();            // {0, 4, 7}      spectrum i = peaks offsets[i] … offsets[i+1]-1
ps.lengths();            // {4, 3}
ps.mz();                 // {91.0542, 125.0233, 185.0961, 246.1125, 91.0542, 125.0233, 246.1125}
ps.intensity();          // {0.245, 1.0, 0.349, 0.735, 0.099, 0.281, 1.0}
ps.data();               // {91.0542, 0.245, 125.0233, 1.0, …}   (m/z, intensity) pairs
ps.metadata_columns();   // {"formula"}
ps.metadata();           // arrow::Table, 7 rows × 1 column ("C7H7+", "C6H5O3+", "", …)
```

---

## 3. Views: selecting and sorting

`select`, `slice`, `filter`, and `sort_by` return **views**. A view shares the
metadata table and the peaks with the dataset it came from; no peaks are
copied.

```cpp
MSDataset v = d.select({1});          // only spectrum 1
v.size();                             // 1
v.value(0, "Name");                   // "MassSpecID0000002"
v.n_peaks_total();                    // 3
v.peaks().index();                    // {1}   ← position in the shared storage

d.slice(0, 1);                        // spectrum 0           (start, stop)
d.filter({false, true});              // spectrum 1           (one bool per spectrum)
d.select({1, 0});                     // both, reversed
d.sort_by("CollisionEnergy");         // "20.0" < "30.0" → MassSpecID0000002, MassSpecID0000001
d.sort_by("CollisionEnergy", false);  // descending; stable; missing values always last
```

Writing through a view changes the shared data:

```cpp
v.set_column("Checked", Value(true));   // sets row 1 only; row 0 is missing
v.columns();                            // {…, "NumPeaks", "Checked"}
d.has_column("Checked");                // false  ← each view has its own column list
d.full_view(true).column_values("Checked");   // {missing, true}   (reset columns → visible)
```

Make an independent dataset with `copy()`:

```cpp
MSDataset c = v.copy();      // only spectrum 1, own storage
c.peaks().index();           // {0}
c.peaks().offsets();         // {0, 3}
```

Other view operations:

```cpp
d.set_columns({"Name", "NumPeaks"});   // show these columns, in this order
d.reset_view();                        // all spectra again (columns kept)
d.reset_view(true);                    // all spectra and all columns
MSDataset whole = v.full_view();       // like reset_view, but returns a new view
```

---

## 4. Editing metadata

```cpp
d.set_column("Score", std::vector<Value>{0.5, int64_t{2}});   // one value per spectrum
d.metadata()->schema()->GetFieldByName("Score")->type();      // double (int + double → double)
d.set_column("Flag", Value(true));                            // same value for all spectra
d.set_value(0, "Name", std::string("first"));                 // one cell
d[0].set("Name", std::string("first"));                       // the same, via the record
```

How the column type follows from the values:

| Values | Column type |
|---|---|
| only text | `string` |
| only booleans | `bool` |
| only integers | `int64` |
| integers and floats | `double` |
| text mixed with anything | `string` (other values converted with `to_display_string`) |
| NaN | stored as missing |

Dataset-level metadata:

```cpp
d.set_description("Two spectra of C16H17NO4");
d.set_attribute("source", "MassSpec");       // key/value pairs, insertion order kept
d.attribute("source");                       // std::optional<std::string> "MassSpec"
d.add_tag("example");                        // true (false if already present)
d.tags();                                    // {"example"}
```

Sequential IDs:

```cpp
processing::set_spec_id(d, "SpecID", "SP");  // SpecID = "SP1", "SP2"
// 12 spectra → "SP01" … "SP12" (zero-padded to the largest number)
// returns false and changes nothing if the column exists (unless overwrite = true)
```

---

## 5. Editing peaks

```cpp
PeakSeries& ps = d.peaks();
ps.normalize();                        // in place: highest peak of each spectrum = 1.0
PeakSeries n = ps.normalized(100.0);   // new series, highest peak = 100

PeakSeries byint = ps.sorted_by_intensity();   // new series, descending intensity
byint.mz();                // {125.0233, 246.1125, 185.0961, 91.0542, 246.1125, 125.0233, 91.0542}

std::vector<int64_t> perm;
ps.sort_by_mz(true, &perm);            // in place; perm[k] = old position of new peak k

ps.set_metadata_column("note", std::vector<Value>(ps.n_peaks_total(), std::string("")));
ps.set_metadata_column("source", Value(std::string("MSP")));   // same value for all peaks
```

Building a dataset from scratch:

```cpp
PeakSeries peaks = PeakSeries::from_spectra({
    {{100.0, 1.0}, {150.0, 0.5}},      // spectrum 0: 2 peaks
    {{90.0, 1.0}},                     // spectrum 1: 1 peak
});
peaks.offsets();                       // {0, 2, 3}

MSDataset fresh(nullptr, peaks);       // no metadata columns yet
fresh.set_column("Name", std::vector<Value>{std::string("a"), std::string("b")});
```

Combining datasets:

```cpp
MSDataset all = MSDataset::concat({d, fresh}, "combined");
all.size();                            // 4
all.column_values("Name");             // {"first", "MassSpecID0000002", "a", "b"}
// columns: union in order of first appearance; cells a dataset lacks are missing
```

---

## 6. Saving

```cpp
d.save("out.msds");                                  // = io::write_msds(d, "out.msds")

io::MsdsWriteOptions w;
w.save_view = false;     // store the whole storage + view index (default: only what is visible)
w.append = true;         // add dataset_1, dataset_2, … instead of replacing the file
io::write_msds(d, "out.msds", w);

std::vector<uint8_t> bytes = io::msds_to_bytes(d);   // an .msds file in memory
MSDataset back = io::msds_from_bytes(bytes);

io::write_msp(d, "out.msp");                         // see docs/format/text-formats.md
io::write_mgf(d, "out.mgf");
io::write_tsv(d, "out.tsv");                         // Peak cell: "mz,intensity,formula;91.05…,0.245,C7H7+;…"
io::write_csv(d, "out.csv");

io::DelimitedWriteOptions plain;
plain.peak_headers = std::vector<std::string>{};     // TSV/CSV without per-peak metadata
io::write_tsv(d, "plain.tsv", plain);                // Peak cell: "91.05…,0.245;125.02…,1;…"

io::TextWriteOptions t;
t.headers = std::vector<std::string>{"Name", "PrecursorMZ"};   // columns to write
t.header_map = {{"PrecursorMZ", "PRECURSORMZ"}};               // rename in the output
t.peak_headers = std::vector<std::string>{"formula"};          // peak columns to write
t.delimiter = " ";                                             // between m/z and intensity
io::write_msp(d, "selected.msp", t);
```

---

## 7. Small helpers

```cpp
format_float(1e-5);                 // "1e-05"
format_float(100.0);                // "100.0"
to_display_string(Value(true));     // "True"
to_display_string(Value());         // ""   (missing)
```

## 8. Errors

| Exception | When |
|---|---|
| `std::invalid_argument` | wrong argument or invalid data (bad TSV row, wrong number of values, …) |
| `std::out_of_range` | spectrum or peak position out of range |
| `msentity::KeyError` | unknown column |
| `msentity::FormatError` | file is not valid HDF5/Parquet/MSDS |
| `std::runtime_error` | file cannot be opened or written |

---

## 9. Command-line tool

`msentity-tool` exposes the library for quick checks and for the
conformance tests:

| Command | Does |
|---|---|
| `msentity-tool dump example.msp` | print the dataset as JSON (columns, values, offsets, peaks, peak metadata) |
| `msentity-tool convert example.msp example.msds` | convert; output format from the extension |
| `msentity-tool meta example.msds` | print description, attributes, tags |

Options: `--type T` (input format), `--spec-id-prefix P`, `--infer-types`
(TSV/CSV), `--no-save-view` (raw MSDS writing).
