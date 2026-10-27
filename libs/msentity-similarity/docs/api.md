# C++ API guide

Every example uses [examples/example.msp](examples/example.msp) (2 spectra) and
shows the real output in comments. Headers:

```cpp
#include <msentity/msentity.hpp>                 // MSDataset, io::load_ms_dataset, ...
#include <msentity/similarity/similarity.hpp>    // everything in this guide

using namespace msentity;
namespace sim = msentity::similarity;
```

| Header | Contents |
|---|---|
| `msentity/similarity/calculation.hpp` | `Method`, `Options`, `pair_scores`, `all_pairs_matrix`, `library_search`, `similarity_by_key` (plain scores / Arrow tables) |
| `msentity/similarity/similarity_dataset.hpp` | `SimilarityDataset` — a result with metadata, the `.mssim` reader/writer |
| `msentity/similarity/calculate.hpp` | `calculate_library_search`, `calculate_similarity` — calculation + metadata → `SimilarityDataset` |
| `msentity/similarity/c/msentity_similarity_c.h` | the C API used by the bindings ([bindings.md](bindings.md#c-1)) |

## 1. Options

```cpp
sim::Options options;                       // defaults:
options.method = sim::Method::Cosine;       //   Cosine or ReverseCosine
options.bin_width = 0.01;                   //   m/z bin width, > 0
options.intensity_exponent = 1.0;           //   intensity^exponent before scoring, > 0
options.threads = 0;                        //   0 = all hardware threads
```

`sim::parse_method("reverse_cosine")` and `sim::method_name(Method::Cosine)`
(`"cosine"`) convert to and from the names used in files and bindings. The
scores themselves are defined in [similarity.md](similarity.md).

## 2. Plain scores

```cpp
MSDataset ds = io::load_ms_dataset("example.msp");

// Chosen pairs: (ds[0], ds[1]) and (ds[1], ds[0])
std::vector<float> scores = sim::pair_scores(ds, {0, 1}, ds, {1, 0});
// scores = {0.759714, 0.759714}

// Every pair, row-major len(ds1) × len(ds2)
sim::Options reverse;
reverse.method = sim::Method::ReverseCosine;
std::vector<float> matrix = sim::all_pairs_matrix(ds, ds, reverse);
// matrix = {1, 0.788097,      ← query 0 vs reference 0, 1
//           0.759714, 1}      ← query 1 vs reference 0, 1
```

`library_search` and `similarity_by_key` in `calculation.hpp` return the bare
Arrow table; the `calculate_*` functions below wrap it with metadata, which is
what you normally want.

## 3. Library search → `SimilarityDataset`

```cpp
sim::SimilarityDataset hits = sim::calculate_library_search(
    ds, ds,                    // query, reference
    0.7,                       // keep pairs with score >= 0.7
    {},                        // Options
    true,                      // include_matched_data: embed the compared spectra
    "example.msp",             // query_source: recorded in the metadata
    "example.msp",             // reference_source
    [](int64_t done, int64_t total) {});  // optional progress callback

hits.size();      // 4
hits.columns();   // {"index1", "index2", "cosine_similarity", "data_index1", "data_index2"}
```

The result table:

```text
 index1  index2  cosine_similarity  data_index1  data_index2
      0       0  1.0                          0            0
      0       1  0.759714                     0            1
      1       0  0.759714                     1            0
      1       1  1.0                          1            1
```

* `index1` / `index2` — positions in the query / reference dataset.
* `cosine_similarity` — the score (this name is used for every method).
* `data_index1` / `data_index2` — rows of the embedded spectra (only with
  `include_matched_data`).

Reading values and the compared spectra:

```cpp
std::vector<Value> score = hits.column_values("cosine_similarity");
std::get<double>(score[1]);              // 0.759714

auto [query, reference] = hits.match_records(1);   // two one-spectrum MSDatasets
std::get<std::string>(query.value(0, "Name"));     // "MassSpecID0000001"
std::get<std::string>(reference.value(0, "Name")); // "MassSpecID0000002"

hits.table();            // std::shared_ptr<arrow::Table>
hits.metadata_json();    // {"created_at":"…","mode":"library_search",…}  (see format/mssim.md)
hits.matched_datasets(); // std::optional<pair<MSDataset, MSDataset>>: the embedded spectra
hits.matched_source_indices(); // their positions in the inputs: {0, 1} and {0, 1}
```

## 4. Filtering, sorting, statistics

```cpp
// Pairs of different spectra, best first
std::vector<bool> mask;
auto i1 = hits.column_values("index1"), i2 = hits.column_values("index2");
for (std::size_t r = 0; r < i1.size(); ++r) mask.push_back(i1[r] != i2[r]);

sim::SimilarityDataset different =
    hits.filter(mask)                                  // 2 rows; metadata row_count = 2,
        .sort_by({"cosine_similarity"}, {false});      //   source_row_count = 4
                                                       // stable; missing values last

sim::ScoreSummary s = hits.describe_scores();
// s.count = 4, s.mean = 0.879857, s.median = 0.879857, s.q1, s.q3, s.min, s.max
// (quartiles by linear interpolation, as in NumPy/pandas)

sim::Histogram h = hits.histogram(5);                  // 1–200 bins over [0, 1]
// h.counts = {0, 0, 0, 2, 2}, h.edges = {0, 0.2, 0.4, 0.6, 0.8, 1}  (1.0 is in the last bin)
```

## 5. Files

```cpp
different.save("different.mssim");      // HDF5 .mssim (extension required); atomic write.
                                        // Embedded spectra are reduced to the ones still
                                        // referenced, so filtered results stay small.
auto loaded = sim::SimilarityDataset::load("different.mssim");
loaded.matched_datasets()->first.size();   // 2

hits.export_table("hits.csv");          // table only: .csv, .tsv, or .parquet
```

`hits.csv`:

```text
index1,index2,cosine_similarity,data_index1,data_index2
0,0,1.0,0,0
0,1,0.7597143650054932,0,1
1,0,0.7597143650054932,1,0
1,1,1.0,1,1
```

The file layout is specified in [format/mssim.md](format/mssim.md).

## 6. By key

Compare the spectra that share a key value — for example the same compound
measured twice:

```cpp
processing::set_spec_id(ds, "SpecID", "SP");            // SP1, SP2
MSDataset other = ds.select({1, 0});                    // same spectra, other order

sim::SimilarityDataset by_key = sim::calculate_similarity(
    ds, other,
    "SpecID", "SpecID",    // key column in ds, key column in other
    {},                    // Options
    false,                 // include_matched_data (default false)
    "a.msds", "b.msds");   // sources for the metadata

by_key.columns();          // {"SpecID", "index1", "index2", "cosine_similarity"}
// SpecID index1 index2 cosine_similarity
// SP1         0      1  1.0
// SP2         1      0  1.0
```

Key values must be unique within each dataset (`std::invalid_argument`
otherwise); an unknown key column throws `KeyError`.

## 7. Errors

| Exception | When |
|---|---|
| `std::invalid_argument` | bad options (`bin_width <= 0`, threshold outside [0, 1], unknown method), duplicate keys, invalid tables or metadata, wrong file extension |
| `std::out_of_range` | a spectrum position or row out of range |
| `msentity::KeyError` | unknown column |
| `msentity::FormatError` | not an `.mssim` file, unsupported schema version, or the file cannot be opened or written (message starting with `HDF5 error: cannot open` / `cannot create`) |

## 8. Command-line tool

`mssim-tool` (built with `MSENTITY_SIMILARITY_BUILD_TOOLS=ON`, the default for
a top-level build) runs the same calculations on files of any format msentity
reads (MSP, MGF, MSDS, TSV, CSV):

```console
$ mssim-tool search example.msp example.msp hits.mssim --threshold 0.7
4 pairs written to hits.mssim
$ mssim-tool by-key a.msds b.msds pairs.mssim --key1 SpecID --key2 SpecID --embed
$ mssim-tool info hits.mssim          # size, columns, metadata, score summary (JSON)
$ mssim-tool export hits.mssim hits.tsv
```

Options for `search` and `by-key`: `--method cosine|reverse_cosine`,
`--bin-width W`, `--exponent E`, `--threads N`; `search` embeds the spectra
unless `--lightweight` is given, `by-key` only with `--embed`.
