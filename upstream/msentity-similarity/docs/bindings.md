# Python, Java, C#, and C

All four use the same shared library, `libmsentity_similarity_c`, through its C
API. Datasets come from the msentity binding of the same language and are passed
to the library as `.msds` bytes, so the two libraries never share C++ objects
and can be built and upgraded separately.

| Language | Package | Needs | Datasets from |
|---|---|---|---|
| Python | `msentity_similarity` | Python ≥ 3.10, NumPy, the `msentity` package | `msentity.Dataset` |
| Java | `io.github.motohiroogawa:msentity-similarity` | Java ≥ 22, `--enable-native-access=ALL-UNNAMED`, `msentity` artifact | `io.github.motohiroogawa.msentity.Dataset` |
| C# | `MsEntity.Similarity` | .NET ≥ 8, `MsEntity` package | `MsEntity.Dataset` |
| C | `msentity/similarity/c/msentity_similarity_c.h` | — | `.msds` bytes (`msentity_to_msds_bytes`) |

Every example uses [examples/example.msp](examples/example.msp) (2 spectra) and
gives the real result. The calculations are defined in
[similarity.md](similarity.md) and the file format in
[format/mssim.md](format/mssim.md).

## Python

```python
import msentity as ms
import msentity_similarity as mss

ds = ms.load("example.msp")

# Plain scores (NumPy float32)
mss.pair_scores(ds, [0, 1], ds, [1, 0])          # array([0.75971437, 0.75971437])
mss.all_pairs(ds, ds, method="reverse_cosine")   # array([[1.        , 0.78809655],
                                                 #        [0.75971437, 1.        ]])
# keyword options: method="cosine"|"reverse_cosine", bin_width=0.01,
#                  intensity_exponent=1.0, threads=0

# Library search → SimilarityDataset
hits = mss.library_search(ds, ds, 0.7,
                          query_source="example.msp",          # recorded in the metadata
                          progress=lambda done, total: None)    # optional
len(hits)                       # 4
hits.columns                    # ['index1', 'index2', 'cosine_similarity', 'data_index1', 'data_index2']
hits["cosine_similarity"]       # [1.0, 0.7597143650054932, 0.7597143650054932, 1.0]
hits.metadata["parameters"]     # {'threshold': 0.7, 'method': 'cosine', 'bin_width': 0.01,
                                #  'intensity_exponent': 1.0}
query, ref = hits.match_records(1)   # msentity.Record of each side
query["Name"], ref["Name"]           # ('MassSpecID0000001', 'MassSpecID0000002')
ref.mz                               # array([ 91.0542, 125.0233, 246.1125])

# Filter, sort, statistics
different = hits.filter([a != b for a, b in zip(hits["index1"], hits["index2"])])
best = different.sort_by("cosine_similarity", ascending=False)
hits.describe_scores()          # {'count': 4, 'mean': 0.879857…, 'q1': 0.759714…,
                                #  'median': 0.879857…, 'q3': 1.0, 'min': 0.759714…, 'max': 1.0}
hits.histogram(5)               # (array([0, 0, 0, 2, 2]), array([0. , 0.2, 0.4, 0.6, 0.8, 1. ]))
hits.to_pandas()                # DataFrame (needs pyarrow + pandas); to_arrow() for pyarrow

# Files
best.save("best.mssim")
loaded = mss.load("best.mssim")
loaded.matched_datasets         # (msentity.Dataset, msentity.Dataset) of the embedded spectra
loaded.source_indices(1)        # array([0, 1]): their positions in the query input
hits.export("hits.csv")         # table only: .csv, .tsv, .parquet

# By key
ds.set_spec_id("SpecID", "SP")
pairs = mss.similarity_by_key(ds, ds[[1, 0]], "SpecID", "SpecID")
pairs.to_pandas()
#   SpecID  index1  index2  cosine_similarity
# 0    SP1       0       1                1.0
# 1    SP2       1       0                1.0
```

Errors are msentity's exception classes (`mss.InvalidArgumentError` is
`ms.InvalidArgumentError`, also a `ValueError`; `PositionError`, `ColumnError`,
`FormatError`, `FileError`).

Native library: `MSSIM_C_LIBRARY`, else `msentity_similarity/lib/`, else
`<repository>/build/`.

## Java

```java
import io.github.motohiroogawa.msentity.Dataset;
import io.github.motohiroogawa.msentity.similarity.*;

try (Dataset ds = Dataset.load(Path.of("example.msp"))) {
  Similarity.pairScores(ds, new long[] {0, 1}, ds, new long[] {1, 0});
  // [0.75971437, 0.75971437]
  float[][] matrix = Similarity.allPairs(ds, ds,
      Options.defaults().withMethod(Options.Method.REVERSE_COSINE));
  // [[1.0, 0.78809655], [0.75971437, 1.0]]

  try (SimilarityDataset hits = Similarity.librarySearch(
           ds, ds, 0.7, Options.defaults(),
           true,                            // include matched data
           "example.msp", "example.msp",    // sources for the metadata
           (done, total) -> {})) {          // progress, may be null
    hits.size();                            // 4
    hits.columns();      // [index1, index2, cosine_similarity, data_index1, data_index2]
    hits.column("cosine_similarity");       // [1.0, 0.7597143650054932, 0.7597143650054932, 1.0]
    hits.metadataJson();                    // {"created_at":"…","mode":"library_search",…}

    try (SimilarityDataset.Match m = hits.matchRecords(1)) {   // one-spectrum Datasets
      m.first().value(0, "Name");           // MassSpecID0000001
      m.second().value(0, "Name");          // MassSpecID0000002
    }

    try (SimilarityDataset best = hits.sortBy("cosine_similarity", false)) {
      best.save(Path.of("best.mssim"));
    }
    hits.describeScores();   // ScoreSummary[count=4, mean=0.8798…, q1=0.7597…, median=0.8798…, …]
    hits.histogram(5);       // Histogram[counts=[0, 0, 0, 2, 2], edges=[0.0, 0.2, …, 1.0]]
    hits.exportTable(Path.of("hits.tsv"));
  }

  ds.setSpecId("SpecID", "SP", false, 1);
  try (Dataset other = ds.select(1, 0);
       SimilarityDataset pairs = Similarity.similarityByKey(ds, other)) {  // key "SpecID"
    pairs.columns();         // [SpecID, index1, index2, cosine_similarity]
  }
}
SimilarityDataset loaded = SimilarityDataset.load(Path.of("best.mssim"));
loaded.matchedDataset(1);    // Dataset of the embedded query spectra
loaded.sourceIndices(1);     // [0, 1]
```

Errors: `SimilarityException` and its subclasses `InvalidArgument`,
`PositionOutOfRange`, `ColumnNotFound`, `Format`, `FileAccess` (same status
codes as msentity). Native library: `-Dmssim.library=…`, else
`MSSIM_C_LIBRARY`, else the copy bundled in the JAR, else the system path.
Run with `--enable-native-access=ALL-UNNAMED`.

## C#

```csharp
using MsEntity;
using MsEntity.Similarity;

using var ds = Dataset.Load("example.msp");

SpectrumSimilarity.PairScores(ds, new long[] { 0, 1 }, ds, new long[] { 1, 0 });
// { 0.75971437, 0.75971437 }
float[,] matrix = SpectrumSimilarity.AllPairs(ds, ds,
    new SimilarityOptions(Method: SimilarityMethod.ReverseCosine));
// { { 1, 0.78809655 }, { 0.75971437, 1 } }

using var hits = SpectrumSimilarity.LibrarySearch(ds, ds, threshold: 0.7,
    querySource: "example.msp", progress: (done, total) => { });
hits.Count;                          // 4
hits.Columns;                        // index1, index2, cosine_similarity, data_index1, data_index2
hits["cosine_similarity"];           // 1.0, 0.7597143650054932, 0.7597143650054932, 1.0
hits.MetadataJson;                   // {"created_at":"…","mode":"library_search",…}

var (query, reference) = hits.MatchRecords(1);    // one-spectrum Datasets (dispose them)
query.GetValue(0, "Name");           // MassSpecID0000001
reference.GetValue(0, "Name");       // MassSpecID0000002

using var best = hits.SortBy("cosine_similarity", ascending: false);
best.Save("best.mssim");
hits.DescribeScores();               // ScoreSummary { Count = 4, Mean = 0.8798…, … }
hits.Histogram(5).Counts;            // { 0, 0, 0, 2, 2 }
hits.ExportTable("hits.csv");

ds.SetSpecId("SpecID", "SP");
using var other = ds.Select(1, 0);
using var pairs = SpectrumSimilarity.SimilarityByKey(ds, other);   // key "SpecID"
pairs.Columns;                       // SpecID, index1, index2, cosine_similarity

using var loaded = SimilarityDataset.Load("best.mssim");
loaded.SourceIndices(1);             // { 0, 1 }
```

Errors: `SimilarityException` and its subclasses
`SimilarityArgumentException`, `SimilarityPositionException`,
`SimilarityColumnNotFoundException`, `SimilarityFormatException`,
`SimilarityFileException`. Native library: `MSSIM_C_LIBRARY`, else the
application directory or `runtimes/<rid>/native/` of the NuGet package, else
the system path.

## C

The C API follows msentity's conventions: functions return `MSENTITY_OK` (0) or
an `MSENTITY_ERR_*` code, `mssim_last_error()` gives the message (per thread),
strings are freed with `mssim_string_free`, byte buffers with
`mssim_bytes_free`, and cells (`msentity_value`) with `mssim_values_clear`.

```c
#include <msentity/c/msentity_c.h>
#include <msentity/similarity/c/msentity_similarity_c.h>

msentity_dataset* ds;
msentity_load("example.msp", NULL, NULL, 0, &ds);
uint8_t* bytes; int64_t size;
msentity_to_msds_bytes(ds, &bytes, &size);       /* the dataset as .msds bytes */

mssim_options options;
mssim_default_options(&options);                 /* cosine, 0.01, 1.0, all threads */
float matrix[4];
mssim_all_pairs(bytes, size, bytes, size, &options, matrix, 4);
/* matrix = {1, 0.759714, 0.759714, 1} */

mssim_result* hits;
if (mssim_library_search(bytes, size, bytes, size, 0.7, &options,
                         1,                      /* include matched data */
                         "example.msp", "example.msp",
                         NULL, NULL,             /* progress callback, user data */
                         &hits) != MSENTITY_OK) {
  fprintf(stderr, "%s\n", mssim_last_error());
}
mssim_size(hits);                                /* 4 */
msentity_value scores[4];
mssim_get_column(hits, "cosine_similarity", scores, 4);   /* scores[1].float_value = 0.7597… */
mssim_values_clear(scores, 4);
mssim_save(hits, "hits.mssim");

mssim_free(hits);
msentity_bytes_free(bytes);
msentity_free(ds);
```

The header [msentity_similarity_c.h](../include/msentity/similarity/c/msentity_similarity_c.h)
lists every function: calculations (`mssim_pair_scores`, `mssim_all_pairs`,
`mssim_library_search`, `mssim_similarity_by_key`), files (`mssim_load`,
`mssim_save`, `mssim_export_table`), the table (`mssim_size`,
`mssim_column_name`, `mssim_get_column`, `mssim_table_arrow`), metadata, matched
data (`mssim_matched_dataset` returns `.msds` bytes for `msentity_from_msds_bytes`),
and `mssim_filter`, `mssim_sort_by`, `mssim_describe_scores`, `mssim_histogram`.
