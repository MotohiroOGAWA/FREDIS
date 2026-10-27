# Python, Java, and C#

The library can be used from Python, Java, and C#. All three call the same
C API (`libmsentity_c`), so they read and write exactly the same files and
behave the same way: a file saved from one language opens unchanged in the
others.

```text
 Python (ctypes + NumPy)   Java (Foreign Function & Memory API)   C# (P/Invoke)
          │                              │                               │
          └──────────────────────────────┼───────────────────────────────┘
                                         ▼
                     libmsentity_c   — C API: include/msentity/c/msentity_c.h
                                         ▼
                     msentity C++ core (data model, MSDS/MSP/MGF/TSV/CSV)
```

| | Python | Java | C# |
|---|---|---|---|
| Directory | `bindings/python` | `bindings/java` | `bindings/csharp` |
| Package | `msentity` | `io.github.motohiroogawa.msentity` | `MsEntity` |
| Requires | Python ≥ 3.10, NumPy (pyarrow/pandas optional) | Java ≥ 22 | .NET ≥ 8 |
| Native access | ctypes (standard library) | `java.lang.foreign` (standard library) | P/Invoke |
| Main type | `Dataset` | `Dataset` (`AutoCloseable`) | `Dataset` (`IDisposable`) |

---

## 1. Setup

Install a release package as described in [install.md](install.md); the
packages contain the native library. When working from a source checkout,
build the library first (`cmake -S . -B build && cmake --build build`); each
binding then finds `build/libmsentity_c.*`.

Each binding finds the native library in this order:

| Language | 1. explicit | 2. packaged | 3. other |
|---|---|---|---|
| Python | `MSENTITY_C_LIBRARY` | `msentity/lib/` (wheel) | `<repo>/build/`, system library path |
| Java | `-Dmsentity.library=…`, then `MSENTITY_C_LIBRARY` | copy inside the JAR for the current platform | system library path |
| C# | `MSENTITY_C_LIBRARY` | `runtimes/<rid>/native/` (NuGet), application directory | system library path |

Running the tests (also part of `ctest --test-dir build`):

```console
cd bindings/python && python -m unittest discover -s tests       # 14 tests
cd bindings/java   && mvn test                                     # 7 tests
cd bindings/csharp && dotnet test MsEntity.slnx                   # 7 tests
```

---

## 2. The same task in each language

Load [`docs/examples/example.msp`](examples/example.msp) (two spectra; some
peaks have a `formula`), look at it, take a view, edit, and save. Lines such as
`ds.Count;` stand for the expression; the comment shows its actual result.

### Python

```python
import msentity as ms

ds = ms.load("example.msp")
len(ds)                          # 2
ds.columns                       # ['Name', 'PrecursorMZ', 'AdductType', 'Formula', 'CollisionEnergy', 'NumPeaks']
ds["Name"]                       # ['MassSpecID0000001', 'MassSpecID0000002']
ds.value(1, "CollisionEnergy")   # '20.0'

rec = ds[0]                      # one spectrum
rec["Name"]                      # 'MassSpecID0000001'
rec.mz                           # array([ 91.0542, 125.0233, 185.0961, 246.1125])
rec.intensity                    # array([0.245, 1.   , 0.349, 0.735])
ds.offsets                       # array([0, 4, 7])
ds.peak_column("formula")[:2]    # ['C7H7+', 'C6H5O3+']

view = ds[[1]]                   # also ds[1:], ds[mask], ds.sort_by("CollisionEnergy")
view["Checked"] = True           # writes into the shared data
copy = view.copy()               # independent

ds.description = "two spectra"
ds.set_attribute("source", "MassSpec")
ds.add_tag("example")
ds.save("example.msds")          # or .msp .mgf .tsv .csv

ds.to_pandas()                   # metadata as a DataFrame (pyarrow + pandas)
```

### Java

```java
import io.github.motohiroogawa.msentity.Dataset;
import java.nio.file.Path;
import java.util.List;

try (Dataset ds = Dataset.load(Path.of("example.msp"))) {
  ds.size();                            // 2
  ds.columns();                         // [Name, PrecursorMZ, AdductType, Formula, CollisionEnergy, NumPeaks]
  ds.column("Name");                    // [MassSpecID0000001, MassSpecID0000002]
  ds.value(1, "CollisionEnergy");       // "20.0"

  Dataset.Spectrum sp = ds.spectrum(0);
  sp.mz();                              // [91.0542, 125.0233, 185.0961, 246.1125]
  sp.intensity();                       // [0.245, 1.0, 0.349, 0.735]
  ds.offsets();                         // [0, 4, 7]
  ds.peakColumn("formula").get(1);      // "C6H5O3+"

  Dataset view = ds.select(1);          // also slice(1, 2), filter(mask), sortBy("CollisionEnergy", true)
  view.fillColumn("Checked", true);     // writes into the shared data
  Dataset copy = view.copy();           // independent

  ds.setDescription("two spectra");
  ds.setAttribute("source", "MassSpec");
  ds.addTag("example");
  ds.save(Path.of("example.msds"));     // or .msp .mgf .tsv .csv
}
```

Run with `--enable-native-access=ALL-UNNAMED` (otherwise Java prints a
warning on first use of native code).

### C#

```csharp
using MsEntity;

using var ds = Dataset.Load("example.msp");
ds.Count;                               // 2
ds.Columns;                             // [Name, PrecursorMZ, AdductType, Formula, CollisionEnergy, NumPeaks]
ds["Name"];                             // [MassSpecID0000001, MassSpecID0000002]
ds.GetValue(1, "CollisionEnergy");      // "20.0"

Spectrum sp = ds.GetSpectrum(0);
sp.Mz;                                  // [91.0542, 125.0233, 185.0961, 246.1125]
sp.Intensity;                           // [0.245, 1, 0.349, 0.735]
ds.Offsets;                             // [0, 4, 7]
ds.GetPeakColumn("formula")[1];         // "C6H5O3+"

using var view = ds.Select(1);          // also Slice(1, 2), Filter(mask), SortBy("CollisionEnergy")
view.FillColumn("Checked", true);       // writes into the shared data
using var copy = view.Copy();           // independent

ds.Description = "two spectra";
ds.SetAttribute("source", "MassSpec");
ds.AddTag("example");
ds.Save("example.msds");                // or .msp .mgf .tsv .csv
```

---

## 3. API correspondence

| Task | Python | Java | C# |
|---|---|---|---|
| load a file | `ms.load(path, file_type=None, spec_id_prefix=None, infer_types=False)` | `Dataset.load(path, fileType, specIdPrefix, inferTypes)` | `Dataset.Load(path, fileType, specIdPrefix, inferTypes)` |
| parse text | `ms.read_text(text, "msp")` | `Dataset.readText(text, "msp", false)` | `Dataset.ReadText(text, "msp")` |
| from peaks | `ms.from_peaks([[(mz, i), …], …])` | `Dataset.fromPeaks(double[][][])` | `Dataset.FromPeaks((double, double)[][])` |
| join | `ms.concat([a, b])` | `Dataset.concat(a, b)` | `Dataset.Concat(a, b)` |
| number of spectra / peaks | `len(ds)`, `ds.n_peaks_total` | `size()`, `nPeaksTotal()` | `Count`, `PeakCount` |
| visible columns | `ds.columns` (assignable) | `columns()`, `setColumns(list)` | `Columns` (settable) |
| one cell | `ds.value(i, col)`, `ds[i][col]` | `value(i, col)` | `GetValue(i, col)` |
| a column | `ds[col]`, `ds.column(col)` | `column(col)` | `ds[col]`, `GetColumn(col)` |
| set a column | `ds[col] = values` or a single value | `setColumn(col, list)`, `fillColumn(col, value)` | `SetColumn(col, values)`, `FillColumn(col, value)` |
| set a cell | `ds[i][col] = v` | `setValue(i, col, v)` | `SetValue(i, col, v)` |
| views | `ds[a:b]`, `ds[[i, j]]`, `ds[mask]`, `sort_by` | `slice`, `select`, `filter`, `sortBy` | `Slice`, `Select`, `Filter`, `SortBy` |
| copy / full view | `copy()`, `full_view(reset_columns)` | `copy()`, `fullView(reset)` | `Copy()`, `FullView(reset)` |
| peaks | `ds.offsets`, `ds.mz`, `ds.intensity`, `ds.spectrum(i)` | `offsets()`, `mz()`, `intensity()`, `spectrum(i)` | `Offsets`, `Mz`, `Intensity`, `GetSpectrum(i)` |
| scale / sort peaks | `normalize(scale)`, `sort_peaks("mz")` | `normalize`, `sortPeaksByMz`, `sortPeaksByIntensity` | `Normalize`, `SortPeaksByMz`, `SortPeaksByIntensity` |
| peak metadata | `peak_columns`, `peak_column`, `set_peak_column` | `peakColumns`, `peakColumn`, `setPeakColumn` | `PeakColumns`, `GetPeakColumn`, `SetPeakColumn` |
| dataset metadata | `description`, `attributes`, `tags`, `set_attribute`, `add_tag`, … | `description()`, `attributes()`, `tags()`, … | `Description`, `Attributes`, `Tags`, … |
| sequential IDs | `set_spec_id(column, prefix)` | `setSpecId(column, prefix, overwrite, start)` | `SetSpecId(column, prefix)` |
| save | `save(path, format=None, save_view=True, append=False)` | `save(path)`, `save(path, format, saveView, append)` | `Save(path, format, saveView, append)` |
| release | automatic, or `close()` / `with` | `close()` / try-with-resources (GC as fallback) | `Dispose()` / `using` (finalizer as fallback) |
| Arrow / DataFrame | `to_arrow()`, `peak_metadata_to_arrow()`, `to_pandas()` | — | — |

---

## 4. Values

A metadata cell has one of five values (see
[format/msds.md §4.3](format/msds.md#43-column-types-and-cell-values)):

| Value | Python | Java | C# |
|---|---|---|---|
| missing | `None` | `null` | `null` |
| boolean | `bool` | `Boolean` | `bool` |
| integer | `int` | `Long` | `long` |
| float | `float` | `Double` | `double` |
| text | `str` | `String` | `string` |

When writing, other integer and floating-point types (`numpy.int64`, `Integer`,
`int`, `float`, …) are accepted, and NaN is stored as missing. A column's type
follows from its values (text only → text, integers → integer, integers and
floats → float, any mix with text → text).

## 5. Errors

| C status | Python | Java | C# |
|---|---|---|---|
| invalid argument / data | `InvalidArgumentError` (a `ValueError`) | `MsEntityException.InvalidArgument` | `MsEntityArgumentException` |
| position out of range | `PositionError` (an `IndexError`) | `MsEntityException.PositionOutOfRange` | `MsEntityPositionException` |
| unknown column | `ColumnError` (a `KeyError`) | `MsEntityException.ColumnNotFound` | `MsEntityColumnNotFoundException` |
| malformed file | `FormatError` | `MsEntityException.Format` | `MsEntityFormatException` |
| cannot open / write file | `FileError` (an `OSError`) | `MsEntityException.FileAccess` | `MsEntityFileException` |

All derive from a common base (`MsEntityError`, `MsEntityException`) whose
message is the library's error text.

## 6. Views and lifetime

* A view (`select`, slices, filters, `sort_by`, `full_view`) is its own
  object that shares data with the dataset it came from. Closing or disposing
  the original does not affect existing views; the data lives until the last
  object using it is released.
* Writing a column through a view changes the shared data, but the new column
  appears only in that view's column list (see the
  [C++ guide](api.md#3-views-selecting-and-sorting)).
* Arrays (`mz`, `offsets`, …) and value lists are copies; changing them does
  not change the dataset.

---

## 7. The C API

`include/msentity/c/msentity_c.h` is a plain C interface that any language
with a C FFI can use. Its conventions:

| Topic | Rule |
|---|---|
| handles | `msentity_dataset*`, released with `msentity_free`; views are new handles |
| errors | functions return `MSENTITY_OK` (0) or an `MSENTITY_ERR_*` code; `msentity_last_error()` has the message (per thread) |
| strings | UTF-8; strings returned through `char**` are freed with `msentity_string_free` |
| values | `msentity_value` {`int32 type`, `int32 bool_value`, `int64 int_value`, `double float_value`, `char* string_value`} (32 bytes); returned strings are freed with `msentity_values_clear` |
| arrays | copied into caller buffers whose length is passed and checked |
| Arrow | `msentity_metadata_arrow` / `msentity_peak_metadata_arrow` export an Arrow C stream (`ArrowArrayStream`) |

```c
#include <msentity/c/msentity_c.h>

msentity_dataset* ds = NULL;
if (msentity_load("example.msp", NULL, NULL, 0, &ds) != MSENTITY_OK) {
  fprintf(stderr, "%s\n", msentity_last_error());
  return 1;
}
int64_t n = msentity_size(ds);                       /* 2 */
msentity_value v;
msentity_get_value(ds, 1, "CollisionEnergy", &v);    /* v.type = MSENTITY_TYPE_STRING, "20.0" */
msentity_values_clear(&v, 1);
msentity_save(ds, "example.msds", NULL, 1, 0);
msentity_free(ds);
```

Link with `msentity::msentity_c` (CMake) or `-lmsentity_c`.

---

## 8. Packaging

Release packages are built by `.github/workflows/release.yml` (see
[install.md](install.md#how-releases-are-made)). The native library of each
platform is collected in `bindings/natives/<platform>/` and bundled as:

| Language | Package | Native library inside |
|---|---|---|
| Python | wheel `msentity-<v>-py3-none-<platform>.whl` (`bindings/python/build_wheel.py`) | `msentity/lib/` |
| Java | `msentity-<v>.jar` (all platforms in one JAR) | `io/github/motohiroogawa/msentity/native/<platform>/`, extracted to a temporary directory on first use |
| C# | `MsEntity.<v>.nupkg` | `runtimes/<platform>/native/` |
