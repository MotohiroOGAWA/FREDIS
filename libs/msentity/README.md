# msentity

C++ library for mass-spectrometry datasets: an in-memory data model for
spectra, peaks, and their metadata, and readers/writers for the MSDS, MSP,
MGF, TSV, and CSV formats. Python, Java, and C# use the same library through
its C API, so a file written from one language can be read from any other.

| | |
|---|---|
| Data model | `MSDataset` (Arrow metadata table + peaks), `PeakSeries` (CSR peak storage with views), `Value` |
| Formats | MSDS `.msds`, MSP, MGF, TSV, CSV |
| Languages | C++, C, Python (`msentity`), Java (`io.github.motohiroogawa.msentity`), C# (`MsEntity`) |
| Requirements | C++20, CMake ≥ 3.20, HDF5 ≥ 1.12 (C + HL), Apache Arrow & Parquet C++, nlohmann/json (build only) |

## Installation

Ready-made packages for Linux x64, Windows x64, and macOS arm64 are attached to
each [GitHub Release](https://github.com/MotohiroOGAWA/msentity/releases):

| Language | Install |
|---|---|
| Python | `pip install msentity --find-links https://github.com/MotohiroOGAWA/msentity/releases/expanded_assets/v0.1.0` |
| Java | JitPack: `com.github.MotohiroOGAWA:msentity:v0.1.0` (repository `https://jitpack.io`) |
| C# | `dotnet add package MsEntity --version 0.1.0 --source <folder with MsEntity.0.1.0.nupkg>` |
| C | `msentity-c-0.1.0-<platform>.zip` (header + shared library) |
| C++ | CMake `FetchContent` from this repository, or build and `cmake --install` |

Details for every language: [docs/install.md](docs/install.md).

## Documentation

* [Installation](docs/install.md)
* [C++ API guide](docs/api.md)
* [Python, Java, C#, and C](docs/bindings.md)
* [Architecture and language bindings](docs/architecture.md)
* File format specifications:
  [MSDS](docs/format/msds.md) ·
  [MSP/MGF/TSV/CSV](docs/format/text-formats.md)

## Building from source

Debian/Ubuntu packages (Arrow comes from the
[Apache Arrow APT repository](https://arrow.apache.org/install/)):

```console
sudo apt install cmake ninja-build libhdf5-dev nlohmann-json3-dev libgtest-dev
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update && sudo apt install libarrow-dev libparquet-dev
```

```console
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build            # library, headers, CMake package, msentity-tool
```

Options: `MSENTITY_BUILD_TESTS` (ON), `MSENTITY_BUILD_TOOLS` (ON),
`MSENTITY_BUILD_C_API` (ON, builds `libmsentity_c` for the bindings),
`BUILD_SHARED_LIBS` (OFF).

Use from another CMake project:

```cmake
find_package(msentity REQUIRED)
target_link_libraries(app PRIVATE msentity::msentity)
```

## Example

```cpp
#include <iostream>
#include <msentity/msentity.hpp>

using namespace msentity;

int main() {
  // docs/examples/example.msp: 2 spectra, 7 peaks
  MSDataset dataset = io::load_ms_dataset("example.msp");   // format from extension

  for (int64_t i = 0; i < dataset.size(); ++i) {
    SpectrumRecord record = dataset[i];
    std::cout << to_display_string(record["Name"]) << ": "    // MassSpecID0000001: 4 peaks
              << record.peaks().size() << " peaks\n";        // MassSpecID0000002: 3 peaks
  }

  MSDataset second = dataset.select({1});      // a view: shares data with `dataset`
  second.set_column("Checked", Value(true));

  dataset.save("example.msds");                // MSDS
  io::write_mgf(dataset, "example.mgf");       // or MSP / TSV / CSV

  std::shared_ptr<arrow::Table> table = dataset.metadata();   // metadata as an Arrow table
}
```

## Python, Java, C#

```python
import msentity as ms                      # bindings/python
ds = ms.load("example.msp"); ds["Name"]; ds[0].mz; ds.save("example.msds")
```

```java
try (Dataset ds = Dataset.load(Path.of("example.msp"))) { ds.column("Name"); ds.save(Path.of("example.msds")); }
```

```csharp
using var ds = Dataset.Load("example.msp"); var names = ds["Name"]; ds.Save("example.msds");
```

See [docs/bindings.md](docs/bindings.md) for setup and the full API of each language.

## Command-line tool

```console
msentity-tool dump input.msp [--type msp] [--spec-id-prefix SP] [--infer-types]   # JSON
msentity-tool convert input.msp output.msds [--no-save-view] [--infer-types]       # any → any
msentity-tool meta data.msds
```

## Tests

* `tests/test_*.cpp` (GoogleTest): data model, every reader and writer, the C
  API, and format details (chunk sizes, JSON and number formatting, Parquet
  layout).
* `bindings/*/tests`: Python, Java, and C# suites, run by `ctest` when the
  toolchain is installed (Python + NumPy, Maven + JDK ≥ 22, .NET SDK ≥ 8).

## Status

Done: data model, all file formats, C API, and Python/Java/C# bindings.
Planned: binary packages (wheel, Maven, NuGet) — see the
[roadmap](docs/architecture.md#6-roadmap).

## License

MIT — see [LICENSE](LICENSE).
