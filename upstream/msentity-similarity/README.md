# msentity-similarity

Spectrum similarity for [msentity](https://github.com/MotohiroOGAWA/msentity)
datasets, and the `.mssim` file format for the results. Like msentity, it is a
C++ library with a C API, used from Python, Java, and C# through the same
shared library.

| | |
|---|---|
| Scores | binned cosine and reverse cosine (bin width, intensity exponent), multithreaded |
| Modes | chosen pairs, all-pairs matrix, library search with threshold, pairs by metadata key |
| Results | `SimilarityDataset`: Arrow table + calculation metadata + optionally the compared spectra; filter, sort, statistics |
| Format | `.mssim` (HDF5 + Parquet + embedded `.msds`) — [specification](docs/format/mssim.md) |
| Languages | C++, C, Python (`msentity_similarity`), Java (`io.github.motohiroogawa.msentity.similarity`), C# (`MsEntity.Similarity`) |
| Depends on | msentity (fetched by CMake), C++20, CMake ≥ 3.21 |

## Example

```python
import msentity as ms
import msentity_similarity as mss

ds = ms.load("example.msp")                       # 2 spectra
hits = mss.library_search(ds, ds, threshold=0.7)
hits.to_pandas()
#    index1  index2  cosine_similarity  data_index1  data_index2
# 0       0       0           1.000000            0            0
# 1       0       1           0.759714            0            1
# 2       1       0           0.759714            1            0
# 3       1       1           1.000000            1            1
query, reference = hits.match_records(1)          # the two compared spectra
hits.save("hits.mssim")
```

The same in C++:

```cpp
#include <msentity/msentity.hpp>
#include <msentity/similarity/similarity.hpp>

auto ds = msentity::io::load_ms_dataset("example.msp");
auto hits = msentity::similarity::calculate_library_search(ds, ds, 0.7);
hits.save("hits.mssim");
```

and on the command line:

```console
$ mssim-tool search example.msp example.msp hits.mssim --threshold 0.7
4 pairs written to hits.mssim
```

## Building

```console
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure    # C++, C API, and binding tests
```

See [docs/install.md](docs/install.md) for every language.

## Documentation

* [Installation](docs/install.md)
* [How scores are calculated](docs/similarity.md)
* [C++ API and mssim-tool](docs/api.md)
* [Python, Java, C#, and C](docs/bindings.md)
* [MSSIM file format](docs/format/mssim.md)

## Relation to msentity

msentity-similarity depends on msentity, never the other way round. Datasets
cross between the two C libraries as `.msds` bytes, so each can be built,
packaged, and versioned on its own.

## License

MIT — see [LICENSE](LICENSE).
