# Installation

msentity-similarity is built on [msentity](https://github.com/MotohiroOGAWA/msentity).
Binary packages are **not published yet**; build from source as described here.
Each language needs the msentity binding of that language as well.

## 1. Build the libraries

Requirements are those of msentity: C++20, CMake ≥ 3.21, HDF5 (C + HL),
Apache Arrow & Parquet C++, nlohmann/json, and GoogleTest for the tests
(Debian/Ubuntu packages are listed in msentity's README).

```console
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

CMake downloads msentity at the commit pinned in `CMakeLists.txt`. To use a
local checkout instead (for example when changing both):

```console
cmake -S . -B build -G Ninja -DFETCHCONTENT_SOURCE_DIR_MSENTITY=../msentity
```

The build produces:

| File | Used by |
|---|---|
| `build/libmsentity_similarity.a` | C++ (`msentity::similarity`) |
| `build/libmsentity_similarity_c.so` (`.dylib`, `.dll`) | C, Python, Java, C# |
| `build/_deps/msentity-build/libmsentity_c.so` | the msentity bindings |
| `build/mssim-tool` | command line ([api.md §8](api.md#8-command-line-tool)) |

Options:

| Option | Default | Meaning |
|---|---|---|
| `MSENTITY_SIMILARITY_BUILD_TESTS` | ON at top level | C++ and C API tests |
| `MSENTITY_SIMILARITY_BUILD_TOOLS` | ON at top level | `mssim-tool` |
| `MSENTITY_SIMILARITY_BUILD_C_API` | ON | `libmsentity_similarity_c` for the bindings |
| `MSENTITY_SIMILARITY_TEST_BINDINGS` | ON | add the Python/Java/C# tests to ctest when the toolchain is found |
| `MSENTITY_STATIC_DEPS` | OFF | link Arrow/Parquet/HDF5 statically (self-contained C library for packaging) |

## 2. C++

From another CMake project:

```cmake
include(FetchContent)
FetchContent_Declare(msentity_similarity
  GIT_REPOSITORY <repository URL or local path>
  GIT_TAG        <commit>)
FetchContent_MakeAvailable(msentity_similarity)

target_link_libraries(app PRIVATE msentity::similarity)   # also brings msentity::msentity
```

Tests, tools, and binding tests are off when msentity-similarity is not the
top-level project.

## 3. Python

```console
pip install <msentity>/bindings/python            # the msentity package
pip install bindings/python                       # msentity_similarity
export MSENTITY_C_LIBRARY=$PWD/build/_deps/msentity-build/libmsentity_c.so
export MSSIM_C_LIBRARY=$PWD/build/libmsentity_similarity_c.so
python -c "import msentity_similarity as mss; print(mss.__version__)"   # 0.1.0
```

Instead of the environment variables, the libraries can be copied into
`msentity/lib/` and `msentity_similarity/lib/` of the packages before
installing; a wheel built that way needs nothing else.

## 4. Java

```console
(cd <msentity>/bindings/java && mvn install -DskipTests)   # io.github.motohiroogawa:msentity
(cd bindings/java && mvn install -DskipTests)              # …:msentity-similarity
```

```xml
<dependency>
  <groupId>io.github.motohiroogawa</groupId>
  <artifactId>msentity-similarity</artifactId>
  <version>0.1.0</version>
</dependency>
```

Run with `--enable-native-access=ALL-UNNAMED -Dmssim.library=<…/libmsentity_similarity_c.so>
-Dmsentity.library=<…/libmsentity_c.so>`. Libraries placed in
`bindings/natives/<platform>/` (`linux-x64`, `win-x64`, `osx-arm64`) before
`mvn install` are bundled in the JAR and found automatically.

## 5. C#

```console
dotnet pack bindings/csharp/MsEntity.Similarity -c Release -o packages
```

The project references the MsEntity project of a sibling msentity checkout
(`../msentity`) when it exists, otherwise the `MsEntity` NuGet package
(`-p:UseMsEntityPackage=true` forces the package). Set `MSSIM_C_LIBRARY` and
`MSENTITY_C_LIBRARY`, or put the libraries next to the application. Libraries
placed in `bindings/natives/<rid>/` before `dotnet pack` are packed as
`runtimes/<rid>/native/`.

## 6. C

Use `include/msentity/similarity/c/msentity_similarity_c.h` (it includes
msentity's `msentity/c/msentity_c.h`) and link `libmsentity_similarity_c`.
Datasets are created with msentity's C API and passed as `.msds` bytes
(`msentity_to_msds_bytes`).
