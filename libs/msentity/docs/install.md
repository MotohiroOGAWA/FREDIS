# Installation

Every release on [GitHub Releases](https://github.com/MotohiroOGAWA/msentity/releases)
provides ready-to-use packages. They contain a self-contained native library
(Arrow, Parquet, and HDF5 are built in), so Python, Java, C#, and C users need
no C++ toolchain and no extra system packages.

| Platform | Python wheel tag | Java / C# / C platform |
|---|---|---|
| Linux x64 (glibc ≥ 2.35: Ubuntu 22.04+, Debian 12+, RHEL 10+, …) | `manylinux_2_35_x86_64` | `linux-x64` |
| Windows x64 | `win_amd64` | `win-x64` |
| macOS arm64 (Apple silicon, macOS ≥ 14) | `macosx_14_0_arm64` | `osx-arm64` |

In the commands below replace `v0.1.0` / `0.1.0` with the release you want.

---

## Python (≥ 3.10)

```console
pip install msentity --find-links https://github.com/MotohiroOGAWA/msentity/releases/expanded_assets/v0.1.0
pip install pyarrow pandas        # optional: Dataset.to_arrow() / to_pandas()
```

pip picks the wheel for your platform from the release. The wheel works with
every Python 3 version from 3.10 on (it uses ctypes, not a compiled module).

```python
import msentity as ms
ds = ms.load("example.msp")
```

## Java (≥ 22)

Through [JitPack](https://jitpack.io), which serves the JAR of the release:

Maven:

```xml
<repositories>
  <repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
  </repository>
</repositories>

<dependencies>
  <dependency>
    <groupId>com.github.MotohiroOGAWA</groupId>
    <artifactId>msentity</artifactId>
    <version>v0.1.0</version>
  </dependency>
</dependencies>
```

Gradle:

```groovy
repositories { maven { url 'https://jitpack.io' } }
dependencies { implementation 'com.github.MotohiroOGAWA:msentity:v0.1.0' }
```

Alternatively download `msentity-0.1.0.jar` from the release and put it on the
class path. Run with `--enable-native-access=ALL-UNNAMED`:

```console
java --enable-native-access=ALL-UNNAMED -cp app.jar:msentity-0.1.0.jar app.Main
```

The JAR contains the native library of every supported platform and loads the
right one automatically.

## C# (.NET ≥ 8)

Download `MsEntity.0.1.0.nupkg` from the release into a folder (for example
`~/nupkg`), then:

```console
dotnet add package MsEntity --version 0.1.0 --source ~/nupkg
```

or add the folder as a package source once (`nuget.config`):

```xml
<configuration>
  <packageSources>
    <add key="msentity" value="~/nupkg" />
  </packageSources>
</configuration>
```

The package contains `runtimes/<platform>/native/`, which .NET loads
automatically.

## C

Download `msentity-c-0.1.0-<platform>.zip`:

```text
msentity-c-0.1.0-linux-x64/
├── include/msentity/c/msentity_c.h
├── lib/libmsentity_c.so          (msentity_c.dll + msentity_c.lib on Windows,
│                                  libmsentity_c.dylib on macOS)
└── LICENSE
```

```console
cc app.c -I msentity-c-0.1.0-linux-x64/include -L msentity-c-0.1.0-linux-x64/lib -lmsentity_c
```

## C++

The C++ API exposes Arrow types, so C++ projects build msentity from source
against their own Arrow, Parquet, and HDF5.

### Dependencies

| | Debian / Ubuntu | vcpkg |
|---|---|---|
| Arrow + Parquet C++ | `libarrow-dev libparquet-dev` from the [Apache Arrow APT repository](https://arrow.apache.org/install/) | `arrow[parquet]` |
| HDF5 (C + high-level) | `libhdf5-dev` | `hdf5[hl,zlib]` |
| nlohmann/json | `nlohmann-json3-dev` | `nlohmann-json` |
| CMake ≥ 3.21, C++20 compiler | `cmake g++` | — |

With vcpkg, the repository's `vcpkg.json` installs them automatically:
`cmake -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake …`.

### Use it from your CMake project

```cmake
include(FetchContent)
FetchContent_Declare(msentity
  GIT_REPOSITORY https://github.com/MotohiroOGAWA/msentity.git
  GIT_TAG v0.1.0)
FetchContent_MakeAvailable(msentity)

target_link_libraries(app PRIVATE msentity::msentity)     # C++ API
# or msentity::msentity_c                                 # C API
```

### Or install it

```console
git clone --branch v0.1.0 https://github.com/MotohiroOGAWA/msentity.git
cmake -S msentity -B msentity/build -DCMAKE_BUILD_TYPE=Release
cmake --build msentity/build
cmake --install msentity/build --prefix /usr/local
```

```cmake
find_package(msentity REQUIRED)
target_link_libraries(app PRIVATE msentity::msentity)
```

---

## From a source checkout (development)

```console
git clone https://github.com/MotohiroOGAWA/msentity.git && cd msentity
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build                       # C++, C API, and every binding

pip install -e bindings/python               # uses build/libmsentity_c.so
(cd bindings/java && mvn install)            # uses build/libmsentity_c.so
dotnet add reference bindings/csharp/MsEntity/MsEntity.csproj   # set MSENTITY_C_LIBRARY
```

## How releases are made

Pushing a tag `v<version>` (equal to the version in `CMakeLists.txt`) runs
[`.github/workflows/release.yml`](../.github/workflows/release.yml):

1. builds `libmsentity_c` with Arrow, Parquet, and HDF5 linked statically
   (vcpkg) on Linux x64, Windows x64, and macOS arm64, and runs the C++ tests
   on each;
2. builds the wheel, JAR, and NuGet package, and tests each binding on each
   platform with the packaged library;
3. publishes everything as a GitHub Release.

The same workflow can be started from the Actions tab without a tag to test
the build without publishing.
