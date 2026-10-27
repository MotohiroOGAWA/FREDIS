# Documentation

| Read this | To learn |
|---|---|
| [install.md](install.md) | how to install msentity for C++, Python, Java, C#, or C |
| [api.md](api.md) | how to use the C++ library — every example with its real result |
| [bindings.md](bindings.md) | how to use it from Python, Java, and C# (and the C API) |
| [format/msds.md](format/msds.md) | what is inside an `.msds` file, field by field, and the rules for writing one |
| [format/text-formats.md](format/text-formats.md) | how MSP, MGF, TSV, and CSV are read and written |
| [architecture.md](architecture.md) | how the library is organized, how data sits in memory, planned language bindings |

Example files used throughout:

| File | Content |
|---|---|
| [examples/plain.msp](examples/plain.msp) | 2 spectra (4 + 3 peaks), 5 metadata keys — an ordinary MSP file |
| [examples/example.msp](examples/example.msp) | the same spectra with a `formula` value on some peaks (per-peak metadata) |
| [examples/example.msds](examples/example.msds) | the same data converted with `msentity-tool convert example.msp example.msds` |
