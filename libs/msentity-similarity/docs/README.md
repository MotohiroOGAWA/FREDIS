# Documentation

| Read this | To learn |
|---|---|
| [install.md](install.md) | how to build and install msentity-similarity for C++, Python, Java, C#, or C |
| [similarity.md](similarity.md) | how the scores are calculated, worked through on real spectra |
| [api.md](api.md) | how to use the C++ library and `mssim-tool` — every example with its real result |
| [bindings.md](bindings.md) | how to use it from Python, Java, C#, and C |
| [format/mssim.md](format/mssim.md) | what is inside an `.mssim` file, field by field, and the rules for writing one |

Example files:

| File | Content |
|---|---|
| [examples/example.msp](examples/example.msp) | 2 spectra (4 + 3 peaks); the same file as in msentity |
| [examples/example.mssim](examples/example.mssim) | `mssim-tool search example.msp example.msp example.mssim --threshold 0.7`: 4 pairs, spectra embedded |
