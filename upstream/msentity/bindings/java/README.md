# msentity (Java)

Java binding of the msentity C++ library, using the Foreign Function & Memory
API (no JNI code). Guide: [docs/bindings.md](../../docs/bindings.md).

```java
try (Dataset ds = Dataset.load(Path.of("example.msp"))) {
  ds.column("Name");          // [MassSpecID0000001, MassSpecID0000002]
  ds.spectrum(0).mz();        // [91.0542, 125.0233, 185.0961, 246.1125]
  ds.save(Path.of("example.msds"));
}
```

| | |
|---|---|
| Requires | Java ≥ 22; run with `--enable-native-access=ALL-UNNAMED` |
| Coordinates | `io.github.motohiroogawa:msentity:0.1.0` |
| Native library | `-Dmsentity.library=…`, else `MSENTITY_C_LIBRARY`, else the system library path |
| Files | `Dataset.java` (API), `Native.java` (downcall handles), `MsEntityException.java` |

Development:

```console
cmake --build ../../build      # builds libmsentity_c
mvn test                       # uses ../../build/libmsentity_c.so
```
