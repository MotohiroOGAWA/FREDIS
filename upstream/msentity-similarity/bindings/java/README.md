# msentity-similarity (Java)

Java binding of msentity-similarity, using the Foreign Function & Memory API
(no JNI code). Guide: [docs/bindings.md](../../docs/bindings.md#java).

```java
try (Dataset ds = Dataset.load(Path.of("example.msp"));
     SimilarityDataset hits = Similarity.librarySearch(ds, ds, 0.7)) {
  hits.column("cosine_similarity");   // [1.0, 0.7597143650054932, 0.7597143650054932, 1.0]
  hits.save(Path.of("hits.mssim"));
}
```

| | |
|---|---|
| Requires | Java ≥ 22; run with `--enable-native-access=ALL-UNNAMED`; the `io.github.motohiroogawa:msentity` artifact |
| Coordinates | `io.github.motohiroogawa:msentity-similarity:0.1.0` |
| Native library | `-Dmssim.library=…`, else `MSSIM_C_LIBRARY`, else bundled in the JAR, else the system path |
| Files | `Similarity.java` (calculations), `SimilarityDataset.java` (results), `Options.java`, `Native.java`, `SimilarityException.java` |

Development: `mvn install -DskipTests` in msentity's `bindings/java`, then
`mvn test` here (uses `../../build/libmsentity_similarity_c.so` and
`../../../msentity/build/libmsentity_c.so` when they exist).
