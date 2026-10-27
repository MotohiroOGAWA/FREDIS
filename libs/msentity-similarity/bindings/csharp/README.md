# msentity-similarity (C#)

.NET binding of msentity-similarity. Guide: [docs/bindings.md](../../docs/bindings.md#c).

```csharp
using var ds = Dataset.Load("example.msp");
using var hits = SpectrumSimilarity.LibrarySearch(ds, ds, threshold: 0.7);
hits["cosine_similarity"];   // 1.0, 0.7597143650054932, 0.7597143650054932, 1.0
hits.Save("hits.mssim");
```

| | |
|---|---|
| Requires | .NET ≥ 8; the `MsEntity` package (or a sibling msentity checkout) |
| Package | `MsEntity.Similarity` 0.1.0 |
| Native library | `MSSIM_C_LIBRARY`, else the application directory / `runtimes/<rid>/native/`, else the system path |
| Files | `SpectrumSimilarity.cs` (calculations), `SimilarityDataset.cs` (results), `NativeMethods.cs`, `SimilarityException.cs` |

Development: `dotnet test MsEntity.Similarity.slnx` (copies the libraries of
`../../build` and `../../../msentity/build` next to the tests when they exist).
