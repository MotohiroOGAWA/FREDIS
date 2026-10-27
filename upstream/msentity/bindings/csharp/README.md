# MsEntity (C#)

.NET binding of the msentity C++ library (P/Invoke). Guide:
[docs/bindings.md](../../docs/bindings.md).

```csharp
using var ds = Dataset.Load("example.msp");
var names = ds["Name"];               // [MassSpecID0000001, MassSpecID0000002]
var mz = ds.GetSpectrum(0).Mz;        // [91.0542, 125.0233, 185.0961, 246.1125]
ds.Save("example.msds");
```

| | |
|---|---|
| Requires | .NET ≥ 8 (the library targets net8.0; the tests run on net10.0) |
| Native library | `MSENTITY_C_LIBRARY`, else the application directory / `runtimes/<rid>/native/`, else the system library path |
| Files | `MsEntity/Dataset.cs` (API), `MsEntity/NativeMethods.cs` (P/Invoke), `MsEntity/MsEntityException.cs` |

Development:

```console
cmake --build ../../build          # builds libmsentity_c
dotnet test MsEntity.slnx          # copies ../../build/libmsentity_c.so next to the tests
```
