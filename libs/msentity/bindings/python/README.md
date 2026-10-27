# msentity (Python)

Python binding of the msentity C++ library. Guide: [docs/bindings.md](../../docs/bindings.md).

```python
import msentity as ms

ds = ms.load("example.msp")
ds["Name"]          # ['MassSpecID0000001', 'MassSpecID0000002']
ds[0].mz            # array([ 91.0542, 125.0233, 185.0961, 246.1125])
ds.save("example.msds")
```

| | |
|---|---|
| Requires | Python ≥ 3.10, NumPy; pyarrow + pandas for `to_arrow()` / `to_pandas()` |
| Native library | `MSENTITY_C_LIBRARY`, else `msentity/lib/`, else `<repo>/build/` |
| Files | `msentity/__init__.py` (API), `msentity/_native.py` (ctypes declarations) |

Development:

```console
cmake --build ../../build                     # builds libmsentity_c
python -m unittest discover -s tests -v       # from bindings/python
pip install -e .                              # optional
```

A distributable wheel copies `libmsentity_c` and its dependencies into
`msentity/lib/` before `python -m build`.
