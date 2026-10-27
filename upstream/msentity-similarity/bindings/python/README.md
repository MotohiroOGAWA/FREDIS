# msentity-similarity (Python)

Python binding of msentity-similarity. Guide: [docs/bindings.md](../../docs/bindings.md#python).

```python
import msentity as ms
import msentity_similarity as mss

ds = ms.load("example.msp")
hits = mss.library_search(ds, ds, threshold=0.7)
hits["cosine_similarity"]   # [1.0, 0.7597143650054932, 0.7597143650054932, 1.0]
hits.save("hits.mssim")
```

| | |
|---|---|
| Requires | Python ≥ 3.10, NumPy, the `msentity` package; pyarrow + pandas for `to_arrow()` / `to_pandas()` |
| Native library | `MSSIM_C_LIBRARY`, else `msentity_similarity/lib/`, else `<repo>/build/` |
| Files | `msentity_similarity/__init__.py` (API), `msentity_similarity/_native.py` (ctypes declarations) |

Development (from this directory, with msentity checked out next to this repository):

```console
cmake --build ../../build
PYTHONPATH=.:../../../msentity/bindings/python \
MSENTITY_C_LIBRARY=../../build/_deps/msentity-build/libmsentity_c.so \
  python -m unittest discover -s tests -v
```
