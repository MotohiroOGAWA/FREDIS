# How scores are calculated

msentity-similarity compares spectra by **binned cosine similarity**. This page
defines the calculation precisely and works through it on real data, so results
can be checked by hand or reproduced by another implementation.

## 1. Binning

Each spectrum is turned into a sparse vector:

1. Every peak goes into the bin `floor(mz / bin_width)` (default `bin_width` = 0.01).
2. A negative intensity is replaced by 0.
3. Each intensity is raised to the power `intensity_exponent` (default 1).
4. Intensities that fall into the same bin are added.

For the two spectra of [examples/example.msp](examples/example.msp), with the defaults:

| m/z | bin | spectrum 0 | spectrum 1 |
|---|---|---|---|
| 91.0542 | 9105 | 245 | 99 |
| 125.0233 | 12502 | 1000 | 281 |
| 185.0961 | 18509 | 349 | — |
| 246.1125 | 24611 | 735 | 1000 |

## 2. Cosine (`method = "cosine"`)

```text
            Σ  a[b] · r[b]            (over bins present in both)
cosine = ─────────────────────
           ‖a‖ · ‖r‖                  (‖x‖ = sqrt of Σ x[b]² over all bins of x)
```

For spectrum 0 (a) vs spectrum 1 (r):

```text
a·r = 245·99 + 1000·281 + 735·1000 = 1 040 255
‖a‖ = sqrt(245² + 1000² + 349² + 735²) = 1312.2694
‖r‖ = sqrt(99² + 281² + 1000²)         = 1043.4376
cosine = 1 040 255 / (1312.2694 · 1043.4376) = 0.75971
```

The score is symmetric: spectrum 1 vs spectrum 0 is also 0.75971.

## 3. Reverse cosine (`method = "reverse_cosine"`)

The same, except that the **query** norm counts only the query peaks whose bin
also occurs in the reference. Extra query peaks (for example impurities or
fragments missing from the library spectrum) then do not lower the score:

```text
                    a·r
reverse_cosine = ────────────────
                  ‖a matched‖ · ‖r‖
```

Spectrum 0 as query vs spectrum 1 as reference: the 185.0961 peak has no partner,
so

```text
‖a matched‖ = sqrt(245² + 1000² + 735²) = 1265.0099
reverse_cosine = 1 040 255 / (1265.0099 · 1043.4376) = 0.78810
```

Spectrum 1 as query vs spectrum 0 as reference: every peak of spectrum 1 is
matched, so the score equals the cosine, 0.75971. Reverse cosine is therefore
**not** symmetric.

## 4. Edge cases and output

* An empty spectrum, or one whose intensities are all 0, scores 0 against anything.
* Scores are clipped to [0, 1] (to absorb rounding) and returned as 32-bit floats.
* `bin_width` and `intensity_exponent` must be positive; otherwise the call fails
  with an invalid-argument error.
* `intensity_exponent = 0.5` (square root) reduces the weight of dominant peaks.
  For the example spectra it raises the cosine from 0.75971 to 0.86077.

## 5. Calculation modes

| Mode | Pairs compared | Result rows |
|---|---|---|
| `pair_scores` | the pairs you list: (ds1[i1[k]], ds2[i2[k]]) | one score per pair (array) |
| `all_pairs_matrix` | every spectrum of ds1 with every spectrum of ds2 | len(ds1) × len(ds2) matrix |
| `library_search` | every query × every reference spectrum | one row per pair with score ≥ threshold, ordered by query then reference |
| `similarity_by_key` | spectra whose key values are equal (`key1` in ds1 = `key2` in ds2) | one row per matching key, in ds1 order |

Notes on `similarity_by_key`:

* Each key value must occur at most once in each dataset; duplicates are an error
  (the message lists up to five of them).
* Rows whose key is missing (null) are skipped.
* Integer and floating keys compare by value (`3` matches `3.0`); strings compare
  exactly; a string never matches a number.

`library_search` and `all_pairs_matrix` use several threads when there is
enough work (from 2048 pairs). The order of the output does not depend on the
number of threads.
