#pragma once

#include <arrow/table.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <msentity/msentity.hpp>

namespace msentity::similarity {

// Binned spectrum similarity (docs/similarity.md).
//
// Each spectrum becomes a sparse vector: peak m/z is binned as
// floor(mz / bin_width), negative intensities become 0, intensities are
// raised to intensity_exponent and summed per bin.
//   cosine          dot / sqrt(|a|² |b|²)
//   reverse_cosine  dot / sqrt(|a matched|² |b|²) — peaks of the first
//                   (query) spectrum without a partner in the second do not
//                   count, so noise in the query is tolerated.
// Scores are clipped to [0, 1]; a zero denominator gives 0.
enum class Method { Cosine, ReverseCosine };

// "cosine" / "reverse_cosine".
std::string method_name(Method method);
Method parse_method(const std::string& name);

struct Options {
  Method method = Method::Cosine;
  double bin_width = 0.01;           // m/z bin width (> 0)
  double intensity_exponent = 1.0;   // e.g. 0.5 for square-root intensities (> 0)
  int threads = 0;                   // worker threads; 0 = all hardware threads
};

// (completed pairs, total pairs)
using ProgressCallback = std::function<void(int64_t, int64_t)>;

// Score of ds1[index1[k]] vs ds2[index2[k]] for every k (positions in the
// dataset views).
std::vector<float> pair_scores(const MSDataset& ds1, const std::vector<int64_t>& index1,
                               const MSDataset& ds2, const std::vector<int64_t>& index2,
                               const Options& options = {});

// Every spectrum of ds1 against every spectrum of ds2; row-major
// (size(ds1) × size(ds2)).
std::vector<float> all_pairs_matrix(const MSDataset& ds1, const MSDataset& ds2,
                                    const Options& options = {});

// Pair spectra whose metadata keys are equal (ds1[key1] == ds2[key2]) and
// score them. Missing keys are skipped; a key value that occurs twice in one
// dataset is an error. Columns: key1 [, key2 if different], index1, index2,
// cosine_similarity (float32); rows in ds1 order.
std::shared_ptr<arrow::Table> similarity_by_key(const MSDataset& ds1, const MSDataset& ds2,
                                                const std::string& key1 = "SpecID",
                                                const std::string& key2 = "SpecID",
                                                const Options& options = {});

// Score every query spectrum against every reference spectrum and keep the
// pairs with score >= threshold. Columns: index1 (query), index2
// (reference), cosine_similarity (float32); rows ordered by index1, index2.
std::shared_ptr<arrow::Table> library_search(const MSDataset& query, const MSDataset& reference,
                                             double threshold = 0.8,
                                             const Options& options = {},
                                             const ProgressCallback& progress = {});

}  // namespace msentity::similarity
