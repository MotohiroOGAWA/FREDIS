#pragma once

#include <string>

#include <msentity/similarity/calculation.hpp>
#include <msentity/similarity/similarity_dataset.hpp>

namespace msentity::similarity {

// Calculations that produce a complete SimilarityDataset: the result table,
// a JSON description of the calculation (method, parameters, inputs,
// creation time, row counts; see docs/format/mssim.md §3), and optionally the
// matched spectra.

// Pair spectra by equal metadata keys (similarity_by_key) and score them.
SimilarityDataset calculate_similarity(const MSDataset& ds1, const MSDataset& ds2,
                                       const std::string& key1 = "SpecID",
                                       const std::string& key2 = "SpecID",
                                       const Options& options = {},
                                       bool include_matched_data = false,
                                       const std::string& source1 = "",
                                       const std::string& source2 = "");

// Search every query spectrum against a reference library (library_search).
SimilarityDataset calculate_library_search(const MSDataset& query, const MSDataset& reference,
                                           double threshold = 0.8, const Options& options = {},
                                           bool include_matched_data = true,
                                           const std::string& query_source = "",
                                           const std::string& reference_source = "",
                                           const ProgressCallback& progress = {});

}  // namespace msentity::similarity
