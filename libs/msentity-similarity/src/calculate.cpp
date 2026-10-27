#include "msentity/similarity/calculate.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <ctime>

namespace msentity::similarity {

namespace {

using json = nlohmann::ordered_json;

// ISO 8601 UTC time with microseconds, e.g. 2026-10-27T01:23:45.678901+00:00.
std::string now_utc() {
  auto now = std::chrono::system_clock::now();
  auto seconds = std::chrono::system_clock::to_time_t(now);
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() %
                1'000'000;
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &seconds);
#else
  gmtime_r(&seconds, &utc);
#endif
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%06lld+00:00",
                utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min,
                utc.tm_sec, static_cast<long long>(micros));
  return buffer;
}

json source(const MSDataset& dataset, const std::string& path, const std::string& role) {
  json attributes = json::object();
  for (const auto& [key, value] : dataset.attributes()) attributes[key] = value;
  return {{"role", role},
          {"path", path},
          {"n_rows", dataset.size()},
          {"description", dataset.description()},
          {"attributes", attributes},
          {"tags", dataset.tags()}};
}

json parameters(const Options& options) {
  return {{"method", method_name(options.method)},
          {"bin_width", options.bin_width},
          {"intensity_exponent", options.intensity_exponent}};
}

}  // namespace

SimilarityDataset calculate_similarity(const MSDataset& ds1, const MSDataset& ds2,
                                       const std::string& key1, const std::string& key2,
                                       const Options& options, bool include_matched_data,
                                       const std::string& source1, const std::string& source2) {
  auto table = similarity_by_key(ds1, ds2, key1, key2, options);
  json params = {{"key1", key1}, {"key2", key2}};
  params.update(parameters(options));
  json metadata = {
      {"created_at", now_utc()},
      {"mode", "by_key"},
      {"algorithm", "binned_" + method_name(options.method) + "_similarity_by_key"},
      {"parameters", params},
      {"matching", "unique_intersection"},
      {"sources", json::array({source(ds1, source1, "first"), source(ds2, source2, "second")})},
      {"row_count", table->num_rows()},
      {"index_basis", "zero-based input dataset position"},
      {"input_scope", "entire in-memory datasets (including unsaved edits)"},
      {"matched_data", include_matched_data ? "embedded" : "lightweight"},
  };
  if (include_matched_data) {
    return SimilarityDataset::with_matched_data(table, metadata.dump(), ds1, ds2);
  }
  return SimilarityDataset(table, metadata.dump());
}

SimilarityDataset calculate_library_search(const MSDataset& query, const MSDataset& reference,
                                           double threshold, const Options& options,
                                           bool include_matched_data,
                                           const std::string& query_source,
                                           const std::string& reference_source,
                                           const ProgressCallback& progress) {
  auto table = library_search(query, reference, threshold, options, progress);
  json params = {{"threshold", threshold}};
  params.update(parameters(options));
  json metadata = {
      {"created_at", now_utc()},
      {"mode", "library_search"},
      {"algorithm", "binned_" + method_name(options.method) + "_similarity_library_search"},
      {"parameters", params},
      {"matching", "all_pairs_at_or_above_threshold"},
      {"sources", json::array({source(query, query_source, "query"),
                               source(reference, reference_source, "reference")})},
      {"row_count", table->num_rows()},
      {"candidate_pair_count", query.size() * reference.size()},
      {"index_basis", "zero-based input dataset position"},
      {"input_scope", "entire in-memory datasets (including unsaved edits)"},
      {"matched_data", include_matched_data ? "embedded" : "lightweight"},
  };
  if (include_matched_data) {
    return SimilarityDataset::with_matched_data(table, metadata.dump(), query, reference);
  }
  return SimilarityDataset(table, metadata.dump());
}

}  // namespace msentity::similarity
