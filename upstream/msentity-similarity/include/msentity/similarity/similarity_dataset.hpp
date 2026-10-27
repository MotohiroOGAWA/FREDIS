#pragma once

#include <arrow/table.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <msentity/msentity.hpp>

namespace msentity::similarity {

// Summary of the cosine_similarity column (NaN when there are no rows).
struct ScoreSummary {
  int64_t count = 0;
  double mean = 0, q1 = 0, median = 0, q3 = 0, min = 0, max = 0;
};

// Score frequencies over [0, 1] (the last bin includes 1.0).
struct Histogram {
  std::vector<int64_t> counts;  // bins
  std::vector<double> edges;    // bins + 1
};

// A similarity result: a table of scored spectrum pairs, a JSON description
// of the calculation, and optionally the matched spectra themselves. This is
// the content of an .mssim file (docs/format/mssim.md).
//
// Invariants (checked by every constructor):
//   * the table has index1, index2, and a numeric cosine_similarity column
//     whose values are finite and within [0, 1];
//   * metadata_json is a JSON object;
//   * with matched data: data_index1/data_index2 columns refer to rows of the
//     matched datasets, and the source indices have one entry per row.
class SimilarityDataset {
 public:
  using DatasetPair = std::pair<MSDataset, MSDataset>;
  using IndexPair = std::pair<std::vector<int64_t>, std::vector<int64_t>>;

  explicit SimilarityDataset(std::shared_ptr<arrow::Table> table,
                             std::string metadata_json = "{}",
                             std::optional<DatasetPair> matched_datasets = std::nullopt,
                             std::optional<IndexPair> matched_source_indices = std::nullopt);

  // Attach every spectrum referenced by index1/index2 once: adds
  // data_index1/data_index2 and keeps the original positions as source
  // indices.
  static SimilarityDataset with_matched_data(std::shared_ptr<arrow::Table> table,
                                             std::string metadata_json, const MSDataset& first,
                                             const MSDataset& second);

  // ---- content ----------------------------------------------------------------
  int64_t size() const;
  const std::shared_ptr<arrow::Table>& table() const { return table_; }
  std::vector<std::string> columns() const;
  std::vector<Value> column_values(const std::string& name) const;
  const std::string& metadata_json() const { return metadata_json_; }
  void set_metadata_json(std::string json);

  bool has_matched_data() const { return matched_datasets_.has_value(); }
  const std::optional<DatasetPair>& matched_datasets() const { return matched_datasets_; }
  const std::optional<IndexPair>& matched_source_indices() const { return source_indices_; }
  // The two matched spectra of a row, as one-spectrum views.
  DatasetPair match_records(int64_t row) const;

  // ---- operations ------------------------------------------------------------------
  // Rows where mask is true; metadata gets row_count and source_row_count.
  SimilarityDataset filter(const std::vector<bool>& mask) const;
  // Stable sort by one or more columns; missing values last.
  SimilarityDataset sort_by(const std::vector<std::string>& columns,
                            const std::vector<bool>& ascending = {}) const;
  ScoreSummary describe_scores() const;
  Histogram histogram(int bins = 20) const;
  // Write the table as .parquet, .csv, or .tsv (by extension).
  void export_table(const std::string& path) const;

  // ---- files ----------------------------------------------------------------------
  static SimilarityDataset load(const std::string& path);
  // Atomic replace; embedded spectra are compacted to those the table uses.
  void save(const std::string& path) const;

 private:
  void validate() const;

  std::shared_ptr<arrow::Table> table_;
  std::string metadata_json_;
  std::optional<DatasetPair> matched_datasets_;
  std::optional<IndexPair> source_indices_;
};

}  // namespace msentity::similarity
