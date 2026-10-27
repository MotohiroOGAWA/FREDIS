#pragma once

#include <arrow/table.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "msentity/core/peak_series.hpp"
#include "msentity/value.hpp"

namespace msentity {

// Dataset-level attributes. Insertion order is preserved so that files
// round-trip unchanged.
using Attributes = std::vector<std::pair<std::string, std::string>>;

namespace detail {
struct TableRef {
  std::shared_ptr<arrow::Table> table;
};
}  // namespace detail

class MSDataset;

// One spectrum of a dataset view: metadata values plus its peaks.
class SpectrumRecord {
 public:
  Value operator[](const std::string& column) const { return value(column); }
  Value value(const std::string& column) const;
  bool contains(const std::string& column) const;
  void set(const std::string& column, const Value& value);
  const std::vector<std::string>& columns() const;
  int64_t n_peaks() const;
  Spectrum peaks() const;
  int64_t position() const { return position_; }

 private:
  friend class MSDataset;
  SpectrumRecord(MSDataset* dataset, const MSDataset* const_dataset, int64_t position)
      : dataset_(dataset), const_dataset_(const_dataset), position_(position) {}
  MSDataset* dataset_;
  const MSDataset* const_dataset_;
  int64_t position_;
};

// Spectrum-level metadata (an Arrow table, one row per spectrum) combined
// with a PeakSeries. Subsets are views sharing
// the underlying metadata table and peak storage; copy() materializes.
class MSDataset {
 public:
  MSDataset();
  MSDataset(std::shared_ptr<arrow::Table> spectrum_metadata, PeakSeries peaks,
            std::optional<std::vector<std::string>> columns = std::nullopt,
            std::string description = "", Attributes attributes = {},
            std::vector<std::string> tags = {});

  // --- shape ---------------------------------------------------------------
  int64_t size() const { return peaks_.size(); }
  int64_t n_rows() const { return size(); }
  int64_t n_columns() const { return static_cast<int64_t>(columns_.size()); }
  int64_t n_peaks_total() const { return peaks_.n_peaks_total(); }

  // --- columns -------------------------------------------------------------
  const std::vector<std::string>& columns() const { return columns_; }
  void set_columns(std::vector<std::string> columns);
  bool has_column(const std::string& name) const;

  // --- dataset-level metadata ------------------------------------------------
  const std::string& description() const { return description_; }
  void set_description(std::string description) { description_ = std::move(description); }
  const Attributes& attributes() const { return attributes_; }
  void set_attributes(Attributes attributes);
  std::optional<std::string> attribute(const std::string& key) const;
  void set_attribute(const std::string& key, const std::string& value);
  bool remove_attribute(const std::string& key);
  bool has_attribute(const std::string& key) const;
  void clear_attributes() { attributes_.clear(); }
  const std::vector<std::string>& tags() const { return tags_; }
  void set_tags(std::vector<std::string> tags);
  bool add_tag(const std::string& tag);
  bool remove_tag(const std::string& tag);
  bool has_tag(const std::string& tag) const;
  void clear_tags() { tags_.clear(); }

  // --- spectrum metadata -----------------------------------------------------
  // Visible rows and columns as a new table (single chunk per column).
  std::shared_ptr<arrow::Table> metadata() const;
  std::shared_ptr<arrow::ChunkedArray> column(const std::string& name) const;
  std::vector<Value> column_values(const std::string& name) const;
  Value value(int64_t position, const std::string& column) const;
  // Add or update a column for the visible rows. A new column is null for
  // rows outside the view.
  void set_column(const std::string& name, const std::vector<Value>& values);
  void set_column(const std::string& name, const Value& value);
  void set_value(int64_t position, const std::string& column, const Value& value);

  // --- records and views -------------------------------------------------------
  SpectrumRecord operator[](int64_t position);
  SpectrumRecord operator[](int64_t position) const;
  SpectrumRecord record(int64_t position);
  SpectrumRecord record(int64_t position) const;
  MSDataset select(const std::vector<int64_t>& positions) const;
  MSDataset slice(int64_t start, int64_t stop) const;
  MSDataset filter(const std::vector<bool>& mask) const;
  MSDataset copy() const;
  MSDataset& reset_view(bool reset_columns = false);
  MSDataset full_view(bool reset_columns = false) const;

  // Stable sort by a visible column; nulls are placed last.
  MSDataset sort_by(const std::string& column, bool ascending = true) const;

  static MSDataset concat(const std::vector<MSDataset>& datasets, std::string description = "",
                          Attributes attributes = {}, std::vector<std::string> tags = {});

  // --- peaks -------------------------------------------------------------------
  const PeakSeries& peaks() const { return peaks_; }
  PeakSeries& peaks() { return peaks_; }

  // --- persistence (see msentity/io/msds.hpp) -----------------------------------
  void save(const std::string& path, bool save_view = true, bool append = false) const;
  static MSDataset load(const std::string& path, bool load_peak_metadata = true);

  // Underlying metadata table with every spectrum and column.
  std::shared_ptr<arrow::Table> spectrum_metadata_ref() const { return metadata_->table; }

 private:
  friend class SpectrumRecord;
  MSDataset(std::shared_ptr<detail::TableRef> metadata, PeakSeries peaks,
            std::vector<std::string> columns, std::string description, Attributes attributes,
            std::vector<std::string> tags);
  MSDataset with_peaks(PeakSeries peaks) const;

  std::shared_ptr<detail::TableRef> metadata_;
  PeakSeries peaks_;
  std::vector<std::string> columns_;
  std::string description_;
  Attributes attributes_;
  std::vector<std::string> tags_;
};

}  // namespace msentity
