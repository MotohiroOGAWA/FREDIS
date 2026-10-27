#pragma once

#include <arrow/table.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "msentity/core/peak.hpp"
#include "msentity/value.hpp"

namespace msentity {

namespace detail {
// Storage shared by every view of the same peak series (the MSDS peaks
// group in memory).
struct PeakStorage {
  // Interleaved (m/z, intensity) pairs; equivalent to a row-major
  // (n_peaks, 2) float64 array. size() == 2 * n_peaks.
  std::vector<double> data;
  // Peaks of spectrum i are data rows [offsets[i], offsets[i + 1]).
  std::vector<int64_t> offsets{0};
  // Optional peak-level metadata table with one row per peak (n_peaks rows).
  std::shared_ptr<arrow::Table> metadata;

  int64_t n_peaks() const { return static_cast<int64_t>(data.size() / 2); }
  int64_t n_spectra() const { return static_cast<int64_t>(offsets.size()) - 1; }
};
}  // namespace detail

class Spectrum;

// Collection of spectra stored in one flattened peak array (CSR layout) plus
// a view ``index`` selecting/reordering spectra. Copies of a PeakSeries and
// the results of select()/slice()/reorder() are *views* that share storage;
// use copy() to materialize.
class PeakSeries {
 public:
  PeakSeries();
  PeakSeries(std::vector<double> data, std::vector<int64_t> offsets,
             std::shared_ptr<arrow::Table> metadata = nullptr,
             std::optional<std::vector<std::string>> metadata_columns = std::nullopt,
             std::optional<std::vector<int64_t>> index = std::nullopt,
             bool sort_by_mz = false);

  // Build a series from per-spectrum (m/z, intensity) lists.
  static PeakSeries from_spectra(
      const std::vector<std::vector<std::pair<double, double>>>& spectra);

  // --- size / layout of the current view ---------------------------------
  int64_t size() const { return static_cast<int64_t>(index_.size()); }
  int64_t n_peaks_total() const;
  int64_t n_peaks(int64_t position) const;
  std::vector<int64_t> lengths() const;
  std::vector<int64_t> offsets() const;
  std::vector<int64_t> peak_indices() const;
  const std::vector<int64_t>& index() const { return index_; }

  // --- peak values of the current view (materialized) --------------------
  std::vector<double> data() const;  // interleaved (m/z, intensity)
  void set_data(const std::vector<double>& data);
  std::vector<double> mz() const;
  std::vector<double> intensity() const;

  // --- peak metadata ------------------------------------------------------
  bool has_metadata() const { return storage_->metadata != nullptr; }
  // Metadata rows for the visible peaks, restricted to metadata_columns().
  // nullptr when the series has no peak metadata.
  std::shared_ptr<arrow::Table> metadata() const;
  const std::vector<std::string>& metadata_columns() const { return metadata_columns_; }
  void set_metadata_columns(std::vector<std::string> columns);
  // Add or update a metadata column for the visible peaks.
  void set_metadata_column(const std::string& name, const std::vector<Value>& values);
  void set_metadata_column(const std::string& name, const Value& value);

  // --- views --------------------------------------------------------------
  Spectrum operator[](int64_t position) const;
  Spectrum spectrum(int64_t position) const;
  PeakSeries select(const std::vector<int64_t>& positions) const;
  PeakSeries slice(int64_t start, int64_t stop) const;
  PeakSeries reorder(const std::vector<int64_t>& order) const;
  PeakSeries copy() const;
  PeakSeries& reset_view();
  PeakSeries full_view() const;

  // --- transforms ---------------------------------------------------------
  // Scale intensities so that each spectrum's maximum becomes ``scale``.
  PeakSeries& normalize(double scale = 1.0);
  PeakSeries normalized(double scale = 1.0) const;
  // Sort peaks within each spectrum (stable). The optional permutation
  // receives, for each output peak, its view-relative position before sorting.
  PeakSeries& sort_by_mz(bool ascending = true, std::vector<int64_t>* permutation = nullptr);
  PeakSeries sorted_by_mz(bool ascending = true) const;
  PeakSeries& sort_by_intensity(bool ascending = false,
                                std::vector<int64_t>* permutation = nullptr);
  PeakSeries sorted_by_intensity(bool ascending = false) const;

  // Replace the peaks of the visible spectra. ``offsets`` describes the
  // visible spectra (len == size() + 1); hidden spectra become empty.
  void replace_data(const std::vector<double>& data, const std::vector<int64_t>& offsets,
                    std::shared_ptr<arrow::Table> metadata = nullptr);

  // --- underlying storage (all spectra, ignoring the view) ---------------
  const std::vector<double>& data_ref() const { return storage_->data; }
  const std::vector<int64_t>& offsets_ref() const { return storage_->offsets; }
  std::shared_ptr<arrow::Table> metadata_ref() const { return storage_->metadata; }

 private:
  friend class Spectrum;
  friend class MSDataset;
  PeakSeries(std::shared_ptr<detail::PeakStorage> storage,
             std::vector<std::string> metadata_columns, std::vector<int64_t> index);
  PeakSeries& sort_by_column(int column, bool ascending, std::vector<int64_t>* permutation);
  int64_t source_index(int64_t position) const;

  std::shared_ptr<detail::PeakStorage> storage_;
  std::vector<std::string> metadata_columns_;
  std::vector<int64_t> index_;
};

// Lightweight view of one spectrum of a PeakSeries. It shares the series
// storage, so it stays valid after the PeakSeries object goes away.
class Spectrum {
 public:
  int64_t size() const { return end_ - begin_; }
  double mz(int64_t i) const;
  double intensity(int64_t i) const;
  Peak peak(int64_t i) const;
  Peak operator[](int64_t i) const { return peak(i); }
  std::vector<Peak> peaks() const;
  std::vector<double> mz_values() const;
  std::vector<double> intensity_values() const;
  // Peak metadata of this spectrum (nullptr when the series has none).
  std::shared_ptr<arrow::Table> metadata() const;
  const std::vector<std::string>& metadata_columns() const { return metadata_columns_; }
  // Index of the spectrum in the underlying storage.
  int64_t source_index() const { return source_index_; }

 private:
  friend class PeakSeries;
  Spectrum(std::shared_ptr<detail::PeakStorage> storage, std::vector<std::string> columns,
           int64_t source_index);

  std::shared_ptr<detail::PeakStorage> storage_;
  std::vector<std::string> metadata_columns_;
  int64_t source_index_;
  int64_t begin_;
  int64_t end_;
};

}  // namespace msentity
