#include "msentity/core/peak_series.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "internal/table.hpp"

namespace msentity {

using internal::check;
using internal::unwrap;

namespace {

std::vector<int64_t> iota_index(int64_t n) {
  std::vector<int64_t> index(static_cast<std::size_t>(n));
  std::iota(index.begin(), index.end(), 0);
  return index;
}

std::vector<std::string> without_peak_columns(const std::vector<std::string>& columns) {
  std::vector<std::string> out;
  for (const auto& column : columns) {
    if (column != "mz" && column != "intensity") out.push_back(column);
  }
  return out;
}

void validate_storage(const detail::PeakStorage& storage) {
  if (storage.data.size() % 2 != 0) {
    throw std::invalid_argument("data must have shape (n_peaks, 2)");
  }
  if (storage.offsets.empty()) throw std::invalid_argument("offsets must not be empty");
  if (storage.offsets.front() < 0) throw std::invalid_argument("offsets must be non-negative");
  for (std::size_t i = 1; i < storage.offsets.size(); ++i) {
    if (storage.offsets[i] < storage.offsets[i - 1]) {
      throw std::invalid_argument("offsets must be non-decreasing");
    }
  }
  if (storage.offsets.back() > storage.n_peaks()) {
    throw std::invalid_argument("offsets[-1] must not exceed the number of peaks");
  }
  if (storage.metadata && storage.metadata->num_rows() != storage.n_peaks()) {
    throw std::invalid_argument("metadata must have the same number of rows as data");
  }
}

// NaN sorts after every number.
bool less_nan_last(double a, double b) { return !std::isnan(a) && (std::isnan(b) || a < b); }

}  // namespace

PeakSeries::PeakSeries() : storage_(std::make_shared<detail::PeakStorage>()) {}

PeakSeries::PeakSeries(std::vector<double> data, std::vector<int64_t> offsets,
                       std::shared_ptr<arrow::Table> metadata,
                       std::optional<std::vector<std::string>> metadata_columns,
                       std::optional<std::vector<int64_t>> index, bool sort_by_mz)
    : storage_(std::make_shared<detail::PeakStorage>()) {
  storage_->data = std::move(data);
  storage_->offsets = std::move(offsets);
  storage_->metadata = metadata ? internal::normalize_table(metadata) : nullptr;
  validate_storage(*storage_);

  if (!metadata_columns && storage_->metadata) {
    metadata_columns = internal::column_names(*storage_->metadata);
  }
  metadata_columns_ = metadata_columns ? without_peak_columns(*metadata_columns)
                                       : std::vector<std::string>{};
  for (const auto& column : metadata_columns_) {
    if (!storage_->metadata || internal::column_index(*storage_->metadata, column) < 0) {
      throw std::invalid_argument("all metadata columns must exist in metadata: " + column);
    }
  }

  if (index) {
    for (int64_t i : *index) {
      if (i < 0 || i >= storage_->n_spectra()) {
        throw std::out_of_range("spectrum index out of range: " + std::to_string(i));
      }
    }
    index_ = std::move(*index);
  } else {
    index_ = iota_index(storage_->n_spectra());
  }
  if (sort_by_mz) this->sort_by_mz(true);
}

PeakSeries::PeakSeries(std::shared_ptr<detail::PeakStorage> storage,
                       std::vector<std::string> metadata_columns, std::vector<int64_t> index)
    : storage_(std::move(storage)),
      metadata_columns_(std::move(metadata_columns)),
      index_(std::move(index)) {}

PeakSeries PeakSeries::from_spectra(
    const std::vector<std::vector<std::pair<double, double>>>& spectra) {
  std::vector<double> data;
  std::vector<int64_t> offsets{0};
  for (const auto& spectrum : spectra) {
    for (const auto& [mz, intensity] : spectrum) {
      data.push_back(mz);
      data.push_back(intensity);
    }
    offsets.push_back(static_cast<int64_t>(data.size() / 2));
  }
  return PeakSeries(std::move(data), std::move(offsets));
}

int64_t PeakSeries::source_index(int64_t position) const {
  if (position < 0 || position >= size()) {
    throw std::out_of_range("spectrum index out of range: " + std::to_string(position));
  }
  return index_[static_cast<std::size_t>(position)];
}

int64_t PeakSeries::n_peaks_total() const {
  int64_t total = 0;
  const auto& offsets = storage_->offsets;
  for (int64_t i : index_) total += offsets[i + 1] - offsets[i];
  return total;
}

int64_t PeakSeries::n_peaks(int64_t position) const {
  int64_t i = source_index(position);
  return storage_->offsets[i + 1] - storage_->offsets[i];
}

std::vector<int64_t> PeakSeries::lengths() const {
  std::vector<int64_t> lengths;
  lengths.reserve(index_.size());
  const auto& offsets = storage_->offsets;
  for (int64_t i : index_) lengths.push_back(offsets[i + 1] - offsets[i]);
  return lengths;
}

std::vector<int64_t> PeakSeries::offsets() const {
  std::vector<int64_t> offsets{0};
  offsets.reserve(index_.size() + 1);
  for (int64_t length : lengths()) offsets.push_back(offsets.back() + length);
  return offsets;
}

std::vector<int64_t> PeakSeries::peak_indices() const {
  std::vector<int64_t> indices;
  indices.reserve(static_cast<std::size_t>(n_peaks_total()));
  const auto& offsets = storage_->offsets;
  for (int64_t i : index_) {
    for (int64_t p = offsets[i]; p < offsets[i + 1]; ++p) indices.push_back(p);
  }
  return indices;
}

std::vector<double> PeakSeries::data() const {
  std::vector<double> data;
  data.reserve(static_cast<std::size_t>(n_peaks_total()) * 2);
  const auto& offsets = storage_->offsets;
  for (int64_t i : index_) {
    data.insert(data.end(), storage_->data.begin() + offsets[i] * 2,
                storage_->data.begin() + offsets[i + 1] * 2);
  }
  return data;
}

void PeakSeries::set_data(const std::vector<double>& data) {
  if (static_cast<int64_t>(data.size()) != n_peaks_total() * 2) {
    throw std::invalid_argument("data must have the same shape as the current view");
  }
  std::size_t cursor = 0;
  const auto& offsets = storage_->offsets;
  for (int64_t i : index_) {
    std::size_t length = static_cast<std::size_t>((offsets[i + 1] - offsets[i]) * 2);
    std::copy(data.begin() + static_cast<std::ptrdiff_t>(cursor),
              data.begin() + static_cast<std::ptrdiff_t>(cursor + length),
              storage_->data.begin() + offsets[i] * 2);
    cursor += length;
  }
}

std::vector<double> PeakSeries::mz() const {
  auto data = this->data();
  std::vector<double> mz(data.size() / 2);
  for (std::size_t i = 0; i < mz.size(); ++i) mz[i] = data[2 * i];
  return mz;
}

std::vector<double> PeakSeries::intensity() const {
  auto data = this->data();
  std::vector<double> intensity(data.size() / 2);
  for (std::size_t i = 0; i < intensity.size(); ++i) intensity[i] = data[2 * i + 1];
  return intensity;
}

std::shared_ptr<arrow::Table> PeakSeries::metadata() const {
  if (!storage_->metadata) return nullptr;
  auto rows = internal::take_rows(*storage_->metadata, peak_indices());
  return internal::select_columns(*rows, metadata_columns_);
}

void PeakSeries::set_metadata_columns(std::vector<std::string> columns) {
  if (!storage_->metadata) throw std::logic_error("metadata is not available for this PeakSeries");
  columns = without_peak_columns(columns);
  for (const auto& column : columns) {
    if (internal::column_index(*storage_->metadata, column) < 0) {
      throw std::invalid_argument("all metadata columns must exist in metadata");
    }
  }
  metadata_columns_ = std::move(columns);
}

void PeakSeries::set_metadata_column(const std::string& name, const std::vector<Value>& values) {
  auto indices = peak_indices();
  if (values.size() != indices.size()) {
    throw std::invalid_argument("value must have length " + std::to_string(indices.size()) +
                                ", got " + std::to_string(values.size()));
  }
  if (!storage_->metadata) storage_->metadata = internal::make_empty_table(storage_->n_peaks());
  std::vector<Value> column(static_cast<std::size_t>(storage_->n_peaks()));
  int existing = internal::column_index(*storage_->metadata, name);
  if (existing >= 0) column = internal::to_values(*storage_->metadata->column(existing));
  for (std::size_t k = 0; k < indices.size(); ++k) {
    column[static_cast<std::size_t>(indices[k])] = values[k];
  }
  storage_->metadata =
      internal::put_column(storage_->metadata, name, internal::values_to_array(column));
  if (std::find(metadata_columns_.begin(), metadata_columns_.end(), name) ==
      metadata_columns_.end()) {
    metadata_columns_.push_back(name);
  }
}

void PeakSeries::set_metadata_column(const std::string& name, const Value& value) {
  set_metadata_column(name, std::vector<Value>(static_cast<std::size_t>(n_peaks_total()), value));
}

Spectrum PeakSeries::operator[](int64_t position) const { return spectrum(position); }

Spectrum PeakSeries::spectrum(int64_t position) const {
  return Spectrum(storage_, metadata_columns_, source_index(position));
}

PeakSeries PeakSeries::select(const std::vector<int64_t>& positions) const {
  std::vector<int64_t> index;
  index.reserve(positions.size());
  for (int64_t position : positions) {
    // Negative positions count from the end (-1 = last).
    index.push_back(source_index(position < 0 ? position + size() : position));
  }
  return PeakSeries(storage_, metadata_columns_, std::move(index));
}

PeakSeries PeakSeries::slice(int64_t start, int64_t stop) const {
  auto clamp = [this](int64_t v) {
    if (v < 0) v += size();
    return std::clamp<int64_t>(v, 0, size());
  };
  start = clamp(start);
  stop = clamp(stop);
  std::vector<int64_t> index;
  for (int64_t i = start; i < stop; ++i) index.push_back(index_[static_cast<std::size_t>(i)]);
  return PeakSeries(storage_, metadata_columns_, std::move(index));
}

PeakSeries PeakSeries::reorder(const std::vector<int64_t>& order) const {
  std::vector<int64_t> sorted = order;
  std::sort(sorted.begin(), sorted.end());
  if (static_cast<int64_t>(order.size()) != size() || sorted != iota_index(size())) {
    throw std::invalid_argument("order must be a permutation of range(len(self))");
  }
  return select(order);
}

PeakSeries PeakSeries::copy() const {
  return PeakSeries(data(), offsets(), metadata(), metadata_columns_);
}

PeakSeries& PeakSeries::reset_view() {
  index_ = iota_index(storage_->n_spectra());
  return *this;
}

PeakSeries PeakSeries::full_view() const {
  return PeakSeries(storage_, metadata_columns_, iota_index(storage_->n_spectra()));
}

PeakSeries& PeakSeries::normalize(double scale) {
  auto& data = storage_->data;
  const auto& offsets = storage_->offsets;
  std::unordered_set<int64_t> done;
  for (int64_t i : index_) {
    if (!done.insert(i).second) continue;
    int64_t start = offsets[i];
    int64_t end = offsets[i + 1];
    if (end <= start) continue;
    double max_value = -HUGE_VAL;
    bool has_nan = false;
    for (int64_t p = start; p < end; ++p) {
      double v = data[static_cast<std::size_t>(p * 2 + 1)];
      if (std::isnan(v)) has_nan = true;
      max_value = std::max(max_value, v);
    }
    // A spectrum containing NaN, or without a positive maximum, is left unchanged.
    if (has_nan || !(max_value > 0)) continue;
    for (int64_t p = start; p < end; ++p) {
      double& v = data[static_cast<std::size_t>(p * 2 + 1)];
      v = v / max_value * scale;
    }
  }
  return *this;
}

PeakSeries PeakSeries::normalized(double scale) const {
  PeakSeries result = copy();
  result.normalize(scale);
  return result;
}

PeakSeries& PeakSeries::sort_by_column(int column, bool ascending,
                                       std::vector<int64_t>* permutation) {
  auto& data = storage_->data;
  const auto& offsets = storage_->offsets;
  // Global row permutation applied to the peak metadata table.
  std::vector<int64_t> metadata_order;
  if (storage_->metadata) metadata_order = iota_index(storage_->n_peaks());
  if (permutation) permutation->clear();

  std::unordered_set<int64_t> done;
  int64_t view_cursor = 0;
  for (int64_t i : index_) {
    int64_t start = offsets[i];
    int64_t length = offsets[i + 1] - start;
    std::vector<int64_t> order = iota_index(length);
    if (done.count(i) == 0) {
      std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return less_nan_last(data[static_cast<std::size_t>((start + a) * 2 + column)],
                             data[static_cast<std::size_t>((start + b) * 2 + column)]);
      });
      // Descending order is the stable ascending order reversed.
      if (!ascending) std::reverse(order.begin(), order.end());
      std::vector<double> segment(data.begin() + start * 2, data.begin() + (start + length) * 2);
      for (int64_t k = 0; k < length; ++k) {
        data[static_cast<std::size_t>((start + k) * 2)] =
            segment[static_cast<std::size_t>(order[k] * 2)];
        data[static_cast<std::size_t>((start + k) * 2 + 1)] =
            segment[static_cast<std::size_t>(order[k] * 2 + 1)];
        if (storage_->metadata) metadata_order[start + k] = start + order[k];
      }
      done.insert(i);
    } else {
      // Duplicate view entry: already sorted, so the permutation is identity.
    }
    if (permutation) {
      for (int64_t k = 0; k < length; ++k) permutation->push_back(view_cursor + order[k]);
    }
    view_cursor += length;
  }
  if (storage_->metadata) {
    storage_->metadata = internal::take_rows(*storage_->metadata, metadata_order);
  }
  return *this;
}

PeakSeries& PeakSeries::sort_by_mz(bool ascending, std::vector<int64_t>* permutation) {
  return sort_by_column(0, ascending, permutation);
}

PeakSeries PeakSeries::sorted_by_mz(bool ascending) const {
  PeakSeries result = copy();
  result.sort_by_mz(ascending);
  return result;
}

PeakSeries& PeakSeries::sort_by_intensity(bool ascending, std::vector<int64_t>* permutation) {
  return sort_by_column(1, ascending, permutation);
}

PeakSeries PeakSeries::sorted_by_intensity(bool ascending) const {
  PeakSeries result = copy();
  result.sort_by_intensity(ascending);
  return result;
}

void PeakSeries::replace_data(const std::vector<double>& data,
                              const std::vector<int64_t>& offsets,
                              std::shared_ptr<arrow::Table> metadata) {
  if (data.size() % 2 != 0) throw std::invalid_argument("data must have shape (n_peaks, 2)");
  if (static_cast<int64_t>(offsets.size()) != size() + 1) {
    throw std::invalid_argument("len(offsets) - 1 must match the number of visible spectra");
  }
  if (offsets.front() != 0) throw std::invalid_argument("offsets must start with 0");
  for (std::size_t i = 1; i < offsets.size(); ++i) {
    if (offsets[i] < offsets[i - 1]) throw std::invalid_argument("offsets must be non-decreasing");
  }
  int64_t n_new = static_cast<int64_t>(data.size() / 2);
  if (offsets.back() != n_new) throw std::invalid_argument("offsets[-1] must equal len(data)");
  if (metadata && metadata->num_rows() != n_new) {
    throw std::invalid_argument("metadata must have the same number of rows as data");
  }

  const int64_t n_all = storage_->n_spectra();
  std::vector<int64_t> view_position(static_cast<std::size_t>(n_all), -1);
  for (std::size_t p = 0; p < index_.size(); ++p) view_position[index_[p]] = static_cast<int64_t>(p);

  auto storage = std::make_shared<detail::PeakStorage>();
  storage->offsets.assign(1, 0);
  std::vector<int64_t> metadata_rows;
  for (int64_t s = 0; s < n_all; ++s) {
    int64_t p = view_position[s];
    if (p >= 0) {
      int64_t start = offsets[p];
      int64_t end = offsets[p + 1];
      storage->data.insert(storage->data.end(), data.begin() + start * 2, data.begin() + end * 2);
      for (int64_t r = start; r < end; ++r) metadata_rows.push_back(r);
    }
    storage->offsets.push_back(static_cast<int64_t>(storage->data.size() / 2));
  }
  if (metadata) {
    storage->metadata = internal::take_rows(*internal::normalize_table(metadata), metadata_rows);
    metadata_columns_ = without_peak_columns(internal::column_names(*metadata));
  } else {
    metadata_columns_.clear();
  }
  storage_ = std::move(storage);
}

// ---------------------------------------------------------------------------

Spectrum::Spectrum(std::shared_ptr<detail::PeakStorage> storage,
                   std::vector<std::string> columns, int64_t source_index)
    : storage_(std::move(storage)),
      metadata_columns_(std::move(columns)),
      source_index_(source_index),
      begin_(storage_->offsets[source_index]),
      end_(storage_->offsets[source_index + 1]) {}

double Spectrum::mz(int64_t i) const {
  if (i < 0 || i >= size()) throw std::out_of_range("peak index out of range");
  return storage_->data[static_cast<std::size_t>((begin_ + i) * 2)];
}

double Spectrum::intensity(int64_t i) const {
  if (i < 0 || i >= size()) throw std::out_of_range("peak index out of range");
  return storage_->data[static_cast<std::size_t>((begin_ + i) * 2 + 1)];
}

Peak Spectrum::peak(int64_t i) const {
  Peak peak(mz(i), intensity(i));
  if (storage_->metadata) {
    for (const auto& column : metadata_columns_) {
      int c = internal::column_index(*storage_->metadata, column);
      peak.metadata[column] =
          internal::value_at(*storage_->metadata->column(c)->chunk(0), begin_ + i);
    }
  }
  return peak;
}

std::vector<Peak> Spectrum::peaks() const {
  std::vector<Peak> peaks;
  peaks.reserve(static_cast<std::size_t>(size()));
  for (int64_t i = 0; i < size(); ++i) peaks.push_back(peak(i));
  return peaks;
}

std::vector<double> Spectrum::mz_values() const {
  std::vector<double> values;
  for (int64_t i = 0; i < size(); ++i) values.push_back(mz(i));
  return values;
}

std::vector<double> Spectrum::intensity_values() const {
  std::vector<double> values;
  for (int64_t i = 0; i < size(); ++i) values.push_back(intensity(i));
  return values;
}

std::shared_ptr<arrow::Table> Spectrum::metadata() const {
  if (!storage_->metadata) return nullptr;
  auto rows = internal::slice_rows(*storage_->metadata, begin_, end_ - begin_);
  return internal::select_columns(*rows, metadata_columns_);
}

}  // namespace msentity
