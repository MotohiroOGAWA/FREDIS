#include "msentity/core/ms_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "internal/table.hpp"
#include "internal/text.hpp"
#include "msentity/io/msds.hpp"

namespace msentity {

namespace {

bool contains(const std::vector<std::string>& values, const std::string& value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

// Ordering used by sort_by: numbers (and bools) compare numerically,
// strings by code point; mixing the two is an error.
int compare_values(const Value& a, const Value& b) {
  auto numeric = [](const Value& v, double& out) {
    if (const auto* b = std::get_if<bool>(&v)) { out = *b ? 1.0 : 0.0; return true; }
    if (const auto* i = std::get_if<int64_t>(&v)) { out = static_cast<double>(*i); return true; }
    if (const auto* d = std::get_if<double>(&v)) { out = *d; return true; }
    return false;
  };
  double x = 0, y = 0;
  bool nx = numeric(a, x), ny = numeric(b, y);
  if (nx && ny) return x < y ? -1 : (x > y ? 1 : 0);
  const auto* sa = std::get_if<std::string>(&a);
  const auto* sb = std::get_if<std::string>(&b);
  if (sa && sb) return sa->compare(*sb) < 0 ? -1 : (*sa == *sb ? 0 : 1);
  throw std::invalid_argument("cannot sort a column mixing numbers and strings");
}

bool is_missing(const Value& value) {
  if (is_null(value)) return true;
  const auto* d = std::get_if<double>(&value);
  return d != nullptr && std::isnan(*d);
}

}  // namespace

// ---------------------------------------------------------------------------

Value SpectrumRecord::value(const std::string& column) const {
  return const_dataset_->value(position_, column);
}

bool SpectrumRecord::contains(const std::string& column) const {
  return const_dataset_->has_column(column);
}

void SpectrumRecord::set(const std::string& column, const Value& value) {
  if (dataset_ == nullptr) throw std::logic_error("record of a const dataset is read-only");
  dataset_->set_value(position_, column, value);
}

const std::vector<std::string>& SpectrumRecord::columns() const {
  return const_dataset_->columns();
}

int64_t SpectrumRecord::n_peaks() const { return const_dataset_->peaks().n_peaks(position_); }

Spectrum SpectrumRecord::peaks() const { return const_dataset_->peaks().spectrum(position_); }

// ---------------------------------------------------------------------------

MSDataset::MSDataset() : MSDataset(nullptr, PeakSeries()) {}

MSDataset::MSDataset(std::shared_ptr<arrow::Table> spectrum_metadata, PeakSeries peaks,
                     std::optional<std::vector<std::string>> columns, std::string description,
                     Attributes attributes, std::vector<std::string> tags)
    : metadata_(std::make_shared<detail::TableRef>()), peaks_(std::move(peaks)) {
  const int64_t n_spectra = static_cast<int64_t>(peaks_.offsets_ref().size()) - 1;
  metadata_->table = spectrum_metadata ? internal::normalize_table(spectrum_metadata)
                                       : internal::make_empty_table(n_spectra);
  if (metadata_->table->num_rows() != n_spectra) {
    throw std::invalid_argument(
        "The number of rows in spectrum_metadata must match the number of spectra in "
        "peak_series");
  }
  if (columns) {
    for (const auto& column : *columns) {
      if (internal::column_index(*metadata_->table, column) < 0) {
        throw std::invalid_argument("All columns must exist in spectrum_metadata. Missing: " +
                                    column);
      }
    }
    columns_ = std::move(*columns);
  } else {
    columns_ = internal::column_names(*metadata_->table);
  }
  description_ = std::move(description);
  set_attributes(std::move(attributes));
  set_tags(std::move(tags));
}

MSDataset::MSDataset(std::shared_ptr<detail::TableRef> metadata, PeakSeries peaks,
                     std::vector<std::string> columns, std::string description,
                     Attributes attributes, std::vector<std::string> tags)
    : metadata_(std::move(metadata)),
      peaks_(std::move(peaks)),
      columns_(std::move(columns)),
      description_(std::move(description)),
      attributes_(std::move(attributes)),
      tags_(std::move(tags)) {}

MSDataset MSDataset::with_peaks(PeakSeries peaks) const {
  return MSDataset(metadata_, std::move(peaks), columns_, description_, attributes_, tags_);
}

void MSDataset::set_columns(std::vector<std::string> columns) {
  for (const auto& column : columns) {
    if (internal::column_index(*metadata_->table, column) < 0) {
      throw std::invalid_argument("All columns must exist in spectrum_metadata. Missing: " +
                                  column);
    }
  }
  columns_ = std::move(columns);
}

bool MSDataset::has_column(const std::string& name) const { return contains(columns_, name); }

void MSDataset::set_attributes(Attributes attributes) {
  attributes_.clear();
  for (auto& [key, value] : attributes) set_attribute(key, value);
}

std::optional<std::string> MSDataset::attribute(const std::string& key) const {
  for (const auto& [k, v] : attributes_) {
    if (k == key) return v;
  }
  return std::nullopt;
}

void MSDataset::set_attribute(const std::string& key, const std::string& value) {
  if (key.empty()) throw std::invalid_argument("attribute key must not be empty");
  for (auto& [k, v] : attributes_) {
    if (k == key) {
      v = value;
      return;
    }
  }
  attributes_.emplace_back(key, value);
}

bool MSDataset::remove_attribute(const std::string& key) {
  auto it = std::find_if(attributes_.begin(), attributes_.end(),
                         [&](const auto& item) { return item.first == key; });
  if (it == attributes_.end()) return false;
  attributes_.erase(it);
  return true;
}

bool MSDataset::has_attribute(const std::string& key) const { return attribute(key).has_value(); }

void MSDataset::set_tags(std::vector<std::string> tags) {
  tags_.clear();
  for (auto& tag : tags) add_tag(tag);
}

bool MSDataset::add_tag(const std::string& tag) {
  if (contains(tags_, tag)) return false;
  tags_.push_back(tag);
  return true;
}

bool MSDataset::remove_tag(const std::string& tag) {
  auto it = std::find(tags_.begin(), tags_.end(), tag);
  if (it == tags_.end()) return false;
  tags_.erase(it);
  return true;
}

bool MSDataset::has_tag(const std::string& tag) const { return contains(tags_, tag); }

std::shared_ptr<arrow::Table> MSDataset::metadata() const {
  auto selected = internal::select_columns(*metadata_->table, columns_);
  return internal::take_rows(*selected, peaks_.index());
}

std::shared_ptr<arrow::ChunkedArray> MSDataset::column(const std::string& name) const {
  if (!has_column(name)) throw KeyError("column not available in current view: " + name);
  int c = internal::column_index(*metadata_->table, name);
  auto array = internal::take(*metadata_->table->column(c)->chunk(0), peaks_.index());
  return std::make_shared<arrow::ChunkedArray>(array);
}

std::vector<Value> MSDataset::column_values(const std::string& name) const {
  return internal::to_values(*column(name));
}

Value MSDataset::value(int64_t position, const std::string& column) const {
  if (!has_column(column)) throw KeyError("column not available in current view: " + column);
  int64_t row = peaks_.source_index(position);
  int c = internal::column_index(*metadata_->table, column);
  return internal::value_at(*metadata_->table->column(c)->chunk(0), row);
}

void MSDataset::set_column(const std::string& name, const std::vector<Value>& values) {
  if (static_cast<int64_t>(values.size()) != size()) {
    throw std::invalid_argument("value must have length " + std::to_string(size()) + ", got " +
                                std::to_string(values.size()));
  }
  auto& table = metadata_->table;
  std::vector<Value> column(static_cast<std::size_t>(table->num_rows()));
  int existing = internal::column_index(*table, name);
  if (existing >= 0) column = internal::to_values(*table->column(existing));
  const auto& index = peaks_.index();
  for (std::size_t k = 0; k < index.size(); ++k) {
    column[static_cast<std::size_t>(index[k])] = values[k];
  }
  table = internal::put_column(table, name, internal::values_to_array(column));
  if (!has_column(name)) columns_.push_back(name);
}

void MSDataset::set_column(const std::string& name, const Value& value) {
  set_column(name, std::vector<Value>(static_cast<std::size_t>(size()), value));
}

void MSDataset::set_value(int64_t position, const std::string& column, const Value& value) {
  int64_t row = peaks_.source_index(position);
  auto& table = metadata_->table;
  std::vector<Value> values(static_cast<std::size_t>(table->num_rows()));
  int existing = internal::column_index(*table, column);
  if (existing >= 0) values = internal::to_values(*table->column(existing));
  values[static_cast<std::size_t>(row)] = value;
  table = internal::put_column(table, column, internal::values_to_array(values));
  if (!has_column(column)) columns_.push_back(column);
}

SpectrumRecord MSDataset::operator[](int64_t position) { return record(position); }
SpectrumRecord MSDataset::operator[](int64_t position) const { return record(position); }

SpectrumRecord MSDataset::record(int64_t position) {
  peaks_.source_index(position);  // bounds check
  return SpectrumRecord(this, this, position);
}

SpectrumRecord MSDataset::record(int64_t position) const {
  peaks_.source_index(position);
  return SpectrumRecord(nullptr, this, position);
}

MSDataset MSDataset::select(const std::vector<int64_t>& positions) const {
  return with_peaks(peaks_.select(positions));
}

MSDataset MSDataset::slice(int64_t start, int64_t stop) const {
  return with_peaks(peaks_.slice(start, stop));
}

MSDataset MSDataset::filter(const std::vector<bool>& mask) const {
  if (static_cast<int64_t>(mask.size()) != size()) {
    throw std::invalid_argument("boolean index length " + std::to_string(mask.size()) +
                                " does not match dataset length " + std::to_string(size()));
  }
  std::vector<int64_t> positions;
  for (std::size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) positions.push_back(static_cast<int64_t>(i));
  }
  return select(positions);
}

MSDataset MSDataset::copy() const {
  return MSDataset(metadata(), peaks_.copy(), columns_, description_, attributes_, tags_);
}

MSDataset& MSDataset::reset_view(bool reset_columns) {
  if (reset_columns) columns_ = internal::column_names(*metadata_->table);
  peaks_.reset_view();
  return *this;
}

MSDataset MSDataset::full_view(bool reset_columns) const {
  MSDataset result = with_peaks(peaks_.full_view());
  if (reset_columns) result.columns_ = internal::column_names(*metadata_->table);
  return result;
}

MSDataset MSDataset::sort_by(const std::string& column, bool ascending) const {
  auto values = column_values(column);
  std::vector<int64_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    const Value& va = values[static_cast<std::size_t>(a)];
    const Value& vb = values[static_cast<std::size_t>(b)];
    bool ma = is_missing(va), mb = is_missing(vb);
    if (ma || mb) return !ma && mb;  // missing values last
    int cmp = compare_values(va, vb);
    return ascending ? cmp < 0 : cmp > 0;
  });
  return with_peaks(peaks_.reorder(order));
}

MSDataset MSDataset::concat(const std::vector<MSDataset>& datasets, std::string description,
                            Attributes attributes, std::vector<std::string> tags) {
  if (datasets.empty()) throw std::invalid_argument("datasets must not be empty");
  std::vector<MSDataset> copies;
  for (const auto& dataset : datasets) copies.push_back(dataset.copy());

  std::vector<std::shared_ptr<arrow::Table>> spectrum_tables;
  std::vector<std::string> columns;
  std::vector<double> data;
  std::vector<int64_t> offsets{0};
  bool any_peak_metadata = false;
  std::vector<std::string> peak_columns;
  for (const auto& dataset : copies) {
    spectrum_tables.push_back(dataset.metadata_->table);
    for (const auto& column : dataset.columns_) {
      if (!contains(columns, column)) columns.push_back(column);
    }
    const auto& peaks = dataset.peaks_;
    int64_t base = offsets.back();
    data.insert(data.end(), peaks.data_ref().begin(), peaks.data_ref().end());
    for (std::size_t i = 1; i < peaks.offsets_ref().size(); ++i) {
      offsets.push_back(base + peaks.offsets_ref()[i]);
    }
    if (peaks.has_metadata()) any_peak_metadata = true;
    for (const auto& column : peaks.metadata_columns()) {
      if (!contains(peak_columns, column)) peak_columns.push_back(column);
    }
  }

  std::shared_ptr<arrow::Table> peak_metadata;
  if (any_peak_metadata) {
    // Datasets without peak metadata contribute null rows.
    std::vector<std::shared_ptr<arrow::Table>> peak_tables;
    for (const auto& dataset : copies) {
      const auto& peaks = dataset.peaks_;
      int64_t n = static_cast<int64_t>(peaks.data_ref().size() / 2);
      peak_tables.push_back(peaks.has_metadata() ? peaks.metadata_ref()
                                                 : internal::make_empty_table(n));
    }
    peak_metadata = internal::concat_tables(peak_tables);
  }

  PeakSeries peaks(std::move(data), std::move(offsets), peak_metadata,
                   any_peak_metadata ? std::optional(peak_columns) : std::nullopt);
  return MSDataset(internal::concat_tables(spectrum_tables), std::move(peaks), columns,
                   std::move(description), std::move(attributes), std::move(tags));
}

void MSDataset::save(const std::string& path, bool save_view, bool append) const {
  io::MsdsWriteOptions options;
  options.save_view = save_view;
  options.append = append;
  io::write_msds(*this, path, options);
}

MSDataset MSDataset::load(const std::string& path, bool load_peak_metadata) {
  return io::read_msds(path, load_peak_metadata);
}

}  // namespace msentity
