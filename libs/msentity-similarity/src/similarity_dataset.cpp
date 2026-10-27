#include "msentity/similarity/similarity_dataset.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "internal/hdf5.hpp"
#include "internal/parquet.hpp"
#include "internal/table.hpp"
#include "internal/text.hpp"

namespace msentity::similarity {

namespace h5 = msentity::internal::h5;
namespace fs = std::filesystem;
using msentity::internal::check;
using msentity::internal::column_index;
using msentity::internal::unwrap;

namespace {

const char* kFormat = "msentity.similarity";
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::vector<int64_t> int_column(const arrow::Table& table, const std::string& name) {
  int index = column_index(table, name);
  if (index < 0) throw std::invalid_argument("missing column: " + name);
  std::vector<int64_t> values;
  for (const auto& value : internal::to_values(*table.column(index))) {
    if (const auto* i = std::get_if<int64_t>(&value)) {
      values.push_back(*i);
    } else if (const auto* d = std::get_if<double>(&value);
               d != nullptr && std::isfinite(*d) && *d == std::floor(*d)) {
      values.push_back(static_cast<int64_t>(*d));
    } else {
      throw std::invalid_argument("'" + name + "' must contain integers");
    }
  }
  return values;
}

std::vector<double> score_values(const arrow::Table& table) {
  std::vector<double> scores;
  for (const auto& value : internal::to_values(
           *table.column(column_index(table, "cosine_similarity")))) {
    double score = kNaN;
    if (const auto* d = std::get_if<double>(&value)) score = *d;
    if (const auto* i = std::get_if<int64_t>(&value)) score = static_cast<double>(*i);
    if (const auto* b = std::get_if<bool>(&value)) score = *b ? 1.0 : 0.0;
    scores.push_back(score);
  }
  return scores;
}

nlohmann::ordered_json parse_object(const std::string& text) {
  auto json = nlohmann::ordered_json::parse(text, nullptr, false);
  if (json.is_discarded() || !json.is_object()) {
    throw std::invalid_argument("metadata must be a JSON object");
  }
  return json;
}

std::shared_ptr<arrow::Array> int64_array(const std::vector<int64_t>& values) {
  arrow::Int64Builder builder;
  check(builder.AppendValues(values));
  return unwrap(builder.Finish());
}

// Linear interpolation between order statistics (pandas/NumPy default).
double quantile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return kNaN;
  double position = q * static_cast<double>(sorted.size() - 1);
  auto lower = static_cast<std::size_t>(std::floor(position));
  auto upper = static_cast<std::size_t>(std::ceil(position));
  double fraction = position - static_cast<double>(lower);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

// Ordering for sort_by: numbers by value, text by code point, missing last.
int compare(const Value& a, const Value& b) {
  auto numeric = [](const Value& v, double& out) {
    if (const auto* x = std::get_if<double>(&v)) { out = *x; return true; }
    if (const auto* x = std::get_if<int64_t>(&v)) { out = static_cast<double>(*x); return true; }
    if (const auto* x = std::get_if<bool>(&v)) { out = *x ? 1.0 : 0.0; return true; }
    return false;
  };
  double x = 0, y = 0;
  if (numeric(a, x) && numeric(b, y)) return x < y ? -1 : (x > y ? 1 : 0);
  const auto* sa = std::get_if<std::string>(&a);
  const auto* sb = std::get_if<std::string>(&b);
  if (sa && sb) return sa->compare(*sb) < 0 ? -1 : (*sa == *sb ? 0 : 1);
  throw std::invalid_argument("cannot sort a column mixing numbers and strings");
}

bool missing(const Value& value) {
  if (is_null(value)) return true;
  const auto* d = std::get_if<double>(&value);
  return d != nullptr && std::isnan(*d);
}

std::string csv_field(const std::string& value, char delimiter) {
  if (value.find_first_of(std::string{delimiter, '"', '\n', '\r'}) == std::string::npos) {
    return value;
  }
  return "\"" + internal::replace_all(value, "\"", "\"\"") + "\"";
}

}  // namespace

// ---------------------------------------------------------------------------

SimilarityDataset::SimilarityDataset(std::shared_ptr<arrow::Table> table, std::string metadata_json,
                                     std::optional<DatasetPair> matched_datasets,
                                     std::optional<IndexPair> matched_source_indices)
    : table_(table ? internal::normalize_table(table) : nullptr),
      metadata_json_(std::move(metadata_json)),
      matched_datasets_(std::move(matched_datasets)),
      source_indices_(std::move(matched_source_indices)) {
  validate();
}

void SimilarityDataset::validate() const {
  if (!table_) throw std::invalid_argument("table must not be null");
  parse_object(metadata_json_);
  std::vector<std::string> missing_columns;
  for (const char* column : {"cosine_similarity", "index1", "index2"}) {
    if (column_index(*table_, column) < 0) missing_columns.push_back(column);
  }
  if (!missing_columns.empty()) {
    std::string list;
    for (const auto& m : missing_columns) list += (list.empty() ? "'" : ", '") + m + "'";
    throw std::invalid_argument("Similarity table is missing required columns: [" + list + "]");
  }
  auto type = table_->column(column_index(*table_, "cosine_similarity"))->type()->id();
  if (!(arrow::is_integer(type) || arrow::is_floating(type) || type == arrow::Type::BOOL)) {
    throw std::invalid_argument("Similarity scores must be numeric");
  }
  for (double score : score_values(*table_)) {
    if (!std::isfinite(score) || score < 0 || score > 1) {
      throw std::invalid_argument("Similarity scores must be finite values between 0 and 1");
    }
  }
  if (matched_datasets_.has_value() != source_indices_.has_value()) {
    throw std::invalid_argument("matched datasets and source indices must be provided together");
  }
  if (!matched_datasets_) return;
  for (int side = 1; side <= 2; ++side) {
    const MSDataset& dataset = side == 1 ? matched_datasets_->first : matched_datasets_->second;
    const auto& sources = side == 1 ? source_indices_->first : source_indices_->second;
    std::string column = "data_index" + std::to_string(side);
    if (column_index(*table_, column) < 0) {
      throw std::invalid_argument("Embedded matches require the '" + column + "' column");
    }
    if (static_cast<int64_t>(sources.size()) != dataset.size()) {
      throw std::invalid_argument("Embedded source indices must match their datasets");
    }
    for (int64_t index : int_column(*table_, column)) {
      if (index < 0 || index >= static_cast<int64_t>(sources.size())) {
        throw std::invalid_argument("'" + column + "' contains an invalid embedded index");
      }
    }
  }
}

SimilarityDataset SimilarityDataset::with_matched_data(std::shared_ptr<arrow::Table> table,
                                                       std::string metadata_json,
                                                       const MSDataset& first,
                                                       const MSDataset& second) {
  if (!table) throw std::invalid_argument("table must not be null");
  std::vector<MSDataset> datasets;
  std::vector<std::vector<int64_t>> sources;
  for (int side = 1; side <= 2; ++side) {
    const MSDataset& dataset = side == 1 ? first : second;
    auto indices = int_column(*table, "index" + std::to_string(side));
    std::unordered_map<int64_t, int64_t> lookup;
    std::vector<int64_t> unique, local;
    for (int64_t index : indices) {
      if (index < 0 || index >= dataset.size()) {
        throw std::invalid_argument("'index" + std::to_string(side) +
                                    "' contains an invalid source index");
      }
      auto [it, inserted] = lookup.emplace(index, static_cast<int64_t>(unique.size()));
      if (inserted) unique.push_back(index);
      local.push_back(it->second);
    }
    table = internal::put_column(table, "data_index" + std::to_string(side), int64_array(local));
    datasets.push_back(dataset.select(unique).copy());
    sources.push_back(std::move(unique));
  }
  return SimilarityDataset(table, std::move(metadata_json),
                           DatasetPair{std::move(datasets[0]), std::move(datasets[1])},
                           IndexPair{std::move(sources[0]), std::move(sources[1])});
}

int64_t SimilarityDataset::size() const { return table_->num_rows(); }

std::vector<std::string> SimilarityDataset::columns() const {
  return internal::column_names(*table_);
}

std::vector<Value> SimilarityDataset::column_values(const std::string& name) const {
  int index = column_index(*table_, name);
  if (index < 0) throw KeyError("column not found: " + name);
  return internal::to_values(*table_->column(index));
}

void SimilarityDataset::set_metadata_json(std::string json) {
  parse_object(json);
  metadata_json_ = std::move(json);
}

SimilarityDataset::DatasetPair SimilarityDataset::match_records(int64_t row) const {
  if (!matched_datasets_) {
    throw std::invalid_argument("This similarity result does not contain matched data");
  }
  if (row < 0 || row >= size()) throw std::out_of_range("row out of range");
  int64_t first = int_column(*table_, "data_index1")[static_cast<std::size_t>(row)];
  int64_t second = int_column(*table_, "data_index2")[static_cast<std::size_t>(row)];
  return {matched_datasets_->first.select({first}), matched_datasets_->second.select({second})};
}

SimilarityDataset SimilarityDataset::filter(const std::vector<bool>& mask) const {
  if (static_cast<int64_t>(mask.size()) != size()) {
    throw std::invalid_argument("mask must have one entry per row");
  }
  std::vector<int64_t> rows;
  for (std::size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) rows.push_back(static_cast<int64_t>(i));
  }
  auto json = parse_object(metadata_json_);
  json["row_count"] = rows.size();
  json["source_row_count"] = size();
  return SimilarityDataset(internal::take_rows(*table_, rows), json.dump(), matched_datasets_,
                           source_indices_);
}

SimilarityDataset SimilarityDataset::sort_by(const std::vector<std::string>& columns,
                                             const std::vector<bool>& ascending) const {
  if (columns.empty()) throw std::invalid_argument("columns must not be empty");
  if (!ascending.empty() && ascending.size() != columns.size()) {
    throw std::invalid_argument("ascending must have one entry per column");
  }
  std::vector<std::vector<Value>> keys;
  for (const auto& column : columns) keys.push_back(column_values(column));
  std::vector<int64_t> order(static_cast<std::size_t>(size()));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    for (std::size_t c = 0; c < keys.size(); ++c) {
      const Value& va = keys[c][static_cast<std::size_t>(a)];
      const Value& vb = keys[c][static_cast<std::size_t>(b)];
      bool ma = missing(va), mb = missing(vb);
      if (ma || mb) {
        if (ma != mb) return mb;  // missing values last
        continue;
      }
      int cmp = compare(va, vb);
      if (cmp != 0) return (ascending.empty() || ascending[c]) ? cmp < 0 : cmp > 0;
    }
    return false;
  });
  return SimilarityDataset(internal::take_rows(*table_, order), metadata_json_, matched_datasets_,
                           source_indices_);
}

ScoreSummary SimilarityDataset::describe_scores() const {
  std::vector<double> scores = score_values(*table_);
  ScoreSummary summary;
  summary.count = static_cast<int64_t>(scores.size());
  if (scores.empty()) {
    summary.mean = summary.q1 = summary.median = summary.q3 = summary.min = summary.max = kNaN;
    return summary;
  }
  std::sort(scores.begin(), scores.end());
  summary.mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
  summary.q1 = quantile(scores, 0.25);
  summary.median = quantile(scores, 0.5);
  summary.q3 = quantile(scores, 0.75);
  summary.min = scores.front();
  summary.max = scores.back();
  return summary;
}

Histogram SimilarityDataset::histogram(int bins) const {
  if (bins < 1 || bins > 200) throw std::invalid_argument("bins must be an integer between 1 and 200");
  Histogram result;
  result.counts.assign(static_cast<std::size_t>(bins), 0);
  for (int i = 0; i <= bins; ++i) result.edges.push_back(static_cast<double>(i) / bins);
  for (double score : score_values(*table_)) {
    auto bin = static_cast<int>(std::floor(score * bins));
    bin = std::clamp(bin, 0, bins - 1);  // 1.0 belongs to the last bin
    ++result.counts[static_cast<std::size_t>(bin)];
  }
  return result;
}

void SimilarityDataset::export_table(const std::string& path) const {
  std::string ext = internal::to_lower_ascii(fs::path(path).extension().string());
  fs::path target(path);
  if (target.has_parent_path()) fs::create_directories(target.parent_path());
  if (ext == ".parquet") {
    auto bytes = internal::table_to_parquet(*table_);
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                                static_cast<std::streamsize>(bytes.size()));
    return;
  }
  if (ext != ".csv" && ext != ".tsv") {
    throw std::invalid_argument("Table export format must be parquet, csv, or tsv");
  }
  char delimiter = ext == ".csv" ? ',' : '\t';
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open file for writing: " + path);
  auto names = columns();
  for (std::size_t c = 0; c < names.size(); ++c) {
    out << (c ? std::string(1, delimiter) : "") << csv_field(names[c], delimiter);
  }
  out << "\n";
  std::vector<std::vector<Value>> values;
  for (const auto& name : names) values.push_back(column_values(name));
  for (int64_t row = 0; row < size(); ++row) {
    for (std::size_t c = 0; c < names.size(); ++c) {
      const Value& v = values[c][static_cast<std::size_t>(row)];
      out << (c ? std::string(1, delimiter) : "") << csv_field(to_display_string(v), delimiter);
    }
    out << "\n";
  }
}

// ---- files -----------------------------------------------------------------

SimilarityDataset SimilarityDataset::load(const std::string& path) {
  h5::Handle file = h5::open_file(path, h5::OpenMode::ReadOnly);
  if (h5::read_string_attribute(file.id(), "format").value_or("") != kFormat) {
    throw FormatError("Not an msentity similarity file: " + path);
  }
  int64_t version = h5::read_int_attribute(file.id(), "schema_version").value_or(0);
  if (version != 1 && version != 2) throw FormatError("Unsupported similarity schema version");

  auto payload = h5::read_bytes_dataset(file.id(), "table.parquet");
  auto table = internal::parquet_to_table(payload.data(), payload.size());
  auto metadata = h5::read_string_dataset(file.id(), "metadata.json");
  if (metadata.size() != 1) throw FormatError("metadata.json must be a scalar string");

  std::optional<DatasetPair> datasets;
  std::optional<IndexPair> sources;
  if (version == 2 && h5::link_exists(file.id(), "matched_data")) {
    h5::Handle embedded = h5::open_group(file.id(), "matched_data");
    std::vector<MSDataset> loaded;
    std::vector<std::vector<int64_t>> indices;
    for (const char* side : {"1", "2"}) {
      h5::Handle group = h5::open_group(embedded.id(), side);
      loaded.push_back(io::msds_from_bytes(h5::read_bytes_dataset(group.id(), "dataset.msds")));
      indices.push_back(h5::read_int64_dataset(group.id(), "source_indices"));
    }
    datasets = DatasetPair{std::move(loaded[0]), std::move(loaded[1])};
    sources = IndexPair{std::move(indices[0]), std::move(indices[1])};
  }
  return SimilarityDataset(table, metadata.front(), std::move(datasets), std::move(sources));
}

void SimilarityDataset::save(const std::string& path) const {
  fs::path target(path);
  if (internal::to_lower_ascii(target.extension().string()) != ".mssim") {
    throw std::invalid_argument("Similarity files must use the .mssim extension");
  }
  if (target.has_parent_path()) fs::create_directories(target.parent_path());

  // Keep only the embedded spectra the table refers to, renumbering
  // data_index1/2 in order of first appearance.
  std::shared_ptr<arrow::Table> table = table_;
  std::vector<MSDataset> datasets;
  std::vector<std::vector<int64_t>> sources;
  if (matched_datasets_) {
    for (int side = 1; side <= 2; ++side) {
      std::string column = "data_index" + std::to_string(side);
      std::unordered_map<int64_t, int64_t> lookup;
      std::vector<int64_t> referenced, renumbered;
      for (int64_t old_index : int_column(*table, column)) {
        auto [it, inserted] = lookup.emplace(old_index, static_cast<int64_t>(referenced.size()));
        if (inserted) referenced.push_back(old_index);
        renumbered.push_back(it->second);
      }
      table = internal::put_column(table, column, int64_array(renumbered));
      const MSDataset& dataset = side == 1 ? matched_datasets_->first : matched_datasets_->second;
      const auto& source = side == 1 ? source_indices_->first : source_indices_->second;
      datasets.push_back(dataset.select(referenced).copy());
      std::vector<int64_t> kept;
      for (int64_t index : referenced) kept.push_back(source[static_cast<std::size_t>(index)]);
      sources.push_back(std::move(kept));
    }
  }

  auto payload = internal::table_to_parquet(*table);
  std::random_device random;
  fs::path temporary = target;
  temporary += ".tmp" + std::to_string(random());
  try {
    {
      h5::Handle file = h5::open_file(temporary.string(), h5::OpenMode::Truncate);
      h5::write_string_attribute(file.id(), "format", kFormat);
      h5::write_int_attribute(file.id(), "schema_version", 2);
      h5::write_string_attribute(file.id(), "data_mode",
                                 matched_datasets_ ? "embedded" : "lightweight");
      h5::write_bytes_dataset(file.id(), "table.parquet", payload.data(), payload.size(), false);
      h5::write_scalar_string_dataset(file.id(), "metadata.json", metadata_json_);
      if (matched_datasets_) {
        h5::Handle embedded = h5::create_group(file.id(), "matched_data");
        for (int side = 0; side < 2; ++side) {
          h5::Handle group = h5::create_group(embedded.id(), std::to_string(side + 1));
          h5::write_int64_dataset(group.id(), "source_indices", sources[static_cast<std::size_t>(side)]);
          auto bytes = io::msds_to_bytes(datasets[static_cast<std::size_t>(side)]);
          h5::write_bytes_dataset(group.id(), "dataset.msds", bytes.data(), bytes.size(), true);
        }
      }
    }
    fs::rename(temporary, target);
  } catch (...) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
}

}  // namespace msentity::similarity
