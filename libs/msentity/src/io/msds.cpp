#include "msentity/io/msds.hpp"

#include <algorithm>
#include <stdexcept>

#include "internal/hdf5.hpp"
#include "internal/json.hpp"
#include "internal/parquet.hpp"
#include "internal/table.hpp"

namespace msentity::io {

namespace h5 = internal::h5;

namespace {

constexpr int64_t kArrowBytesLimit = 2'147'483'646;
constexpr int64_t kInitialRowsPerPart = 2'000'000;
const std::string kDatasetPrefix = "dataset_";

std::string part_name(const std::string& name, int64_t index) {
  char suffix[32];
  std::snprintf(suffix, sizeof(suffix), "__part_%03lld", static_cast<long long>(index));
  return name + suffix;
}

void save_bytes(hid_t group, const std::string& name, const std::vector<uint8_t>& blob) {
  if (h5::link_exists(group, name)) h5::delete_link(group, name);
  h5::Handle dataset = h5::write_bytes_dataset(group, name, blob.data(), blob.size(), true);
  h5::write_string_attribute(dataset.id(), "bytes_format", "uint8_1d");
  h5::write_int_attribute(dataset.id(), "nbytes", static_cast<int64_t>(blob.size()));
}

// Store a table as Parquet bytes, split into parts above max_part_bytes
// (MSDataset._save_parquet_h5).
void save_parquet(hid_t group, const std::string& name, const arrow::Table& table,
                  int64_t max_part_bytes) {
  if (h5::link_exists(group, name)) h5::delete_link(group, name);
  for (int64_t i = 0; h5::link_exists(group, part_name(name, i)); ++i) {
    h5::delete_link(group, part_name(name, i));
  }
  h5::delete_attribute(group, name + "__chunked");
  h5::delete_attribute(group, name + "__num_parts");

  auto size_of = [](const std::vector<uint8_t>& blob) {
    return std::max<int64_t>(static_cast<int64_t>(blob.size()),
                             internal::parquet_uncompressed_bytes(blob));
  };

  std::vector<uint8_t> blob = internal::table_to_parquet(table);
  if (size_of(blob) <= max_part_bytes) {
    save_bytes(group, name, blob);
    h5::write_bool_attribute(group, name + "__chunked", false);
    h5::write_int_attribute(group, name + "__num_parts", 1);
    return;
  }

  const int64_t n_rows = table.num_rows();
  int64_t rows_per_part = std::min(kInitialRowsPerPart, std::max<int64_t>(1, n_rows));
  std::vector<std::vector<uint8_t>> parts;
  int64_t start = 0;
  while (start < n_rows) {
    int64_t end = std::min(n_rows, start + rows_per_part);
    auto chunk_blob = internal::table_to_parquet(*internal::slice_rows(table, start, end - start));
    if (size_of(chunk_blob) > max_part_bytes) {
      if (rows_per_part == 1) {
        throw std::invalid_argument("even one-row parquet chunk exceeds max_part_bytes=" +
                                    std::to_string(max_part_bytes));
      }
      rows_per_part = std::max<int64_t>(1, rows_per_part / 2);
      continue;
    }
    parts.push_back(std::move(chunk_blob));
    start = end;
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    save_bytes(group, part_name(name, static_cast<int64_t>(i)), parts[i]);
  }
  h5::write_bool_attribute(group, name + "__chunked", true);
  h5::write_int_attribute(group, name + "__num_parts", static_cast<int64_t>(parts.size()));
}

bool has_parquet(hid_t group, const std::string& name) {
  return h5::link_exists(group, name) ||
         h5::read_bool_attribute(group, name + "__chunked").value_or(false);
}

std::shared_ptr<arrow::Table> load_parquet(hid_t group, const std::string& name) {
  if (h5::read_bool_attribute(group, name + "__chunked").value_or(false)) {
    int64_t num_parts = h5::read_int_attribute(group, name + "__num_parts").value_or(0);
    if (num_parts <= 0) throw FormatError("invalid num_parts for '" + name + "'");
    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (int64_t i = 0; i < num_parts; ++i) {
      auto blob = h5::read_bytes_dataset(group, part_name(name, i));
      tables.push_back(internal::parquet_to_table(blob.data(), blob.size()));
    }
    return internal::concat_tables(tables);
  }
  auto blob = h5::read_bytes_dataset(group, name);
  if (static_cast<int64_t>(blob.size()) > kArrowBytesLimit) {
    throw FormatError("'" + name +
                      "' is stored as a single parquet blob larger than the Arrow limit");
  }
  return internal::parquet_to_table(blob.data(), blob.size());
}

std::vector<std::string> dataset_group_names(hid_t file) {
  std::vector<std::pair<long long, std::string>> numbered;
  for (const auto& name : h5::member_names(file)) {
    if (name.rfind(kDatasetPrefix, 0) != 0) continue;
    std::string suffix = name.substr(kDatasetPrefix.size());
    if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos) {
      throw FormatError("invalid dataset group name: " + name);
    }
    numbered.emplace_back(std::stoll(suffix), name);
  }
  std::sort(numbered.begin(), numbered.end());
  std::vector<std::string> names;
  for (auto& [number, name] : numbered) names.push_back(std::move(name));
  return names;
}

void write_into(hid_t file, const MSDataset& source, const MsdsWriteOptions& options) {
  MSDataset dataset = options.save_view ? source.copy() : source;

  h5::Handle metadata_group = h5::link_exists(file, "metadata")
                                  ? h5::open_group(file, "metadata")
                                  : h5::create_group(file, "metadata");
  h5::write_string_attribute(metadata_group.id(), "description", dataset.description());
  h5::write_string_attribute(metadata_group.id(), "attributes_json",
                             internal::canonical_json(dataset.attributes()));
  h5::write_string_attribute(metadata_group.id(), "tags_json",
                             internal::canonical_json(dataset.tags()));

  auto existing = dataset_group_names(file);
  std::string group_name = kDatasetPrefix + "0";
  if (options.append && !existing.empty()) {
    long long next = std::stoll(existing.back().substr(kDatasetPrefix.size())) + 1;
    group_name = kDatasetPrefix + std::to_string(next);
  } else if (h5::link_exists(file, group_name)) {
    h5::delete_link(file, group_name);
  }
  h5::Handle group = h5::create_group(file, group_name);

  const PeakSeries& peaks = dataset.peaks();
  h5::Handle peaks_group = h5::create_group(group.id(), "peaks");
  const auto& data = peaks.data_ref();
  h5::write_double_dataset(peaks_group.id(), "data", data.data(),
                           {static_cast<hsize_t>(data.size() / 2), 2});
  h5::write_int64_dataset(peaks_group.id(), "offsets", peaks.offsets_ref());
  h5::write_int64_dataset(peaks_group.id(), "index", peaks.index());
  if (auto peak_metadata = peaks.metadata_ref()) {
    save_parquet(peaks_group.id(), "metadata_parquet", *peak_metadata, options.max_part_bytes);
    h5::write_string_dataset(peaks_group.id(), "metadata_columns", peaks.metadata_columns());
  }

  h5::write_string_attribute(group.id(), "columns_json",
                             internal::canonical_json(dataset.columns()));
  save_parquet(group.id(), "spectrum_metadata_parquet", *dataset.spectrum_metadata_ref(),
               options.max_part_bytes);
}

MSDataset read_from(hid_t file, bool load_peak_metadata) {
  std::string description;
  Attributes attributes;
  std::vector<std::string> tags;
  if (h5::link_exists(file, "metadata")) {
    h5::Handle metadata_group = h5::open_group(file, "metadata");
    description = h5::read_string_attribute(metadata_group.id(), "description").value_or("");
    attributes = internal::parse_json_attributes(
        h5::read_string_attribute(metadata_group.id(), "attributes_json").value_or("{}"));
    tags = internal::parse_json_string_list(
        h5::read_string_attribute(metadata_group.id(), "tags_json").value_or("[]"));
  }

  auto group_names = dataset_group_names(file);
  if (group_names.empty()) throw FormatError("no dataset groups found");

  std::vector<MSDataset> datasets;
  for (const auto& group_name : group_names) {
    h5::Handle group = h5::open_group(file, group_name);
    h5::Handle peaks_group = h5::open_group(group.id(), "peaks");

    auto shape = h5::dataset_shape(peaks_group.id(), "data");
    if (shape.size() != 2 || shape[1] != 2) {
      throw FormatError("peaks/data must have shape (n_peaks, 2) in " + group_name);
    }
    auto data = h5::read_double_dataset(peaks_group.id(), "data");
    auto offsets = h5::read_int64_dataset(peaks_group.id(), "offsets");
    auto index = h5::read_int64_dataset(peaks_group.id(), "index");

    // Parquet cannot store a row count without columns, so a zero-column
    // table has as many rows as the peak arrays imply (format spec §4.1).
    const int64_t n_peaks = static_cast<int64_t>(data.size() / 2);
    const int64_t n_spectra = static_cast<int64_t>(offsets.size()) - 1;

    std::shared_ptr<arrow::Table> peak_metadata;
    if (load_peak_metadata && has_parquet(peaks_group.id(), "metadata_parquet")) {
      peak_metadata = load_parquet(peaks_group.id(), "metadata_parquet");
      if (peak_metadata->num_columns() == 0) peak_metadata = internal::make_empty_table(n_peaks);
    }
    std::optional<std::vector<std::string>> metadata_columns;
    if (peak_metadata && h5::link_exists(peaks_group.id(), "metadata_columns")) {
      metadata_columns = h5::read_string_dataset(peaks_group.id(), "metadata_columns");
    }

    if (!has_parquet(group.id(), "spectrum_metadata_parquet")) {
      throw FormatError("missing 'spectrum_metadata_parquet' in group '" + group_name + "'");
    }
    std::optional<std::vector<std::string>> columns;
    if (auto columns_json = h5::read_string_attribute(group.id(), "columns_json")) {
      if (!columns_json->empty()) columns = internal::parse_json_string_list(*columns_json);
    }
    auto spectrum_metadata = load_parquet(group.id(), "spectrum_metadata_parquet");
    if (spectrum_metadata->num_columns() == 0) {
      spectrum_metadata = internal::make_empty_table(n_spectra);
    }

    PeakSeries peaks(std::move(data), std::move(offsets), peak_metadata, metadata_columns,
                     std::move(index));
    datasets.emplace_back(spectrum_metadata, std::move(peaks), columns, description,
                          attributes, tags);
  }
  if (datasets.size() == 1) return std::move(datasets.front());
  // The file-level description/attributes/tags apply to the joined dataset.
  return MSDataset::concat(datasets, description, attributes, tags);
}

}  // namespace

void write_msds(const MSDataset& dataset, const std::string& path,
                const MsdsWriteOptions& options) {
  h5::Handle file = h5::open_file(
      path, options.append ? h5::OpenMode::ReadWriteOrCreate : h5::OpenMode::Truncate);
  write_into(file.id(), dataset, options);
}

MSDataset read_msds(const std::string& path, bool load_peak_metadata) {
  h5::Handle file = h5::open_file(path, h5::OpenMode::ReadOnly);
  return read_from(file.id(), load_peak_metadata);
}

MSDatasetMeta read_msds_meta(const std::string& path) {
  h5::Handle file = h5::open_file(path, h5::OpenMode::ReadOnly);
  if (!h5::link_exists(file.id(), "metadata")) {
    throw KeyError("HDF5 file does not contain a '/metadata' group");
  }
  h5::Handle group = h5::open_group(file.id(), "metadata");
  MSDatasetMeta meta;
  meta.description = h5::read_string_attribute(group.id(), "description").value_or("");
  // Malformed JSON yields empty values instead of an error.
  try {
    meta.attributes = internal::parse_json_attributes(
        h5::read_string_attribute(group.id(), "attributes_json").value_or("{}"));
  } catch (const FormatError&) {
  }
  try {
    meta.tags = internal::parse_json_string_list(
        h5::read_string_attribute(group.id(), "tags_json").value_or("[]"));
  } catch (const FormatError&) {
  }
  return meta;
}

std::vector<uint8_t> msds_to_bytes(const MSDataset& dataset, const MsdsWriteOptions& options) {
  h5::Handle file = h5::create_memory_file();
  MsdsWriteOptions in_memory = options;
  in_memory.append = false;
  write_into(file.id(), dataset, in_memory);
  return h5::file_image(file);
}

MSDataset msds_from_bytes(const std::vector<uint8_t>& bytes, bool load_peak_metadata) {
  h5::Handle file = h5::open_image(bytes.data(), bytes.size());
  return read_from(file.id(), load_peak_metadata);
}

}  // namespace msentity::io
