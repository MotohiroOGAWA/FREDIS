// Implementation of the C API (include/msentity/c/msentity_c.h).

#include "msentity/c/msentity_c.h"

#include <arrow/c/bridge.h>
#include <arrow/table.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal/table.hpp"
#include "internal/text.hpp"
#include "msentity/msentity.hpp"

struct msentity_dataset {
  msentity::MSDataset dataset;
};

namespace {

using msentity::MSDataset;
using msentity::Value;

thread_local std::string last_error;

int32_t fail(int32_t status, const char* message) {
  last_error = message;
  return status;
}

// Run `body`, translating exceptions into status codes.
template <typename Body>
int32_t guarded(Body&& body) {
  try {
    body();
    last_error.clear();
    return MSENTITY_OK;
  } catch (const msentity::KeyError& error) {
    return fail(MSENTITY_ERR_KEY, error.what());
  } catch (const std::out_of_range& error) {
    return fail(MSENTITY_ERR_OUT_OF_RANGE, error.what());
  } catch (const std::invalid_argument& error) {
    return fail(MSENTITY_ERR_INVALID_ARGUMENT, error.what());
  } catch (const msentity::FormatError& error) {
    // A file that does not exist or cannot be created is an I/O error.
    std::string message = error.what();
    bool io = message.find("cannot open") != std::string::npos ||
              message.find("cannot create") != std::string::npos;
    return fail(io ? MSENTITY_ERR_IO : MSENTITY_ERR_FORMAT, error.what());
  } catch (const std::filesystem::filesystem_error& error) {
    return fail(MSENTITY_ERR_IO, error.what());
  } catch (const std::runtime_error& error) {
    std::string message = error.what();
    bool io = message.rfind("cannot open", 0) == 0 || message.rfind("failed to write", 0) == 0;
    return fail(io ? MSENTITY_ERR_IO : MSENTITY_ERR_UNKNOWN, error.what());
  } catch (const std::exception& error) {
    return fail(MSENTITY_ERR_UNKNOWN, error.what());
  } catch (...) {
    return fail(MSENTITY_ERR_UNKNOWN, "unknown error");
  }
}

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

const MSDataset& get(const msentity_dataset* handle) {
  require(handle != nullptr, "dataset handle is NULL");
  return handle->dataset;
}

MSDataset& get(msentity_dataset* handle) {
  require(handle != nullptr, "dataset handle is NULL");
  return handle->dataset;
}

std::string text(const char* value, const char* what) {
  if (value == nullptr) throw std::invalid_argument(std::string(what) + " is NULL");
  return value;
}

char* duplicate(const std::string& value) {
  char* out = static_cast<char*>(std::malloc(value.size() + 1));
  if (out == nullptr) throw std::bad_alloc();
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

void put_string(char** out, const std::string& value) {
  require(out != nullptr, "output pointer is NULL");
  *out = duplicate(value);
}

void put_handle(msentity_dataset** out, MSDataset dataset) {
  require(out != nullptr, "output pointer is NULL");
  *out = new msentity_dataset{std::move(dataset)};
}

// msentity::Value <-> msentity_value -------------------------------------------
void to_c(const Value& value, msentity_value* out) {
  std::memset(out, 0, sizeof(*out));
  if (std::holds_alternative<std::monostate>(value)) {
    out->type = MSENTITY_TYPE_NULL;
  } else if (const auto* b = std::get_if<bool>(&value)) {
    out->type = MSENTITY_TYPE_BOOL;
    out->bool_value = *b ? 1 : 0;
  } else if (const auto* i = std::get_if<int64_t>(&value)) {
    out->type = MSENTITY_TYPE_INT;
    out->int_value = *i;
  } else if (const auto* d = std::get_if<double>(&value)) {
    out->type = MSENTITY_TYPE_FLOAT;
    out->float_value = *d;
  } else {
    out->type = MSENTITY_TYPE_STRING;
    out->string_value = duplicate(std::get<std::string>(value));
  }
}

Value from_c(const msentity_value& value) {
  switch (value.type) {
    case MSENTITY_TYPE_NULL:
      return std::monostate{};
    case MSENTITY_TYPE_BOOL:
      return value.bool_value != 0;
    case MSENTITY_TYPE_INT:
      return value.int_value;
    case MSENTITY_TYPE_FLOAT:
      return value.float_value;
    case MSENTITY_TYPE_STRING:
      return std::string(value.string_value != nullptr ? value.string_value : "");
    default:
      throw std::invalid_argument("unknown msentity_value type " + std::to_string(value.type));
  }
}

std::vector<Value> from_c(const msentity_value* values, int64_t n) {
  require(n == 0 || values != nullptr, "values is NULL");
  std::vector<Value> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) out.push_back(from_c(values[i]));
  return out;
}

void check_length(int64_t given, int64_t expected, const char* what) {
  if (given != expected) {
    throw std::invalid_argument(std::string(what) + " must have " + std::to_string(expected) +
                                " elements, got " + std::to_string(given));
  }
}

void export_values(const std::vector<Value>& values, msentity_value* out, int64_t n) {
  check_length(n, static_cast<int64_t>(values.size()), "output array");
  require(n == 0 || out != nullptr, "output array is NULL");
  int64_t done = 0;
  try {
    for (; done < n; ++done) to_c(values[static_cast<std::size_t>(done)], &out[done]);
  } catch (...) {
    msentity_values_clear(out, done);
    throw;
  }
}

std::string format_from(const char* format, const std::string& path) {
  if (format != nullptr) return msentity::internal::to_lower_ascii(format);
  std::string ext =
      msentity::internal::to_lower_ascii(std::filesystem::path(path).extension().string());
  if (ext == ".h5" || ext == ".hdf5") return "msds";
  if (ext.size() > 1) return ext.substr(1);
  throw std::invalid_argument("cannot determine the format of '" + path + "'");
}

void export_table(const std::shared_ptr<arrow::Table>& table, void* out_stream) {
  require(out_stream != nullptr, "out_stream is NULL");
  auto reader = std::make_shared<arrow::TableBatchReader>(table);
  msentity::internal::check(arrow::ExportRecordBatchReader(
      reader, static_cast<struct ArrowArrayStream*>(out_stream)));
}

}  // namespace

extern "C" {

// ---- library ------------------------------------------------------------------
const char* msentity_version(void) { return msentity::kVersion; }

const char* msentity_last_error(void) { return last_error.c_str(); }

void msentity_string_free(char* value) { std::free(value); }

void msentity_values_clear(msentity_value* values, int64_t n) {
  if (values == nullptr) return;
  for (int64_t i = 0; i < n; ++i) {
    if (values[i].type == MSENTITY_TYPE_STRING) std::free(values[i].string_value);
    values[i].string_value = nullptr;
    values[i].type = MSENTITY_TYPE_NULL;
  }
}

// ---- creating and releasing ----------------------------------------------------
int32_t msentity_load(const char* path, const char* file_type, const char* spec_id_prefix,
                      int32_t infer_types, msentity_dataset** out) {
  return guarded([&] {
    std::optional<std::string> type, prefix;
    if (file_type != nullptr) type = file_type;
    if (spec_id_prefix != nullptr) prefix = spec_id_prefix;
    put_handle(out, msentity::io::load_ms_dataset(text(path, "path"), type, prefix, {},
                                                  infer_types != 0));
  });
}

int32_t msentity_read_text(const char* content, const char* format, int32_t infer_types,
                           msentity_dataset** out) {
  return guarded([&] {
    std::string kind = msentity::internal::to_lower_ascii(text(format, "format"));
    std::string body = text(content, "text");
    msentity::io::DelimitedReadOptions delimited;
    delimited.infer_types = infer_types != 0;
    if (kind == "msp") {
      put_handle(out, msentity::io::read_msp_text(body));
    } else if (kind == "mgf") {
      put_handle(out, msentity::io::read_mgf_text(body));
    } else if (kind == "tsv") {
      put_handle(out, msentity::io::read_tsv_text(body, delimited));
    } else if (kind == "csv") {
      put_handle(out, msentity::io::read_csv_text(body, delimited));
    } else {
      throw std::invalid_argument("unsupported text format '" + kind +
                                  "' (use msp, mgf, tsv, or csv)");
    }
  });
}

int32_t msentity_from_peaks(const double* data, int64_t n_peaks, const int64_t* offsets,
                            int64_t n_offsets, msentity_dataset** out) {
  return guarded([&] {
    require(n_peaks >= 0 && (n_peaks == 0 || data != nullptr), "data is NULL");
    require(n_offsets >= 1 && offsets != nullptr, "offsets must have at least one element");
    std::vector<double> peaks(data, data + n_peaks * 2);
    std::vector<int64_t> bounds(offsets, offsets + n_offsets);
    put_handle(out, MSDataset(nullptr, msentity::PeakSeries(std::move(peaks), std::move(bounds))));
  });
}

int32_t msentity_concat(const msentity_dataset* const* datasets, int64_t n,
                        msentity_dataset** out) {
  return guarded([&] {
    require(n > 0 && datasets != nullptr, "datasets must not be empty");
    std::vector<MSDataset> items;
    for (int64_t i = 0; i < n; ++i) items.push_back(get(datasets[i]));
    put_handle(out, MSDataset::concat(items));
  });
}

void msentity_free(msentity_dataset* dataset) { delete dataset; }

// ---- saving ---------------------------------------------------------------------------
int32_t msentity_save(const msentity_dataset* handle, const char* path, const char* format,
                      int32_t save_view, int32_t append) {
  return guarded([&] {
    const MSDataset& dataset = get(handle);
    std::string target = text(path, "path");
    std::string kind = format_from(format, target);
    if (kind == "msds") {
      msentity::io::MsdsWriteOptions options;
      options.save_view = save_view != 0;
      options.append = append != 0;
      msentity::io::write_msds(dataset, target, options);
    } else if (kind == "msp") {
      msentity::io::write_msp(dataset, target);
    } else if (kind == "mgf") {
      msentity::io::write_mgf(dataset, target);
    } else if (kind == "tsv") {
      msentity::io::write_tsv(dataset, target);
    } else if (kind == "csv") {
      msentity::io::write_csv(dataset, target);
    } else {
      throw std::invalid_argument("unsupported format '" + kind +
                                  "' (use msds, msp, mgf, tsv, or csv)");
    }
  });
}

// ---- shape and columns ------------------------------------------------------------------
int64_t msentity_size(const msentity_dataset* handle) {
  return handle != nullptr ? handle->dataset.size() : 0;
}

int64_t msentity_n_peaks_total(const msentity_dataset* handle) {
  return handle != nullptr ? handle->dataset.n_peaks_total() : 0;
}

int64_t msentity_n_columns(const msentity_dataset* handle) {
  return handle != nullptr ? handle->dataset.n_columns() : 0;
}

int32_t msentity_column_name(const msentity_dataset* handle, int64_t index, char** out) {
  return guarded([&] {
    const auto& columns = get(handle).columns();
    if (index < 0 || index >= static_cast<int64_t>(columns.size())) {
      throw std::out_of_range("column index out of range");
    }
    put_string(out, columns[static_cast<std::size_t>(index)]);
  });
}

int32_t msentity_has_column(const msentity_dataset* handle, const char* name) {
  return handle != nullptr && name != nullptr && handle->dataset.has_column(name) ? 1 : 0;
}

int32_t msentity_set_columns(msentity_dataset* handle, const char* const* names, int64_t n) {
  return guarded([&] {
    require(n == 0 || names != nullptr, "names is NULL");
    std::vector<std::string> columns;
    for (int64_t i = 0; i < n; ++i) columns.push_back(text(names[i], "column name"));
    get(handle).set_columns(std::move(columns));
  });
}

// ---- metadata values -----------------------------------------------------------------------
int32_t msentity_get_value(const msentity_dataset* handle, int64_t position, const char* column,
                           msentity_value* out) {
  return guarded([&] {
    require(out != nullptr, "output pointer is NULL");
    to_c(get(handle).value(position, text(column, "column")), out);
  });
}

int32_t msentity_get_column(const msentity_dataset* handle, const char* column,
                            msentity_value* out, int64_t n) {
  return guarded([&] { export_values(get(handle).column_values(text(column, "column")), out, n); });
}

int32_t msentity_set_column(msentity_dataset* handle, const char* column,
                            const msentity_value* values, int64_t n) {
  return guarded([&] { get(handle).set_column(text(column, "column"), from_c(values, n)); });
}

int32_t msentity_set_column_scalar(msentity_dataset* handle, const char* column,
                                   const msentity_value* value) {
  return guarded([&] {
    require(value != nullptr, "value is NULL");
    get(handle).set_column(text(column, "column"), from_c(*value));
  });
}

int32_t msentity_set_value(msentity_dataset* handle, int64_t position, const char* column,
                           const msentity_value* value) {
  return guarded([&] {
    require(value != nullptr, "value is NULL");
    get(handle).set_value(position, text(column, "column"), from_c(*value));
  });
}

// ---- views -------------------------------------------------------------------------------------
int32_t msentity_select(const msentity_dataset* handle, const int64_t* positions, int64_t n,
                        msentity_dataset** out) {
  return guarded([&] {
    require(n == 0 || positions != nullptr, "positions is NULL");
    put_handle(out, get(handle).select(std::vector<int64_t>(positions, positions + n)));
  });
}

int32_t msentity_slice(const msentity_dataset* handle, int64_t start, int64_t stop,
                       msentity_dataset** out) {
  return guarded([&] { put_handle(out, get(handle).slice(start, stop)); });
}

int32_t msentity_filter(const msentity_dataset* handle, const uint8_t* mask, int64_t n,
                        msentity_dataset** out) {
  return guarded([&] {
    require(n == 0 || mask != nullptr, "mask is NULL");
    std::vector<bool> keep(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) keep[static_cast<std::size_t>(i)] = mask[i] != 0;
    put_handle(out, get(handle).filter(keep));
  });
}

int32_t msentity_sort_by(const msentity_dataset* handle, const char* column, int32_t ascending,
                         msentity_dataset** out) {
  return guarded(
      [&] { put_handle(out, get(handle).sort_by(text(column, "column"), ascending != 0)); });
}

int32_t msentity_copy(const msentity_dataset* handle, msentity_dataset** out) {
  return guarded([&] { put_handle(out, get(handle).copy()); });
}

int32_t msentity_full_view(const msentity_dataset* handle, int32_t reset_columns,
                           msentity_dataset** out) {
  return guarded([&] { put_handle(out, get(handle).full_view(reset_columns != 0)); });
}

// ---- peaks ----------------------------------------------------------------------------------------
int32_t msentity_offsets(const msentity_dataset* handle, int64_t* out, int64_t n) {
  return guarded([&] {
    auto offsets = get(handle).peaks().offsets();
    check_length(n, static_cast<int64_t>(offsets.size()), "offsets");
    require(out != nullptr, "output array is NULL");
    std::copy(offsets.begin(), offsets.end(), out);
  });
}

int32_t msentity_peaks(const msentity_dataset* handle, double* mz, double* intensity,
                       int64_t n) {
  return guarded([&] {
    const auto& peaks = get(handle).peaks();
    check_length(n, peaks.n_peaks_total(), "peak arrays");
    auto data = peaks.data();
    for (int64_t i = 0; i < n; ++i) {
      if (mz) mz[i] = data[static_cast<std::size_t>(2 * i)];
      if (intensity) intensity[i] = data[static_cast<std::size_t>(2 * i + 1)];
    }
  });
}

int32_t msentity_spectrum_size(const msentity_dataset* handle, int64_t position, int64_t* out) {
  return guarded([&] {
    require(out != nullptr, "output pointer is NULL");
    *out = get(handle).peaks().n_peaks(position);
  });
}

int32_t msentity_spectrum_peaks(const msentity_dataset* handle, int64_t position, double* mz,
                                double* intensity, int64_t n) {
  return guarded([&] {
    msentity::Spectrum spectrum = get(handle).peaks().spectrum(position);
    check_length(n, spectrum.size(), "peak arrays");
    for (int64_t i = 0; i < n; ++i) {
      if (mz) mz[i] = spectrum.mz(i);
      if (intensity) intensity[i] = spectrum.intensity(i);
    }
  });
}

int32_t msentity_normalize(msentity_dataset* handle, double scale) {
  return guarded([&] { get(handle).peaks().normalize(scale); });
}

int32_t msentity_sort_peaks(msentity_dataset* handle, int32_t by, int32_t ascending) {
  return guarded([&] {
    require(by == 0 || by == 1, "by must be 0 (m/z) or 1 (intensity)");
    if (by == 0) {
      get(handle).peaks().sort_by_mz(ascending != 0);
    } else {
      get(handle).peaks().sort_by_intensity(ascending != 0);
    }
  });
}

// ---- peak metadata ----------------------------------------------------------------------------------
int64_t msentity_n_peak_columns(const msentity_dataset* handle) {
  return handle != nullptr ? static_cast<int64_t>(handle->dataset.peaks().metadata_columns().size())
                           : 0;
}

int32_t msentity_peak_column_name(const msentity_dataset* handle, int64_t index, char** out) {
  return guarded([&] {
    const auto& columns = get(handle).peaks().metadata_columns();
    if (index < 0 || index >= static_cast<int64_t>(columns.size())) {
      throw std::out_of_range("peak column index out of range");
    }
    put_string(out, columns[static_cast<std::size_t>(index)]);
  });
}

int32_t msentity_get_peak_column(const msentity_dataset* handle, const char* column,
                                 msentity_value* out, int64_t n) {
  return guarded([&] {
    std::string name = text(column, "column");
    auto table = get(handle).peaks().metadata();
    int index = table ? msentity::internal::column_index(*table, name) : -1;
    if (index < 0) throw msentity::KeyError("peak column not available: " + name);
    export_values(msentity::internal::to_values(*table->column(index)), out, n);
  });
}

int32_t msentity_set_peak_column(msentity_dataset* handle, const char* column,
                                 const msentity_value* values, int64_t n) {
  return guarded(
      [&] { get(handle).peaks().set_metadata_column(text(column, "column"), from_c(values, n)); });
}

// ---- dataset-level metadata ----------------------------------------------------------------------------
int32_t msentity_get_description(const msentity_dataset* handle, char** out) {
  return guarded([&] { put_string(out, get(handle).description()); });
}

int32_t msentity_set_description(msentity_dataset* handle, const char* value) {
  return guarded([&] { get(handle).set_description(text(value, "description")); });
}

int64_t msentity_n_attributes(const msentity_dataset* handle) {
  return handle != nullptr ? static_cast<int64_t>(handle->dataset.attributes().size()) : 0;
}

int32_t msentity_attribute(const msentity_dataset* handle, int64_t index, char** key,
                           char** value) {
  return guarded([&] {
    const auto& attributes = get(handle).attributes();
    if (index < 0 || index >= static_cast<int64_t>(attributes.size())) {
      throw std::out_of_range("attribute index out of range");
    }
    require(key != nullptr && value != nullptr, "output pointer is NULL");
    const auto& [k, v] = attributes[static_cast<std::size_t>(index)];
    *key = duplicate(k);
    *value = duplicate(v);
  });
}

int32_t msentity_set_attribute(msentity_dataset* handle, const char* key, const char* value) {
  return guarded([&] { get(handle).set_attribute(text(key, "key"), text(value, "value")); });
}

int32_t msentity_remove_attribute(msentity_dataset* handle, const char* key, int32_t* removed) {
  return guarded([&] {
    bool result = get(handle).remove_attribute(text(key, "key"));
    if (removed) *removed = result ? 1 : 0;
  });
}

int64_t msentity_n_tags(const msentity_dataset* handle) {
  return handle != nullptr ? static_cast<int64_t>(handle->dataset.tags().size()) : 0;
}

int32_t msentity_tag(const msentity_dataset* handle, int64_t index, char** out) {
  return guarded([&] {
    const auto& tags = get(handle).tags();
    if (index < 0 || index >= static_cast<int64_t>(tags.size())) {
      throw std::out_of_range("tag index out of range");
    }
    put_string(out, tags[static_cast<std::size_t>(index)]);
  });
}

int32_t msentity_add_tag(msentity_dataset* handle, const char* tag, int32_t* added) {
  return guarded([&] {
    bool result = get(handle).add_tag(text(tag, "tag"));
    if (added) *added = result ? 1 : 0;
  });
}

int32_t msentity_remove_tag(msentity_dataset* handle, const char* tag, int32_t* removed) {
  return guarded([&] {
    bool result = get(handle).remove_tag(text(tag, "tag"));
    if (removed) *removed = result ? 1 : 0;
  });
}

int32_t msentity_set_spec_id(msentity_dataset* handle, const char* column, const char* prefix,
                             int32_t overwrite, int64_t start, int32_t* assigned) {
  return guarded([&] {
    bool result = msentity::processing::set_spec_id(
        get(handle), column != nullptr ? column : "SpecID", prefix != nullptr ? prefix : "",
        overwrite != 0, start);
    if (assigned) *assigned = result ? 1 : 0;
  });
}

// ---- MSDS bytes ----------------------------------------------------------------------------------------------
int32_t msentity_to_msds_bytes(const msentity_dataset* handle, uint8_t** out, int64_t* size) {
  return guarded([&] {
    require(out != nullptr && size != nullptr, "output pointer is NULL");
    std::vector<uint8_t> bytes = msentity::io::msds_to_bytes(get(handle));
    auto* buffer = static_cast<uint8_t*>(std::malloc(bytes.empty() ? 1 : bytes.size()));
    if (buffer == nullptr) throw std::bad_alloc();
    std::memcpy(buffer, bytes.data(), bytes.size());
    *out = buffer;
    *size = static_cast<int64_t>(bytes.size());
  });
}

int32_t msentity_from_msds_bytes(const uint8_t* data, int64_t size, msentity_dataset** out) {
  return guarded([&] {
    require(data != nullptr && size > 0, "data is empty");
    std::vector<uint8_t> bytes(data, data + size);
    put_handle(out, msentity::io::msds_from_bytes(bytes));
  });
}

void msentity_bytes_free(uint8_t* data) { std::free(data); }

// ---- Arrow export ------------------------------------------------------------------------------------------
int32_t msentity_metadata_arrow(const msentity_dataset* handle, void* out_stream) {
  return guarded([&] { export_table(get(handle).metadata(), out_stream); });
}

int32_t msentity_peak_metadata_arrow(const msentity_dataset* handle, void* out_stream) {
  return guarded([&] {
    auto table = get(handle).peaks().metadata();
    if (!table) table = msentity::internal::make_empty_table(get(handle).n_peaks_total());
    export_table(table, out_stream);
  });
}

}  // extern "C"
