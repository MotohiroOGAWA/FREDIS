// Implementation of the C API (include/msentity/similarity/c/msentity_similarity_c.h).

#include "msentity/similarity/c/msentity_similarity_c.h"

#include <arrow/c/bridge.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal/table.hpp"
#include "msentity/similarity/similarity.hpp"

struct mssim_result {
  msentity::similarity::SimilarityDataset result;
};

namespace {

using msentity::MSDataset;
using msentity::Value;
using msentity::similarity::SimilarityDataset;

thread_local std::string last_error;

int32_t fail(int32_t status, const char* message) {
  last_error = message;
  return status;
}

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

const SimilarityDataset& get(const mssim_result* handle) {
  require(handle != nullptr, "result handle is NULL");
  return handle->result;
}

SimilarityDataset& get(mssim_result* handle) {
  require(handle != nullptr, "result handle is NULL");
  return handle->result;
}

std::string text(const char* value, const char* fallback = "") {
  return value != nullptr ? value : fallback;
}

char* duplicate(const std::string& value) {
  char* out = static_cast<char*>(std::malloc(value.size() + 1));
  if (out == nullptr) throw std::bad_alloc();
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

MSDataset dataset_from(const uint8_t* data, int64_t size, const char* what) {
  if (data == nullptr || size <= 0) {
    throw std::invalid_argument(std::string(what) + " is empty");
  }
  return msentity::io::msds_from_bytes(std::vector<uint8_t>(data, data + size));
}

msentity::similarity::Options options_from(const mssim_options* options) {
  msentity::similarity::Options result;
  if (options == nullptr) return result;
  require(options->method == MSSIM_METHOD_COSINE || options->method == MSSIM_METHOD_REVERSE_COSINE,
          "method must be MSSIM_METHOD_COSINE or MSSIM_METHOD_REVERSE_COSINE");
  result.method = options->method == MSSIM_METHOD_COSINE
                      ? msentity::similarity::Method::Cosine
                      : msentity::similarity::Method::ReverseCosine;
  result.threads = options->threads;
  result.bin_width = options->bin_width;
  result.intensity_exponent = options->intensity_exponent;
  return result;
}

void put_result(mssim_result** out, SimilarityDataset result) {
  require(out != nullptr, "output pointer is NULL");
  *out = new mssim_result{std::move(result)};
}

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

void check_side(int32_t side) { require(side == 1 || side == 2, "side must be 1 or 2"); }

}  // namespace

extern "C" {

const char* mssim_version(void) { return msentity::similarity::kVersion; }
const char* mssim_last_error(void) { return last_error.c_str(); }
void mssim_string_free(char* value) { std::free(value); }
void mssim_bytes_free(uint8_t* data) { std::free(data); }

void mssim_values_clear(msentity_value* values, int64_t n) {
  if (values == nullptr) return;
  for (int64_t i = 0; i < n; ++i) {
    if (values[i].type == MSENTITY_TYPE_STRING) std::free(values[i].string_value);
    values[i].string_value = nullptr;
    values[i].type = MSENTITY_TYPE_NULL;
  }
}

void mssim_default_options(mssim_options* options) {
  if (options == nullptr) return;
  msentity::similarity::Options defaults;
  options->method = MSSIM_METHOD_COSINE;
  options->threads = defaults.threads;
  options->bin_width = defaults.bin_width;
  options->intensity_exponent = defaults.intensity_exponent;
}

// ---- calculations -----------------------------------------------------------------
int32_t mssim_pair_scores(const uint8_t* ds1, int64_t ds1_size, const int64_t* index1,
                          const uint8_t* ds2, int64_t ds2_size, const int64_t* index2, int64_t n,
                          const mssim_options* options, float* out) {
  return guarded([&] {
    require(n >= 0 && (n == 0 || (index1 && index2 && out)), "index arrays or output are NULL");
    auto scores = msentity::similarity::pair_scores(
        dataset_from(ds1, ds1_size, "ds1"), std::vector<int64_t>(index1, index1 + n),
        dataset_from(ds2, ds2_size, "ds2"), std::vector<int64_t>(index2, index2 + n),
        options_from(options));
    std::copy(scores.begin(), scores.end(), out);
  });
}

int32_t mssim_all_pairs(const uint8_t* ds1, int64_t ds1_size, const uint8_t* ds2,
                        int64_t ds2_size, const mssim_options* options, float* out, int64_t n) {
  return guarded([&] {
    auto matrix = msentity::similarity::all_pairs_matrix(dataset_from(ds1, ds1_size, "ds1"),
                                                         dataset_from(ds2, ds2_size, "ds2"),
                                                         options_from(options));
    if (n != static_cast<int64_t>(matrix.size())) {
      throw std::invalid_argument("output must have " + std::to_string(matrix.size()) +
                                  " elements, got " + std::to_string(n));
    }
    require(n == 0 || out != nullptr, "output is NULL");
    std::copy(matrix.begin(), matrix.end(), out);
  });
}

int32_t mssim_library_search(const uint8_t* query, int64_t query_size, const uint8_t* reference,
                             int64_t reference_size, double threshold,
                             const mssim_options* options, int32_t include_matched_data,
                             const char* query_source, const char* reference_source,
                             mssim_progress_fn progress, void* user_data, mssim_result** out) {
  return guarded([&] {
    msentity::similarity::ProgressCallback callback;
    if (progress != nullptr) {
      callback = [progress, user_data](int64_t done, int64_t total) {
        progress(done, total, user_data);
      };
    }
    put_result(out, msentity::similarity::calculate_library_search(
                        dataset_from(query, query_size, "query"),
                        dataset_from(reference, reference_size, "reference"), threshold,
                        options_from(options), include_matched_data != 0, text(query_source),
                        text(reference_source), callback));
  });
}

int32_t mssim_similarity_by_key(const uint8_t* ds1, int64_t ds1_size, const uint8_t* ds2,
                                int64_t ds2_size, const char* key1, const char* key2,
                                const mssim_options* options, int32_t include_matched_data,
                                const char* source1, const char* source2, mssim_result** out) {
  return guarded([&] {
    put_result(out, msentity::similarity::calculate_similarity(
                        dataset_from(ds1, ds1_size, "ds1"), dataset_from(ds2, ds2_size, "ds2"),
                        text(key1, "SpecID"), text(key2, "SpecID"), options_from(options),
                        include_matched_data != 0, text(source1), text(source2)));
  });
}

// ---- results ------------------------------------------------------------------------
int32_t mssim_load(const char* path, mssim_result** out) {
  return guarded([&] {
    require(path != nullptr, "path is NULL");
    put_result(out, SimilarityDataset::load(path));
  });
}

int32_t mssim_save(const mssim_result* result, const char* path) {
  return guarded([&] {
    require(path != nullptr, "path is NULL");
    get(result).save(path);
  });
}

void mssim_free(mssim_result* result) { delete result; }

int64_t mssim_size(const mssim_result* result) {
  return result != nullptr ? result->result.size() : 0;
}

int64_t mssim_n_columns(const mssim_result* result) {
  return result != nullptr ? static_cast<int64_t>(result->result.columns().size()) : 0;
}

int32_t mssim_column_name(const mssim_result* result, int64_t index, char** out) {
  return guarded([&] {
    auto columns = get(result).columns();
    if (index < 0 || index >= static_cast<int64_t>(columns.size())) {
      throw std::out_of_range("column index out of range");
    }
    require(out != nullptr, "output pointer is NULL");
    *out = duplicate(columns[static_cast<std::size_t>(index)]);
  });
}

int32_t mssim_get_column(const mssim_result* result, const char* column, msentity_value* out,
                         int64_t n) {
  return guarded([&] {
    require(column != nullptr, "column is NULL");
    auto values = get(result).column_values(column);
    if (n != static_cast<int64_t>(values.size())) {
      throw std::invalid_argument("output array must have " + std::to_string(values.size()) +
                                  " elements, got " + std::to_string(n));
    }
    require(n == 0 || out != nullptr, "output array is NULL");
    int64_t done = 0;
    try {
      for (; done < n; ++done) to_c(values[static_cast<std::size_t>(done)], &out[done]);
    } catch (...) {
      mssim_values_clear(out, done);
      throw;
    }
  });
}

int32_t mssim_get_metadata_json(const mssim_result* result, char** out) {
  return guarded([&] {
    require(out != nullptr, "output pointer is NULL");
    *out = duplicate(get(result).metadata_json());
  });
}

int32_t mssim_set_metadata_json(mssim_result* result, const char* json) {
  return guarded([&] {
    require(json != nullptr, "json is NULL");
    get(result).set_metadata_json(json);
  });
}

int32_t mssim_has_matched_data(const mssim_result* result) {
  return result != nullptr && result->result.has_matched_data() ? 1 : 0;
}

int32_t mssim_matched_dataset(const mssim_result* result, int32_t side, uint8_t** out,
                              int64_t* size) {
  return guarded([&] {
    check_side(side);
    require(out != nullptr && size != nullptr, "output pointer is NULL");
    const auto& matched = get(result).matched_datasets();
    require(matched.has_value(), "This similarity result does not contain matched data");
    auto bytes = msentity::io::msds_to_bytes(side == 1 ? matched->first : matched->second);
    auto* buffer = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (buffer == nullptr) throw std::bad_alloc();
    std::memcpy(buffer, bytes.data(), bytes.size());
    *out = buffer;
    *size = static_cast<int64_t>(bytes.size());
  });
}

int64_t mssim_matched_size(const mssim_result* result, int32_t side) {
  if (result == nullptr || !result->result.has_matched_data() || (side != 1 && side != 2)) return 0;
  const auto& matched = *result->result.matched_datasets();
  return side == 1 ? matched.first.size() : matched.second.size();
}

int32_t mssim_source_indices(const mssim_result* result, int32_t side, int64_t* out, int64_t n) {
  return guarded([&] {
    check_side(side);
    const auto& sources = get(result).matched_source_indices();
    require(sources.has_value(), "This similarity result does not contain matched data");
    const auto& values = side == 1 ? sources->first : sources->second;
    if (n != static_cast<int64_t>(values.size())) {
      throw std::invalid_argument("output must have " + std::to_string(values.size()) +
                                  " elements, got " + std::to_string(n));
    }
    require(n == 0 || out != nullptr, "output is NULL");
    std::copy(values.begin(), values.end(), out);
  });
}

int32_t mssim_match_positions(const mssim_result* result, int64_t row, int64_t* data_index1,
                              int64_t* data_index2) {
  return guarded([&] {
    require(data_index1 != nullptr && data_index2 != nullptr, "output pointer is NULL");
    const auto& r = get(result);
    if (!r.has_matched_data()) {
      throw std::invalid_argument("This similarity result does not contain matched data");
    }
    if (row < 0 || row >= r.size()) throw std::out_of_range("row out of range");
    *data_index1 = std::get<int64_t>(r.column_values("data_index1")[static_cast<std::size_t>(row)]);
    *data_index2 = std::get<int64_t>(r.column_values("data_index2")[static_cast<std::size_t>(row)]);
  });
}

int32_t mssim_filter(const mssim_result* result, const uint8_t* mask, int64_t n,
                     mssim_result** out) {
  return guarded([&] {
    require(n == 0 || mask != nullptr, "mask is NULL");
    std::vector<bool> keep(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) keep[static_cast<std::size_t>(i)] = mask[i] != 0;
    put_result(out, get(result).filter(keep));
  });
}

int32_t mssim_sort_by(const mssim_result* result, const char* const* columns,
                      const int32_t* ascending, int64_t n, mssim_result** out) {
  return guarded([&] {
    require(n > 0 && columns != nullptr, "columns must not be empty");
    std::vector<std::string> names;
    std::vector<bool> order;
    for (int64_t i = 0; i < n; ++i) {
      require(columns[i] != nullptr, "column name is NULL");
      names.emplace_back(columns[i]);
      order.push_back(ascending == nullptr || ascending[i] != 0);
    }
    put_result(out, get(result).sort_by(names, order));
  });
}

int32_t mssim_describe_scores(const mssim_result* result, double* out) {
  return guarded([&] {
    require(out != nullptr, "output is NULL");
    auto s = get(result).describe_scores();
    double values[7] = {static_cast<double>(s.count), s.mean, s.q1, s.median, s.q3, s.min, s.max};
    std::copy(values, values + 7, out);
  });
}

int32_t mssim_histogram(const mssim_result* result, int32_t bins, int64_t* counts, double* edges) {
  return guarded([&] {
    require(counts != nullptr && edges != nullptr, "output is NULL");
    auto h = get(result).histogram(bins);
    std::copy(h.counts.begin(), h.counts.end(), counts);
    std::copy(h.edges.begin(), h.edges.end(), edges);
  });
}

int32_t mssim_export_table(const mssim_result* result, const char* path) {
  return guarded([&] {
    require(path != nullptr, "path is NULL");
    get(result).export_table(path);
  });
}

int32_t mssim_table_arrow(const mssim_result* result, void* out_stream) {
  return guarded([&] {
    require(out_stream != nullptr, "out_stream is NULL");
    auto reader = std::make_shared<arrow::TableBatchReader>(get(result).table());
    msentity::internal::check(arrow::ExportRecordBatchReader(
        reader, static_cast<struct ArrowArrayStream*>(out_stream)));
  });
}

}  // extern "C"
