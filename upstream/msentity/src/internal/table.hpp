#pragma once

// Arrow helpers built only on the core Arrow library (no arrow_compute), so
// the binary dependency footprint stays small for the language bindings.

#include <arrow/api.h>

#include <memory>
#include <string>
#include <vector>

#include "msentity/value.hpp"

namespace msentity::internal {

void check(const arrow::Status& status);

template <typename T>
T unwrap(arrow::Result<T> result) {
  check(result.status());
  return std::move(result).ValueUnsafe();
}

// Cell access ------------------------------------------------------------
Value value_at(const arrow::Array& array, int64_t row);
std::vector<Value> to_values(const arrow::ChunkedArray& column);
std::vector<Value> to_values(const arrow::Array& array);

// Build an array from values, inferring the type:
//   only strings -> utf8, only bools -> bool, ints -> int64,
//   ints/floats -> float64, any mix with strings -> utf8 (to_display_string),
//   only nulls -> utf8. NaN becomes null.
std::shared_ptr<arrow::Array> values_to_array(const std::vector<Value>& values);
// Non-null utf8 array.
std::shared_ptr<arrow::Array> strings_to_array(const std::vector<std::string>& values);

// Arrays / tables ------------------------------------------------------------
std::shared_ptr<arrow::Array> as_single_array(const arrow::ChunkedArray& column);
std::shared_ptr<arrow::Array> decode_dictionary(const arrow::Array& array);
std::shared_ptr<arrow::Array> take(const arrow::Array& array, const std::vector<int64_t>& rows);
std::shared_ptr<arrow::Table> take_rows(const arrow::Table& table,
                                        const std::vector<int64_t>& rows);
std::shared_ptr<arrow::Table> select_columns(const arrow::Table& table,
                                             const std::vector<std::string>& names);
std::shared_ptr<arrow::Table> slice_rows(const arrow::Table& table, int64_t offset,
                                         int64_t length);
std::shared_ptr<arrow::Table> put_column(const std::shared_ptr<arrow::Table>& table,
                                         const std::string& name,
                                         const std::shared_ptr<arrow::Array>& array);
std::shared_ptr<arrow::Table> make_empty_table(int64_t num_rows);
// Union of columns in first-seen order; missing cells become null and
// conflicting column types are unified via values_to_array().
std::shared_ptr<arrow::Table> concat_tables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables);
// Single chunk per column, dictionaries decoded, schema metadata removed.
std::shared_ptr<arrow::Table> normalize_table(const std::shared_ptr<arrow::Table>& table);

int column_index(const arrow::Table& table, const std::string& name);
std::vector<std::string> column_names(const arrow::Table& table);

}  // namespace msentity::internal
