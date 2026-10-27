#include "internal/table.hpp"

#include <arrow/array/concatenate.h>
#include <arrow/array/util.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace msentity::internal {

void check(const arrow::Status& status) {
  if (!status.ok()) throw std::runtime_error("Arrow error: " + status.ToString());
}

Value value_at(const arrow::Array& array, int64_t row) {
  if (row < 0 || row >= array.length()) throw std::out_of_range("row out of range");
  if (array.IsNull(row)) return std::monostate{};
  switch (array.type_id()) {
    case arrow::Type::NA:
      return std::monostate{};
    case arrow::Type::BOOL:
      return static_cast<const arrow::BooleanArray&>(array).Value(row);
    case arrow::Type::INT8:
      return static_cast<int64_t>(static_cast<const arrow::Int8Array&>(array).Value(row));
    case arrow::Type::INT16:
      return static_cast<int64_t>(static_cast<const arrow::Int16Array&>(array).Value(row));
    case arrow::Type::INT32:
      return static_cast<int64_t>(static_cast<const arrow::Int32Array&>(array).Value(row));
    case arrow::Type::INT64:
      return static_cast<int64_t>(static_cast<const arrow::Int64Array&>(array).Value(row));
    case arrow::Type::UINT8:
      return static_cast<int64_t>(static_cast<const arrow::UInt8Array&>(array).Value(row));
    case arrow::Type::UINT16:
      return static_cast<int64_t>(static_cast<const arrow::UInt16Array&>(array).Value(row));
    case arrow::Type::UINT32:
      return static_cast<int64_t>(static_cast<const arrow::UInt32Array&>(array).Value(row));
    case arrow::Type::UINT64: {
      uint64_t v = static_cast<const arrow::UInt64Array&>(array).Value(row);
      if (v <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return static_cast<int64_t>(v);
      }
      return static_cast<double>(v);
    }
    case arrow::Type::FLOAT:
      return static_cast<double>(static_cast<const arrow::FloatArray&>(array).Value(row));
    case arrow::Type::DOUBLE:
      return static_cast<const arrow::DoubleArray&>(array).Value(row);
    case arrow::Type::STRING:
      return std::string(static_cast<const arrow::StringArray&>(array).GetView(row));
    case arrow::Type::LARGE_STRING:
      return std::string(static_cast<const arrow::LargeStringArray&>(array).GetView(row));
    case arrow::Type::STRING_VIEW:
      return std::string(static_cast<const arrow::StringViewArray&>(array).GetView(row));
    case arrow::Type::BINARY:
      return std::string(static_cast<const arrow::BinaryArray&>(array).GetView(row));
    case arrow::Type::LARGE_BINARY:
      return std::string(static_cast<const arrow::LargeBinaryArray&>(array).GetView(row));
    case arrow::Type::DICTIONARY: {
      const auto& dict = static_cast<const arrow::DictionaryArray&>(array);
      return value_at(*dict.dictionary(), dict.GetValueIndex(row));
    }
    default: {
      auto scalar = unwrap(array.GetScalar(row));
      return scalar->ToString();
    }
  }
}

std::vector<Value> to_values(const arrow::Array& array) {
  std::vector<Value> values;
  values.reserve(static_cast<std::size_t>(array.length()));
  for (int64_t i = 0; i < array.length(); ++i) values.push_back(value_at(array, i));
  return values;
}

std::vector<Value> to_values(const arrow::ChunkedArray& column) {
  std::vector<Value> values;
  values.reserve(static_cast<std::size_t>(column.length()));
  for (const auto& chunk : column.chunks()) {
    for (int64_t i = 0; i < chunk->length(); ++i) values.push_back(value_at(*chunk, i));
  }
  return values;
}

namespace {

bool is_nan_value(const Value& value) {
  const auto* d = std::get_if<double>(&value);
  return d != nullptr && std::isnan(*d);
}

}  // namespace

std::shared_ptr<arrow::Array> values_to_array(const std::vector<Value>& values) {
  bool has_bool = false, has_int = false, has_double = false, has_string = false;
  for (const auto& value : values) {
    if (is_null(value) || is_nan_value(value)) continue;
    if (std::holds_alternative<bool>(value)) has_bool = true;
    if (std::holds_alternative<int64_t>(value)) has_int = true;
    if (std::holds_alternative<double>(value)) has_double = true;
    if (std::holds_alternative<std::string>(value)) has_string = true;
  }

  if (has_string || !(has_bool || has_int || has_double)) {
    arrow::StringBuilder builder;
    check(builder.Reserve(static_cast<int64_t>(values.size())));
    for (const auto& value : values) {
      if (is_null(value) || is_nan_value(value)) {
        check(builder.AppendNull());
      } else {
        check(builder.Append(to_display_string(value)));
      }
    }
    return unwrap(builder.Finish());
  }
  if (has_bool && !has_int && !has_double) {
    arrow::BooleanBuilder builder;
    for (const auto& value : values) {
      if (is_null(value)) {
        check(builder.AppendNull());
      } else {
        check(builder.Append(std::get<bool>(value)));
      }
    }
    return unwrap(builder.Finish());
  }
  auto as_double = [](const Value& value) -> double {
    if (const auto* b = std::get_if<bool>(&value)) return *b ? 1.0 : 0.0;
    if (const auto* i = std::get_if<int64_t>(&value)) return static_cast<double>(*i);
    return std::get<double>(value);
  };
  if (!has_double) {
    arrow::Int64Builder builder;
    for (const auto& value : values) {
      if (is_null(value)) {
        check(builder.AppendNull());
      } else if (const auto* b = std::get_if<bool>(&value)) {
        check(builder.Append(*b ? 1 : 0));
      } else {
        check(builder.Append(std::get<int64_t>(value)));
      }
    }
    return unwrap(builder.Finish());
  }
  arrow::DoubleBuilder builder;
  for (const auto& value : values) {
    if (is_null(value) || is_nan_value(value)) {
      check(builder.AppendNull());
    } else {
      check(builder.Append(as_double(value)));
    }
  }
  return unwrap(builder.Finish());
}

std::shared_ptr<arrow::Array> strings_to_array(const std::vector<std::string>& values) {
  arrow::StringBuilder builder;
  int64_t total = 0;
  for (const auto& v : values) total += static_cast<int64_t>(v.size());
  check(builder.Reserve(static_cast<int64_t>(values.size())));
  check(builder.ReserveData(total));
  for (const auto& v : values) builder.UnsafeAppend(v);
  return unwrap(builder.Finish());
}

std::shared_ptr<arrow::Array> as_single_array(const arrow::ChunkedArray& column) {
  if (column.num_chunks() == 1) return column.chunk(0);
  if (column.num_chunks() == 0) return unwrap(arrow::MakeEmptyArray(column.type()));
  return unwrap(arrow::Concatenate(column.chunks()));
}

std::shared_ptr<arrow::Array> decode_dictionary(const arrow::Array& array) {
  const auto& dict = static_cast<const arrow::DictionaryArray&>(array);
  std::vector<int64_t> rows;
  rows.reserve(static_cast<std::size_t>(dict.length()));
  // Build with nulls preserved: take() on the dictionary handles -1 as null.
  for (int64_t i = 0; i < dict.length(); ++i) {
    rows.push_back(dict.IsNull(i) ? -1 : dict.GetValueIndex(i));
  }
  return take(*dict.dictionary(), rows);
}

std::shared_ptr<arrow::Array> take(const arrow::Array& array, const std::vector<int64_t>& rows) {
  if (array.type_id() == arrow::Type::DICTIONARY) {
    return take(*decode_dictionary(array), rows);
  }
  std::unique_ptr<arrow::ArrayBuilder> builder;
  check(arrow::MakeBuilder(arrow::default_memory_pool(), array.type(), &builder));
  check(builder->Reserve(static_cast<int64_t>(rows.size())));
  arrow::ArraySpan span(*array.data());
  std::size_t i = 0;
  while (i < rows.size()) {
    int64_t start = rows[i];
    if (start < 0) {
      check(builder->AppendNull());
      ++i;
      continue;
    }
    if (start >= array.length()) throw std::out_of_range("take: row index out of range");
    // Coalesce runs of consecutive rows into one slice append.
    std::size_t j = i + 1;
    while (j < rows.size() && rows[j] == rows[j - 1] + 1) ++j;
    int64_t length = static_cast<int64_t>(j - i);
    if (start + length > array.length()) throw std::out_of_range("take: row index out of range");
    check(builder->AppendArraySlice(span, start, length));
    i = j;
  }
  std::shared_ptr<arrow::Array> out;
  check(builder->Finish(&out));
  return out;
}

std::shared_ptr<arrow::Table> take_rows(const arrow::Table& table,
                                        const std::vector<int64_t>& rows) {
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(static_cast<std::size_t>(table.num_columns()));
  for (int c = 0; c < table.num_columns(); ++c) {
    columns.push_back(take(*as_single_array(*table.column(c)), rows));
  }
  return arrow::Table::Make(table.schema(), columns, static_cast<int64_t>(rows.size()));
}

std::shared_ptr<arrow::Table> select_columns(const arrow::Table& table,
                                             const std::vector<std::string>& names) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  for (const auto& name : names) {
    int index = column_index(table, name);
    if (index < 0) throw KeyError("column not found: " + name);
    fields.push_back(table.schema()->field(index));
    columns.push_back(table.column(index));
  }
  return arrow::Table::Make(arrow::schema(fields), columns, table.num_rows());
}

std::shared_ptr<arrow::Table> slice_rows(const arrow::Table& table, int64_t offset,
                                         int64_t length) {
  return normalize_table(table.Slice(offset, length));
}

std::shared_ptr<arrow::Table> put_column(const std::shared_ptr<arrow::Table>& table,
                                         const std::string& name,
                                         const std::shared_ptr<arrow::Array>& array) {
  auto field = arrow::field(name, array->type());
  auto chunked = std::make_shared<arrow::ChunkedArray>(array);
  int index = column_index(*table, name);
  if (index >= 0) return unwrap(table->SetColumn(index, field, chunked));
  return unwrap(table->AddColumn(table->num_columns(), field, chunked));
}

std::shared_ptr<arrow::Table> make_empty_table(int64_t num_rows) {
  return arrow::Table::Make(arrow::schema(arrow::FieldVector{}),
                            std::vector<std::shared_ptr<arrow::Array>>{}, num_rows);
}

std::shared_ptr<arrow::Table> concat_tables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables) {
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  int64_t total_rows = 0;
  for (const auto& table : tables) {
    total_rows += table->num_rows();
    for (const auto& name : column_names(*table)) {
      if (seen.insert(name).second) names.push_back(name);
    }
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> columns;
  for (const auto& name : names) {
    std::shared_ptr<arrow::DataType> common;
    bool conflict = false;
    for (const auto& table : tables) {
      int index = column_index(*table, name);
      if (index < 0) continue;
      auto type = table->column(index)->type();
      if (type->id() == arrow::Type::DICTIONARY) {
        type = static_cast<const arrow::DictionaryType&>(*type).value_type();
      }
      if (type->id() == arrow::Type::NA) continue;
      if (!common) {
        common = type;
      } else if (!common->Equals(*type)) {
        conflict = true;
      }
    }

    std::shared_ptr<arrow::Array> column;
    if (conflict || !common) {
      std::vector<Value> values;
      values.reserve(static_cast<std::size_t>(total_rows));
      for (const auto& table : tables) {
        int index = column_index(*table, name);
        if (index < 0) {
          values.insert(values.end(), static_cast<std::size_t>(table->num_rows()),
                        std::monostate{});
        } else {
          auto part = to_values(*table->column(index));
          values.insert(values.end(), part.begin(), part.end());
        }
      }
      column = values_to_array(values);
    } else {
      arrow::ArrayVector parts;
      for (const auto& table : tables) {
        int index = column_index(*table, name);
        if (index < 0 || table->column(index)->type()->id() == arrow::Type::NA) {
          parts.push_back(unwrap(arrow::MakeArrayOfNull(common, table->num_rows())));
        } else {
          auto array = as_single_array(*table->column(index));
          if (array->type_id() == arrow::Type::DICTIONARY) array = decode_dictionary(*array);
          parts.push_back(array);
        }
      }
      column = parts.empty() ? unwrap(arrow::MakeEmptyArray(common))
                             : unwrap(arrow::Concatenate(parts));
    }
    fields.push_back(arrow::field(name, column->type()));
    columns.push_back(column);
  }
  return arrow::Table::Make(arrow::schema(fields), columns, total_rows);
}

std::shared_ptr<arrow::Table> normalize_table(const std::shared_ptr<arrow::Table>& table) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> columns;
  for (int c = 0; c < table->num_columns(); ++c) {
    auto array = as_single_array(*table->column(c));
    if (array->type_id() == arrow::Type::DICTIONARY) array = decode_dictionary(*array);
    fields.push_back(arrow::field(table->schema()->field(c)->name(), array->type()));
    columns.push_back(array);
  }
  return arrow::Table::Make(arrow::schema(fields), columns, table->num_rows());
}

int column_index(const arrow::Table& table, const std::string& name) {
  const auto& fields = table.schema()->fields();
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (fields[i]->name() == name) return static_cast<int>(i);
  }
  return -1;
}

std::vector<std::string> column_names(const arrow::Table& table) {
  std::vector<std::string> names;
  for (const auto& field : table.schema()->fields()) names.push_back(field->name());
  return names;
}

}  // namespace msentity::internal
