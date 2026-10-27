#include "internal/parquet.hpp"

#include <arrow/io/memory.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>

#include <unordered_set>

#include "internal/table.hpp"
#include "msentity/value.hpp"

namespace msentity::internal {

std::vector<uint8_t> table_to_parquet(const arrow::Table& table) {
  auto sink = unwrap(arrow::io::BufferOutputStream::Create());
  auto properties =
      parquet::WriterProperties::Builder().compression(parquet::Compression::SNAPPY)->build();
  auto arrow_properties = parquet::ArrowWriterProperties::Builder().store_schema()->build();
  check(parquet::arrow::WriteTable(table, arrow::default_memory_pool(), sink,
                                   parquet::DEFAULT_MAX_ROW_GROUP_LENGTH, properties,
                                   arrow_properties));
  auto buffer = unwrap(sink->Finish());
  return std::vector<uint8_t>(buffer->data(), buffer->data() + buffer->size());
}

namespace {

std::shared_ptr<arrow::io::BufferReader> make_reader(const uint8_t* data, std::size_t size) {
  auto buffer = std::make_shared<arrow::Buffer>(data, static_cast<int64_t>(size));
  return std::make_shared<arrow::io::BufferReader>(buffer);
}

// Row-label columns listed in the "pandas" schema metadata key (format spec §4.4).
std::unordered_set<std::string> pandas_index_columns(const arrow::Schema& schema) {
  std::unordered_set<std::string> names;
  auto metadata = schema.metadata();
  if (!metadata) return names;
  int key = metadata->FindKey("pandas");
  if (key < 0) return names;
  auto json = nlohmann::json::parse(metadata->value(key), nullptr, false);
  if (json.is_discarded() || !json.contains("index_columns")) return names;
  for (const auto& item : json["index_columns"]) {
    if (item.is_string()) names.insert(item.get<std::string>());
  }
  return names;
}

}  // namespace

std::shared_ptr<arrow::Table> parquet_to_table(const uint8_t* data, std::size_t size) {
  std::shared_ptr<arrow::Table> table;
  try {
    auto reader = unwrap(parquet::arrow::OpenFile(make_reader(data, size),
                                                  arrow::default_memory_pool()));
    table = unwrap(reader->ReadTable());
  } catch (const parquet::ParquetException& error) {
    throw FormatError(std::string("invalid Parquet data: ") + error.what());
  }

  auto index_columns = pandas_index_columns(*table->schema());
  for (int c = table->num_columns() - 1; c >= 0; --c) {
    if (index_columns.count(table->schema()->field(c)->name()) > 0) {
      table = unwrap(table->RemoveColumn(c));
    }
  }
  return normalize_table(table);
}

int64_t parquet_uncompressed_bytes(const std::vector<uint8_t>& bytes) {
  auto reader = parquet::ParquetFileReader::Open(make_reader(bytes.data(), bytes.size()));
  auto metadata = reader->metadata();
  int64_t total = 0;
  for (int i = 0; i < metadata->num_row_groups(); ++i) {
    total += metadata->RowGroup(i)->total_byte_size();
  }
  return total;
}

}  // namespace msentity::internal
