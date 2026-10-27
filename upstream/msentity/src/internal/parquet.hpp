#pragma once

#include <arrow/api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace msentity::internal {

// Serialize a table to Parquet bytes (Snappy compression, Arrow schema
// stored). No other schema metadata is written.
std::vector<uint8_t> table_to_parquet(const arrow::Table& table);

// Deserialize Parquet bytes. Row-label columns named in the "pandas" schema
// metadata key are dropped and all schema metadata is removed, so the result
// contains only the data columns (format spec §4.4).
std::shared_ptr<arrow::Table> parquet_to_table(const uint8_t* data, std::size_t size);

// Sum of the row groups' uncompressed sizes (Parquet total_byte_size).
int64_t parquet_uncompressed_bytes(const std::vector<uint8_t>& bytes);

}  // namespace msentity::internal
