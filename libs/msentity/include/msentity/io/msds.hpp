#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "msentity/core/ms_dataset.hpp"

namespace msentity::io {

// Dataset-level metadata stored in the /metadata group of an .msds file.
struct MSDatasetMeta {
  std::string description;
  Attributes attributes;
  std::vector<std::string> tags;
};

struct MsdsWriteOptions {
  // Write the current view (true) or the underlying storage and view index
  // as-is (false).
  bool save_view = true;
  // Append a new dataset_<n> group instead of replacing the file.
  bool append = false;
  // Tables whose Parquet encoding exceeds this size are split into parts.
  int64_t max_part_bytes = 1'000'000'000;
};

// Read/write the .msds format (see docs/format/msds.md).
void write_msds(const MSDataset& dataset, const std::string& path,
                const MsdsWriteOptions& options = {});
MSDataset read_msds(const std::string& path, bool load_peak_metadata = true);
MSDatasetMeta read_msds_meta(const std::string& path);

// In-memory variants: the bytes are a complete .msds file.
std::vector<uint8_t> msds_to_bytes(const MSDataset& dataset,
                                   const MsdsWriteOptions& options = {});
MSDataset msds_from_bytes(const std::vector<uint8_t>& bytes, bool load_peak_metadata = true);

}  // namespace msentity::io
