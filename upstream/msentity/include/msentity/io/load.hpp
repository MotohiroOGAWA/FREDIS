#pragma once

#include <optional>
#include <string>

#include "msentity/core/ms_dataset.hpp"
#include "msentity/io/text_formats.hpp"

namespace msentity::io {

// Load MSP, MGF, MSDS (.msds/.hdf5/.h5), TSV, or CSV. The type is taken from
// ``file_type`` ("msp", "mgf", "msds", "tsv", "csv") or the file extension.
// ``infer_types`` applies to TSV/CSV (see DelimitedReadOptions).
MSDataset load_ms_dataset(const std::string& path,
                          const std::optional<std::string>& file_type = std::nullopt,
                          const std::optional<std::string>& spec_id_prefix = std::nullopt,
                          const ReadProgressCallback& progress_callback = {},
                          bool infer_types = false);

}  // namespace msentity::io
