#pragma once

#include <cstdint>
#include <string>

#include "msentity/core/ms_dataset.hpp"

namespace msentity::processing {

// Assign zero-padded sequential IDs (prefix + number) to the visible spectra.
// Returns false without changes when the column exists and !overwrite.
bool set_spec_id(MSDataset& dataset, const std::string& col_name = "SpecID",
                 const std::string& prefix = "", bool overwrite = false, int64_t start = 1);

}  // namespace msentity::processing
