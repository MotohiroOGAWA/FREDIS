#pragma once

#include <map>
#include <string>
#include <utility>

#include "msentity/value.hpp"

namespace msentity {

// A single peak: m/z, intensity, and optional per-peak metadata.
struct Peak {
  double mz = 0.0;
  double intensity = 0.0;
  // Peak-metadata columns visible in the owning series. Missing values are
  // null.
  std::map<std::string, Value> metadata;

  Peak() = default;
  Peak(double mz_, double intensity_, std::map<std::string, Value> metadata_ = {})
      : mz(mz_), intensity(intensity_), metadata(std::move(metadata_)) {}

  // PrecursorMZ - mz. Throws KeyError when "PrecursorMZ" is not in metadata.
  double neutral_loss() const;
};

}  // namespace msentity
