#pragma once

#include <string>
#include <vector>

#include "msentity/core/ms_dataset.hpp"

namespace msentity::internal {

// Canonical JSON (docs/format/msds.md §3.5): ", " / ": " separators and
// non-ASCII characters escaped as \uXXXX, so equal metadata gives equal bytes.
std::string canonical_json(const std::vector<std::string>& values);
std::string canonical_json(const Attributes& attributes);
std::string json_string(const std::string& value);

// Parsers. Non-string items are converted with their JSON text. Throw
// msentity::FormatError on malformed input.
std::vector<std::string> parse_json_string_list(const std::string& text);
Attributes parse_json_attributes(const std::string& text);

}  // namespace msentity::internal
