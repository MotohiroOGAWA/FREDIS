#pragma once

#include <string>

#include "msentity/io/text_formats.hpp"

namespace msentity::io::detail {

enum class TextFormat { Msp, Mgf };

// Shared MSP/MGF writer: the two formats differ only in the record framing
// and the metadata separator.
void write_text_format(const MSDataset& dataset, const std::string& path,
                       const TextWriteOptions& options, TextFormat format);

}  // namespace msentity::io::detail
