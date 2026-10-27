#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

namespace msentity {

// A single metadata cell. This is the language-neutral value model used by
// the C++ API and, later, by the C/Java/C# bindings:
//   null    -> std::monostate
//   boolean -> bool
//   integer -> int64_t
//   float   -> double
//   text    -> std::string (UTF-8)
using Value = std::variant<std::monostate, bool, int64_t, double, std::string>;

inline bool is_null(const Value& value) {
  return std::holds_alternative<std::monostate>(value);
}

// Text of a value as written to MSP/MGF/TSV/CSV ("value text" in
// docs/format/text-formats.md): missing and NaN -> "", true -> "True",
// integers in decimal, floats via format_float.
std::string to_display_string(const Value& value);

// Shortest round-trip text of a double: positional notation for
// 1e-4 <= |x| < 1e16 ("100.0", "0.1"), exponent notation otherwise
// ("1e-05", "1e+16"); "nan", "inf", "-inf".
std::string format_float(double value);

// Error categories beyond the standard exceptions (std::invalid_argument for
// invalid arguments/data, std::out_of_range for positions).
class KeyError : public std::out_of_range {
 public:
  using std::out_of_range::out_of_range;
};

class FormatError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace msentity
