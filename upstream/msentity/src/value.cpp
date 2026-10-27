#include "msentity/value.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string>

#include "msentity/core/peak.hpp"

namespace msentity {

std::string format_float(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value < 0 ? "-inf" : "inf";

  // Shortest round-trip digits, e.g. "1.2345e+02".
  char buffer[64];
  auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                              std::chars_format::scientific);
  std::string sci(buffer, result.ptr);

  bool negative = false;
  std::size_t pos = 0;
  if (sci[pos] == '-') {
    negative = true;
    ++pos;
  }
  std::size_t e_pos = sci.find('e');
  std::string digits;
  for (std::size_t i = pos; i < e_pos; ++i) {
    if (sci[i] != '.') digits.push_back(sci[i]);
  }
  int exponent = std::atoi(sci.c_str() + e_pos + 1);

  std::string out = negative ? "-" : "";
  // Exponent notation outside [1e-4, 1e16).
  if (exponent >= -4 && exponent < 16) {
    if (exponent >= 0) {
      std::size_t int_len = static_cast<std::size_t>(exponent) + 1;
      if (digits.size() <= int_len) {
        out += digits + std::string(int_len - digits.size(), '0') + ".0";
      } else {
        out += digits.substr(0, int_len) + "." + digits.substr(int_len);
      }
    } else {
      out += "0." + std::string(static_cast<std::size_t>(-exponent - 1), '0') + digits;
    }
    return out;
  }
  out += digits.substr(0, 1);
  if (digits.size() > 1) out += "." + digits.substr(1);
  out += exponent < 0 ? "e-" : "e+";
  std::string exp_digits = std::to_string(std::abs(exponent));
  if (exp_digits.size() < 2) exp_digits = "0" + exp_digits;
  out += exp_digits;
  return out;
}

std::string to_display_string(const Value& value) {
  struct Visitor {
    std::string operator()(std::monostate) const { return ""; }
    std::string operator()(bool v) const { return v ? "True" : "False"; }
    std::string operator()(int64_t v) const { return std::to_string(v); }
    std::string operator()(double v) const {
      // NaN counts as missing and is written as an empty value.
      return std::isnan(v) ? "" : format_float(v);
    }
    std::string operator()(const std::string& v) const { return v; }
  };
  return std::visit(Visitor{}, value);
}

double Peak::neutral_loss() const {
  auto it = metadata.find("PrecursorMZ");
  if (it == metadata.end()) throw KeyError("metadata must contain 'PrecursorMZ'");
  const Value& precursor = it->second;
  if (const auto* d = std::get_if<double>(&precursor)) return *d - mz;
  if (const auto* i = std::get_if<int64_t>(&precursor)) return static_cast<double>(*i) - mz;
  throw std::invalid_argument("'PrecursorMZ' must be numeric");
}

}  // namespace msentity
