#include "internal/text.hpp"

#include <cctype>
#include <cstdint>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace msentity::internal {

std::string_view strip(std::string_view text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && is_space(text[begin])) ++begin;
  while (end > begin && is_space(text[end - 1])) --end;
  return text.substr(begin, end - begin);
}

std::string_view rstrip_newline(std::string_view text) {
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
  return text;
}

std::vector<std::string_view> split_whitespace(std::string_view text, int maxsplit) {
  std::vector<std::string_view> parts;
  std::size_t i = 0;
  const std::size_t n = text.size();
  while (i < n) {
    while (i < n && is_space(text[i])) ++i;
    if (i >= n) break;
    if (maxsplit >= 0 && static_cast<int>(parts.size()) == maxsplit) {
      // The remainder keeps its internal and trailing
      // whitespace.
      parts.push_back(text.substr(i));
      return parts;
    }
    std::size_t start = i;
    while (i < n && !is_space(text[i])) ++i;
    parts.push_back(text.substr(start, i - start));
  }
  return parts;
}

std::vector<std::string_view> split_char(std::string_view text, char separator) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || text[i] == separator) {
      parts.push_back(text.substr(start, i - start));
      start = i + 1;
    }
  }
  return parts;
}

std::string to_lower_ascii(std::string_view text) {
  std::string out(text);
  for (char& c : out) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return out;
}

std::string to_upper_ascii(std::string_view text) {
  std::string out(text);
  for (char& c : out) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  }
  return out;
}

bool iequals_ascii(std::string_view a, std::string_view b) {
  return a.size() == b.size() && to_lower_ascii(a) == to_lower_ascii(b);
}

bool ends_with(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

std::string replace_all(std::string_view text, std::string_view from, std::string_view to) {
  std::string out;
  if (from.empty()) return std::string(text);
  std::size_t pos = 0;
  while (true) {
    std::size_t hit = text.find(from, pos);
    if (hit == std::string_view::npos) {
      out.append(text.substr(pos));
      return out;
    }
    out.append(text.substr(pos, hit - pos));
    out.append(to);
    pos = hit + from.size();
  }
}

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Remove PEP 515 underscores ("1_000.5"); every underscore must sit between
// two digits.
std::optional<std::string> remove_digit_underscores(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '_') {
      if (i == 0 || i + 1 >= text.size() || !is_digit(text[i - 1]) || !is_digit(text[i + 1])) {
        return std::nullopt;
      }
      continue;
    }
    out.push_back(text[i]);
  }
  return out;
}

}  // namespace

std::optional<double> parse_number(std::string_view text) {
  std::string_view body = strip(text);
  if (body.empty()) return std::nullopt;
  bool negative = false;
  if (body.front() == '+' || body.front() == '-') {
    negative = body.front() == '-';
    body.remove_prefix(1);
  }
  if (body.empty()) return std::nullopt;
  std::string lowered = to_lower_ascii(body);
  if (lowered == "inf" || lowered == "infinity") {
    return negative ? -HUGE_VAL : HUGE_VAL;
  }
  if (lowered == "nan") return std::nan("");
  if (!is_digit(body.front()) && body.front() != '.') return std::nullopt;
  auto cleaned = remove_digit_underscores(body);
  if (!cleaned) return std::nullopt;
  for (char c : *cleaned) {
    if (!(is_digit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')) {
      return std::nullopt;
    }
  }
  double value = 0.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
  const char* first = cleaned->data();
  const char* last = first + cleaned->size();
  auto [ptr, ec] = std::from_chars(first, last, value, std::chars_format::general);
  if (ptr != last) return std::nullopt;
  if (ec == std::errc::result_out_of_range) {
    // Out of range: return inf or 0.0 instead of failing.
    value = std::strtod(cleaned->c_str(), nullptr);
  } else if (ec != std::errc()) {
    return std::nullopt;
  }
#else
  // Standard libraries without floating-point from_chars (libc++). The text
  // contains only digits, '.', exponent, and signs, so the C locale's decimal
  // point is the only locale-dependent part; the library never changes it.
  char* end = nullptr;
  value = std::strtod(cleaned->c_str(), &end);
  if (end != cleaned->c_str() + cleaned->size()) return std::nullopt;
#endif
  return negative ? -value : value;
}

std::optional<long long> parse_int64(std::string_view text) {
  std::string_view body = strip(text);
  if (body.empty()) return std::nullopt;
  bool negative = false;
  if (body.front() == '+' || body.front() == '-') {
    negative = body.front() == '-';
    body.remove_prefix(1);
  }
  if (body.empty()) return std::nullopt;
  for (char c : body) {
    if (!is_digit(c)) return std::nullopt;
  }
  unsigned long long magnitude = 0;
  auto [ptr, ec] = std::from_chars(body.data(), body.data() + body.size(), magnitude);
  if (ec != std::errc() || ptr != body.data() + body.size()) return std::nullopt;
  const unsigned long long limit = negative ? 9223372036854775808ULL : 9223372036854775807ULL;
  if (magnitude > limit) return std::nullopt;
  if (negative) {
    return magnitude == 9223372036854775808ULL ? INT64_MIN : -static_cast<long long>(magnitude);
  }
  return static_cast<long long>(magnitude);
}

std::string format_g17(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value < 0 ? "-inf" : "inf";
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

std::vector<std::string_view> split_lines(std::string_view text) {
  std::vector<std::string_view> lines;
  std::size_t start = 0;
  std::size_t i = 0;
  while (i < text.size()) {
    if (text[i] == '\n' || text[i] == '\r') {
      lines.push_back(text.substr(start, i - start));
      if (text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
      ++i;
      start = i;
    } else {
      ++i;
    }
  }
  if (start < text.size()) lines.push_back(text.substr(start));
  return lines;
}

std::string normalize_newlines(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\r') {
      out.push_back('\n');
      if (i + 1 < text.size() && text[i + 1] == '\n') ++i;
    } else {
      out.push_back(text[i]);
    }
  }
  return out;
}

std::string read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open file: " + path);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

}  // namespace msentity::internal
