#pragma once

// Text helpers implementing the conventions of docs/format/text-formats.md §8
// (whitespace, stripping, splitting, number syntax).

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace msentity::internal {

// Whitespace: space, \t \n \r \v \f and U+001C-U+001F.
inline bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' ||
         (c >= '\x1c' && c <= '\x1f');
}

std::string_view strip(std::string_view text);
std::string_view rstrip_newline(std::string_view text);
// Split at runs of whitespace (leading/trailing whitespace ignored). With
// maxsplit >= 0, at most maxsplit splits; the remainder is kept as is.
std::vector<std::string_view> split_whitespace(std::string_view text, int maxsplit = -1);
std::vector<std::string_view> split_char(std::string_view text, char separator);

std::string to_lower_ascii(std::string_view text);
std::string to_upper_ascii(std::string_view text);
bool iequals_ascii(std::string_view a, std::string_view b);
bool ends_with(std::string_view text, std::string_view suffix);
std::string replace_all(std::string_view text, std::string_view from, std::string_view to);

// Number syntax: surrounding whitespace, sign, digits with underscores,
// "inf"/"infinity"/"nan". Returns nullopt for invalid input.
std::optional<double> parse_number(std::string_view text);
// Integer literal with optional sign and surrounding whitespace.
std::optional<long long> parse_int64(std::string_view text);

// printf("%.17g")
std::string format_g17(double value);

// Split text into lines at \n, \r\n, or \r; the
// returned lines do not include the terminator.
std::vector<std::string_view> split_lines(std::string_view text);
// "\r\n" and "\r" -> "\n".
std::string normalize_newlines(std::string_view text);

std::string read_file(const std::string& path);

}  // namespace msentity::internal
