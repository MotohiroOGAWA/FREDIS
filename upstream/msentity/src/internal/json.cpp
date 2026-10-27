#include "internal/json.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>

namespace msentity::internal {

namespace {

// Decode one UTF-8 code point; invalid bytes are returned as-is (U+FFFD
// semantics are not needed because input comes from valid UTF-8 strings).
uint32_t next_code_point(const std::string& text, std::size_t& i) {
  auto byte = [&](std::size_t k) { return static_cast<unsigned char>(text[k]); };
  unsigned char c = byte(i);
  int extra = (c < 0x80) ? 0 : (c >= 0xF0) ? 3 : (c >= 0xE0) ? 2 : (c >= 0xC0) ? 1 : 0;
  // ASCII, a stray continuation byte, or a truncated sequence: one byte.
  if (extra == 0 || i + static_cast<std::size_t>(extra) >= text.size()) {
    ++i;
    return c;
  }
  uint32_t cp = c & (0x3F >> extra);
  for (int k = 1; k <= extra; ++k) cp = (cp << 6) | (byte(i + static_cast<std::size_t>(k)) & 0x3F);
  i += static_cast<std::size_t>(extra) + 1;
  return cp;
}

void append_u_escape(std::string& out, uint32_t unit) {
  char buffer[8];
  std::snprintf(buffer, sizeof(buffer), "\\u%04x", unit);
  out += buffer;
}

}  // namespace

std::string json_string(const std::string& value) {
  std::string out = "\"";
  std::size_t i = 0;
  while (i < value.size()) {
    uint32_t cp = next_code_point(value, i);
    switch (cp) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (cp < 0x20 || cp > 0x7E) {
          if (cp >= 0x10000) {
            uint32_t v = cp - 0x10000;
            append_u_escape(out, 0xD800 | (v >> 10));
            append_u_escape(out, 0xDC00 | (v & 0x3FF));
          } else {
            append_u_escape(out, cp);
          }
        } else {
          out.push_back(static_cast<char>(cp));
        }
    }
  }
  out += "\"";
  return out;
}

std::string canonical_json(const std::vector<std::string>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) out += ", ";
    out += json_string(values[i]);
  }
  return out + "]";
}

std::string canonical_json(const Attributes& attributes) {
  std::string out = "{";
  for (std::size_t i = 0; i < attributes.size(); ++i) {
    if (i > 0) out += ", ";
    out += json_string(attributes[i].first) + ": " +
           json_string(attributes[i].second);
  }
  return out + "}";
}

namespace {

nlohmann::ordered_json parse(const std::string& text) {
  try {
    return nlohmann::ordered_json::parse(text);
  } catch (const nlohmann::json::exception& error) {
    throw FormatError(std::string("invalid JSON: ") + error.what());
  }
}

std::string as_text(const nlohmann::ordered_json& value) {
  return value.is_string() ? value.get<std::string>() : value.dump();
}

}  // namespace

std::vector<std::string> parse_json_string_list(const std::string& text) {
  auto json = parse(text);
  if (!json.is_array()) throw FormatError("expected a JSON array");
  std::vector<std::string> values;
  for (const auto& item : json) values.push_back(as_text(item));
  return values;
}

Attributes parse_json_attributes(const std::string& text) {
  auto json = parse(text);
  if (!json.is_object()) throw FormatError("expected a JSON object");
  Attributes attributes;
  for (const auto& [key, value] : json.items()) attributes.emplace_back(key, as_text(value));
  return attributes;
}

}  // namespace msentity::internal
