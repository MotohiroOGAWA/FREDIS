#include "io/item_parser.hpp"

#include <optional>
#include <vector>

#include "internal/text.hpp"

namespace msentity::io::detail {

namespace {

struct Alias {
  const char* canonical;
  std::vector<const char*> aliases;
};

const std::vector<Alias>& column_aliases() {
  static const std::vector<Alias> aliases = {
      {"Name", {}},
      {"Formula", {}},
      {"InChIKey", {}},
      {"InChI", {}},
      {"PrecursorMZ", {"PrecursorM/Z"}},
      {"AdductType", {"PrecursorType"}},
      {"SpectrumType", {}},
      {"InstrumentType", {}},
      {"Instrument", {}},
      {"IonMode", {}},
      {"CollisionEnergy", {}},
      {"ExactMass", {}},
      {"SMILES", {}},
      {"RetentionTime", {"RT"}},
      {"Ontology", {}},
      {"NumPeaks", {}},
  };
  return aliases;
}

const std::vector<const char*>& canonical_adduct_types() {
  static const std::vector<const char*> adducts = {"[M]+",     "[M+H]+", "[M-H]-",
                                                   "[M+Na]+", "[M+K]+", "[M+NH4]+"};
  return adducts;
}

bool is_upper(char c) { return c >= 'A' && c <= 'Z'; }
bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool all_digits(std::string_view text) {
  if (text.empty()) return false;  // str.isdigit() of "" is False
  for (char c : text) {
    if (!is_digit(c)) return false;
  }
  return true;
}

// Matches the regular expression \d*[+-]+ against the whole text.
bool is_charge_suffix(std::string_view text) {
  std::size_t i = 0;
  while (i < text.size() && is_digit(text[i])) ++i;
  if (i == text.size()) return false;
  for (; i < text.size(); ++i) {
    if (text[i] != '+' && text[i] != '-') return false;
  }
  return true;
}

std::string remove_spaces(std::string_view text) {
  std::string out;
  for (char c : text) {
    if (c != ' ') out.push_back(c);
  }
  return std::string(internal::strip(out));
}

}  // namespace

ItemParser::ItemParser(bool canonicalize_column, bool canonicalize_adduct_type)
    : canonicalize_column_(canonicalize_column),
      canonicalize_adduct_type_(canonicalize_adduct_type) {
  for (const auto& alias : column_aliases()) {
    key_map_[normalize_key(alias.canonical)] = alias.canonical;
    for (const char* name : alias.aliases) {
      key_map_.emplace(normalize_key(name), alias.canonical);  // setdefault
    }
  }
  for (const char* adduct : canonical_adduct_types()) {
    adduct_map_[normalize_adduct_type(adduct)] = adduct;
  }
}

std::string ItemParser::capitalize(std::string_view name) {
  // re.split(r'[_\s]+|(?=[A-Z])', name), then part.capitalize() for each
  // non-empty part.
  std::vector<std::string> parts(1);
  for (char c : name) {
    if (c == '_' || internal::is_space(c)) {
      parts.emplace_back();
    } else if (is_upper(c)) {
      parts.emplace_back(1, c);
    } else {
      parts.back().push_back(c);
    }
  }
  std::string result;
  for (auto& part : parts) {
    if (part.empty()) continue;
    std::string lowered = internal::to_lower_ascii(part);
    lowered[0] = internal::to_upper_ascii(std::string_view(&lowered[0], 1))[0];
    result += lowered;
  }
  return result;
}

std::string ItemParser::normalize_key(std::string_view key) {
  std::string capitalized = capitalize(key);
  std::string out;
  for (char c : capitalized) {
    if (c != '/') out.push_back(c);
  }
  return internal::to_lower_ascii(out);
}

std::string ItemParser::normalize_adduct_type(std::string_view value) {
  // re.match(r"\[(.*?)\]", value)
  if (value.empty() || value.front() != '[') return remove_spaces(value);
  std::size_t close = value.find(']');
  if (close == std::string_view::npos) return remove_spaces(value);
  std::string_view core = value.substr(1, close - 1);

  // re.search(r"\](\d*[+-]+)$", value): first "]" followed by a charge.
  std::optional<std::string_view> raw;
  for (std::size_t p = value.find(']'); p != std::string_view::npos; p = value.find(']', p + 1)) {
    if (is_charge_suffix(value.substr(p + 1))) {
      raw = value.substr(p + 1);
      break;
    }
  }

  std::string charge;
  if (!raw) {
    charge = std::string(internal::strip(value.substr(close + 1)));
  } else {
    std::string_view r = *raw;
    std::optional<long long> number;
    bool all_plus = r.find_first_not_of('+') == std::string_view::npos;
    bool all_minus = r.find_first_not_of('-') == std::string_view::npos;
    std::string_view head = r.substr(0, r.size() - 1);
    if (all_plus) {
      number = static_cast<long long>(r.size());
    } else if (all_minus) {
      number = -static_cast<long long>(r.size());
    } else if (r.back() == '+' || r.back() == '-') {
      long long sign = r.back() == '+' ? 1 : -1;
      if (!all_digits(head)) {
        number = sign;
      } else if (auto parsed = internal::parse_int64(head)) {
        number = sign * *parsed;
      }  // else: too large for int64; keep the raw charge text
    }
    if (!number) {
      charge = std::string(r);
    } else if (*number == 0) {
      charge = "";
    } else if (*number == 1) {
      charge = "+";
    } else if (*number == -1) {
      charge = "-";
    } else if (*number > 1) {
      charge = std::to_string(*number) + "+";
    } else {
      charge = std::to_string(-*number) + "-";
    }
  }
  return remove_spaces("[" + std::string(core) + "]" + charge);
}

std::string ItemParser::to_canonical_key(std::string_view name) {
  std::string normalized = normalize_key(name);
  auto it = key_map_.find(normalized);
  if (it == key_map_.end()) it = key_map_.emplace(normalized, capitalize(name)).first;
  return it->second;
}

std::string ItemParser::to_canonical_adduct_type(std::string_view adduct_type) {
  std::string normalized = normalize_adduct_type(adduct_type);
  auto it = adduct_map_.find(normalized);
  if (it == adduct_map_.end()) it = adduct_map_.emplace(normalized, normalized).first;
  return it->second;
}

std::pair<std::string, std::string> ItemParser::parse_item_pair(std::string_view key,
                                                                std::string_view value) {
  key = internal::strip(key);
  value = internal::strip(value);
  std::string canonical_key = canonicalize_column_ ? to_canonical_key(key) : std::string(key);
  std::string parsed_value(value);
  if (canonical_key == "AdductType" && canonicalize_adduct_type_) {
    parsed_value = to_canonical_adduct_type(value);
  }
  return {canonical_key, parsed_value};
}

}  // namespace msentity::io::detail
