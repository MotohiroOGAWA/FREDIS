#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace msentity::io::detail {

// Canonical metadata keys and adduct types (docs/format/text-formats.md §6).
// The cache of first-seen spellings is per instance, i.e. per file read.
class ItemParser {
 public:
  ItemParser(bool canonicalize_column, bool canonicalize_adduct_type);

  // "precursor_mz" -> "PrecursorMz", "NAME" -> "NAME", "Num Peaks" -> "NumPeaks"
  static std::string capitalize(std::string_view name);
  // Lower-case capitalize(key) without "/" (alias lookup key).
  static std::string normalize_key(std::string_view key);
  // "[M+H]1+" -> "[M+H]+", "[M-2H]2-" -> "[M-2H]2-", "[M+H]++" -> "[M+H]2+"
  static std::string normalize_adduct_type(std::string_view value);

  std::string to_canonical_key(std::string_view name);
  std::string to_canonical_adduct_type(std::string_view adduct_type);
  // Strip, canonicalize the key and, for AdductType, the value.
  std::pair<std::string, std::string> parse_item_pair(std::string_view key,
                                                      std::string_view value);

 private:
  bool canonicalize_column_;
  bool canonicalize_adduct_type_;
  std::unordered_map<std::string, std::string> key_map_;
  std::unordered_map<std::string, std::string> adduct_map_;
};

}  // namespace msentity::io::detail
