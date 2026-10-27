#include <gtest/gtest.h>

#include "io/item_parser.hpp"
#include "io/reader_context.hpp"

namespace msentity::io::detail {
namespace {

TEST(ItemParser, Capitalize) {
  EXPECT_EQ(ItemParser::capitalize("precursor_mz"), "PrecursorMz");
  EXPECT_EQ(ItemParser::capitalize("PrecursorMZ"), "PrecursorMZ");
  EXPECT_EQ(ItemParser::capitalize("NAME"), "NAME");
  EXPECT_EQ(ItemParser::capitalize("Num Peaks"), "NumPeaks");
  EXPECT_EQ(ItemParser::capitalize("collision  energy"), "CollisionEnergy");
  EXPECT_EQ(ItemParser::capitalize("DB#"), "DB#");
}

TEST(ItemParser, CanonicalKeys) {
  ItemParser parser(true, true);
  EXPECT_EQ(parser.to_canonical_key("NAME"), "Name");
  EXPECT_EQ(parser.to_canonical_key("PRECURSORMZ"), "PrecursorMZ");
  EXPECT_EQ(parser.to_canonical_key("PrecursorM/Z"), "PrecursorMZ");
  EXPECT_EQ(parser.to_canonical_key("precursor_type"), "AdductType");
  EXPECT_EQ(parser.to_canonical_key("Num Peaks"), "NumPeaks");
  EXPECT_EQ(parser.to_canonical_key("RT"), "RetentionTime");
  EXPECT_EQ(parser.to_canonical_key("INCHIKEY"), "InChIKey");
  // Unknown keys: the first spelling seen wins within one parser.
  EXPECT_EQ(parser.to_canonical_key("COMMENT"), "COMMENT");
  EXPECT_EQ(parser.to_canonical_key("Comment"), "COMMENT");
  EXPECT_EQ(ItemParser(true, true).to_canonical_key("Comment"), "Comment");
}

TEST(ItemParser, AdductTypes) {
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M+H]+"), "[M+H]+");
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M+H]1+"), "[M+H]+");
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M+H]++"), "[M+H]2+");
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M-2H]--"), "[M-2H]2-");
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M-2H]2-"), "[M-2H]2-");
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M]0+"), "[M]");
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M + Na] +"), "[M+Na]+");
  EXPECT_EQ(ItemParser::normalize_adduct_type("[M+H]x"), "[M+H]x");
  EXPECT_EQ(ItemParser::normalize_adduct_type("M+H"), "M+H");
  ItemParser parser(true, true);
  EXPECT_EQ(parser.parse_item_pair(" Precursor_Type ", " [M+H]1+ \n"),
            (std::pair<std::string, std::string>{"AdductType", "[M+H]+"}));
  ItemParser raw(false, false);
  EXPECT_EQ(raw.parse_item_pair("precursor_type", "[M+H]1+"),
            (std::pair<std::string, std::string>{"precursor_type", "[M+H]1+"}));
}

TEST(PeakLine, ParsesMetadata) {
  EXPECT_TRUE(is_peak_header_line("MZ Intensity ann"));
  EXPECT_FALSE(is_peak_header_line("100 200"));

  PeakEntry entry = parse_peak_line("100.0 200.0", nullptr, "column");
  EXPECT_EQ(entry.mz, 100.0);
  EXPECT_EQ(entry.intensity, 200.0);
  EXPECT_TRUE(entry.metadata.empty());

  std::vector<std::string> columns = {"mz", "intensity", "frag", "note"};
  entry = parse_peak_line("100.0 200.0 fragA ; note1", &columns, "column");
  ASSERT_EQ(entry.metadata.size(), 2u);
  EXPECT_EQ(entry.metadata[0], (std::pair<std::string, std::string>{"frag", "fragA"}));
  EXPECT_EQ(entry.metadata[1], (std::pair<std::string, std::string>{"note", "note1"}));

  entry = parse_peak_line("1 2 \"a;b\" ; 'c' ; x", nullptr, "column");
  ASSERT_EQ(entry.metadata.size(), 3u);
  EXPECT_EQ(entry.metadata[0].second, "a;b");
  EXPECT_EQ(entry.metadata[1], (std::pair<std::string, std::string>{"column2", "c"}));
  EXPECT_EQ(entry.metadata[2].first, "column3");

  EXPECT_THROW(parse_peak_line("100", nullptr, "column"), std::invalid_argument);
  EXPECT_THROW(parse_peak_line("abc 1", nullptr, "column"), std::invalid_argument);
  std::vector<std::string> duplicate = {"mz", "intensity", "mz"};
  EXPECT_THROW(parse_peak_line("1 2 x", &duplicate, "column"), std::invalid_argument);
}

}  // namespace
}  // namespace msentity::io::detail
