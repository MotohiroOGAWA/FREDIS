#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "internal/json.hpp"
#include "internal/text.hpp"
#include "msentity/value.hpp"

namespace msentity {
namespace {

TEST(FormatFloat, ShortestRoundTripText) {
  EXPECT_EQ(format_float(100.0), "100.0");
  EXPECT_EQ(format_float(0.1), "0.1");
  EXPECT_EQ(format_float(-0.0), "-0.0");
  EXPECT_EQ(format_float(0.0), "0.0");
  EXPECT_EQ(format_float(1e16), "1e+16");
  EXPECT_EQ(format_float(1234567890123456.0), "1234567890123456.0");
  EXPECT_EQ(format_float(0.0001), "0.0001");
  EXPECT_EQ(format_float(0.00001), "1e-05");
  EXPECT_EQ(format_float(1.5e-5), "1.5e-05");
  EXPECT_EQ(format_float(1e100), "1e+100");
  EXPECT_EQ(format_float(91.0542), "91.0542");
  EXPECT_EQ(format_float(0.24524524524524524), "0.24524524524524524");
  EXPECT_EQ(format_float(123456789012345678.0), "1.2345678901234568e+17");
  EXPECT_EQ(format_float(std::nan("")), "nan");
  EXPECT_EQ(format_float(-std::numeric_limits<double>::infinity()), "-inf");
}

TEST(DisplayString, ValueText) {
  EXPECT_EQ(to_display_string(std::monostate{}), "");
  EXPECT_EQ(to_display_string(std::nan("")), "");
  EXPECT_EQ(to_display_string(true), "True");
  EXPECT_EQ(to_display_string(int64_t{-3}), "-3");
  EXPECT_EQ(to_display_string(2.5), "2.5");
  EXPECT_EQ(to_display_string(std::string("x")), "x");
}

TEST(ParseNumber, AcceptsNumberSyntax) {
  using internal::parse_number;
  EXPECT_EQ(parse_number(" 1.5 "), 1.5);
  EXPECT_EQ(parse_number("+2"), 2.0);
  EXPECT_EQ(parse_number("-.5"), -0.5);
  EXPECT_EQ(parse_number("5."), 5.0);
  EXPECT_EQ(parse_number("1e3"), 1000.0);
  EXPECT_EQ(parse_number("1_000.5"), 1000.5);
  EXPECT_TRUE(std::isinf(*parse_number("-Infinity")));
  EXPECT_TRUE(std::isnan(*parse_number("NaN")));
  EXPECT_FALSE(parse_number("abc"));
  EXPECT_FALSE(parse_number("1e"));
  EXPECT_FALSE(parse_number("0x10"));
  EXPECT_FALSE(parse_number("1__0"));
  EXPECT_FALSE(parse_number(""));
}

TEST(SplitWhitespace, MaxSplitKeepsRemainder) {
  auto parts = internal::split_whitespace("  a  b   c d ", 2);
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0], "a");
  EXPECT_EQ(parts[1], "b");
  EXPECT_EQ(parts[2], "c d ");
  EXPECT_EQ(internal::split_whitespace(" a\tb\n").size(), 2u);
}

TEST(CanonicalJson, EscapesAndSeparators) {
  EXPECT_EQ(internal::canonical_json(std::vector<std::string>{"a", "b"}), "[\"a\", \"b\"]");
  EXPECT_EQ(internal::canonical_json(std::vector<std::string>{}), "[]");
  EXPECT_EQ(internal::canonical_json(Attributes{{"k", "v"}, {"x", "y"}}),
            "{\"k\": \"v\", \"x\": \"y\"}");
  // ensure_ascii=True, including astral characters as surrogate pairs.
  EXPECT_EQ(internal::json_string("é\"\\\n😀"),
            "\"\\u00e9\\\"\\\\\\n\\ud83d\\ude00\"");
  auto attributes = internal::parse_json_attributes("{\"a\": \"1\", \"b\": 2}");
  ASSERT_EQ(attributes.size(), 2u);
  EXPECT_EQ(attributes[1], (std::pair<std::string, std::string>{"b", "2"}));
  EXPECT_THROW(internal::parse_json_string_list("{"), FormatError);
}

}  // namespace
}  // namespace msentity
