// Ports of tests/TestCore/TestMSDataset.py (merge_metadata is not ported).

#include <gtest/gtest.h>

#include "test_helpers.hpp"

namespace msentity {
namespace {

using testing::as_strings;
using testing::make_table;
using testing::sample_dataset;
using testing::strings;

std::vector<std::string> names(const MSDataset& dataset) {
  return as_strings(dataset.column_values("name"));
}

TEST(MSDataset, ShapeAndProperties) {
  MSDataset dataset = sample_dataset();
  EXPECT_EQ(dataset.size(), 3);
  EXPECT_EQ(dataset.n_columns(), 3);
  EXPECT_EQ(dataset.n_peaks_total(), 6);
  EXPECT_EQ(dataset.columns(), (std::vector<std::string>{"name", "precursor_mz", "SMILES"}));
  EXPECT_EQ(dataset.description(), "test dataset");
  EXPECT_EQ(dataset.attribute("source"), "unit-test");
  EXPECT_EQ(dataset.tags(), (std::vector<std::string>{"tag1"}));
  EXPECT_EQ(dataset.metadata()->num_rows(), 3);
  EXPECT_EQ(dataset.peaks().size(), 3);
}

TEST(MSDataset, RejectsMismatchedRows) {
  EXPECT_THROW(MSDataset(make_table({{"a", strings({"x"})}}), testing::sample_peak_series()),
               std::invalid_argument);
  EXPECT_THROW(MSDataset(nullptr, testing::sample_peak_series(),
                         std::vector<std::string>{"missing"}),
               std::invalid_argument);
}

TEST(MSDataset, AttributesAndTags) {
  MSDataset dataset = sample_dataset();
  dataset.set_attribute("k", "v");
  dataset.set_attribute("source", "changed");
  EXPECT_EQ(dataset.attributes(),
            (Attributes{{"source", "changed"}, {"k", "v"}}));  // insertion order kept
  EXPECT_TRUE(dataset.remove_attribute("k"));
  EXPECT_FALSE(dataset.remove_attribute("k"));
  dataset.clear_attributes();
  EXPECT_TRUE(dataset.attributes().empty());

  EXPECT_TRUE(dataset.add_tag("tag2"));
  EXPECT_FALSE(dataset.add_tag("tag2"));
  EXPECT_TRUE(dataset.has_tag("tag2"));
  EXPECT_TRUE(dataset.remove_tag("tag1"));
  dataset.clear_tags();
  EXPECT_TRUE(dataset.tags().empty());
}

TEST(MSDataset, RecordsAndColumns) {
  MSDataset dataset = sample_dataset();
  SpectrumRecord record = dataset[1];
  EXPECT_EQ(std::get<std::string>(record["name"]), "spec2");
  EXPECT_EQ(std::get<double>(record["precursor_mz"]), 250.0);
  EXPECT_EQ(record.n_peaks(), 2);
  EXPECT_EQ(record.peaks().mz(0), 200.0);
  EXPECT_TRUE(record.contains("SMILES"));
  EXPECT_THROW(record["missing"], KeyError);
  EXPECT_THROW(dataset[3], std::out_of_range);
  EXPECT_EQ(names(dataset), (std::vector<std::string>{"spec1", "spec2", "spec3"}));
}

TEST(MSDataset, ViewsSliceSelectFilter) {
  MSDataset dataset = sample_dataset();
  EXPECT_EQ(names(dataset.slice(1, 3)), (std::vector<std::string>{"spec2", "spec3"}));
  EXPECT_EQ(names(dataset.select({2, 0})), (std::vector<std::string>{"spec3", "spec1"}));
  EXPECT_EQ(names(dataset.filter({true, false, true})), (std::vector<std::string>{"spec1", "spec3"}));
  EXPECT_THROW(dataset.filter({true}), std::invalid_argument);
  MSDataset view = dataset.select({2});
  EXPECT_EQ(view.description(), "test dataset");
  EXPECT_EQ(view.n_peaks_total(), 1);
}

TEST(MSDataset, SetColumn) {
  MSDataset dataset = sample_dataset();
  dataset.set_column("score", std::vector<Value>{0.1, 0.2, 0.3});
  EXPECT_EQ(dataset.columns().back(), "score");
  EXPECT_EQ(std::get<double>(dataset.value(2, "score")), 0.3);
  dataset.set_column("flag", Value(std::string("ok")));
  EXPECT_EQ(as_strings(dataset.column_values("flag")), (std::vector<std::string>{"ok", "ok", "ok"}));

  // Setting through a view leaves the other rows null.
  MSDataset view = dataset.select({1});
  view.set_column("only_second", Value(int64_t{7}));
  MSDataset full = view.full_view(true);
  auto values = full.column_values("only_second");
  EXPECT_TRUE(is_null(values[0]));
  EXPECT_EQ(std::get<int64_t>(values[1]), 7);
  EXPECT_THROW(dataset.set_column("bad", std::vector<Value>{1.0}), std::invalid_argument);

  dataset[0].set("name", std::string("renamed"));
  EXPECT_EQ(std::get<std::string>(dataset.value(0, "name")), "renamed");
}

TEST(MSDataset, CopyIsIndependent) {
  MSDataset dataset = sample_dataset();
  MSDataset copy = dataset.select({1, 2}).copy();
  copy.set_column("name", strings({"x", "y"}));
  EXPECT_EQ(names(copy), (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(names(dataset), (std::vector<std::string>{"spec1", "spec2", "spec3"}));
  EXPECT_EQ(copy.peaks().index(), (std::vector<int64_t>{0, 1}));
}

TEST(MSDataset, ResetView) {
  MSDataset dataset = sample_dataset();
  MSDataset view = dataset.select({0});
  view.set_columns({"name"});
  MSDataset full = view.full_view();
  EXPECT_EQ(full.size(), 3);
  EXPECT_EQ(full.columns(), (std::vector<std::string>{"name"}));
  EXPECT_EQ(view.full_view(true).columns().size(), 3u);
  view.reset_view(true);
  EXPECT_EQ(view.size(), 3);
  EXPECT_EQ(view.n_columns(), 3);
}

TEST(MSDataset, SortBy) {
  MSDataset dataset = sample_dataset();
  EXPECT_EQ(names(dataset.sort_by("precursor_mz", false)),
            (std::vector<std::string>{"spec3", "spec2", "spec1"}));
  EXPECT_EQ(names(dataset.sort_by("SMILES")), (std::vector<std::string>{"spec2", "spec3", "spec1"}));
  dataset.set_column("partial", std::vector<Value>{std::monostate{}, 2.0, 1.0});
  EXPECT_EQ(names(dataset.sort_by("partial")), (std::vector<std::string>{"spec3", "spec2", "spec1"}));
  EXPECT_EQ(names(dataset.sort_by("partial", false)),
            (std::vector<std::string>{"spec2", "spec3", "spec1"}));
}

TEST(MSDataset, Concat) {
  MSDataset dataset = sample_dataset();
  MSDataset merged =
      MSDataset::concat({dataset.slice(0, 2).copy(), dataset.slice(2, 3).copy()}, "merged");
  EXPECT_EQ(merged.size(), 3);
  EXPECT_EQ(merged.description(), "merged");
  EXPECT_EQ(names(merged), (std::vector<std::string>{"spec1", "spec2", "spec3"}));
  EXPECT_EQ(merged.n_peaks_total(), 6);
  EXPECT_EQ(merged.peaks().metadata()->num_rows(), 6);
}

TEST(MSDataset, ConcatUnifiesColumns) {
  MSDataset a(make_table({{"x", testing::ints({1})}}), PeakSeries::from_spectra({{{1, 1}}}));
  MSDataset b(make_table({{"x", strings({"two"})}, {"y", strings({"only b"})}}),
              PeakSeries({2, 2}, {0, 1}, make_table({{"note", strings({"n"})}})));
  MSDataset merged = MSDataset::concat({a, b});
  EXPECT_EQ(merged.columns(), (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(as_strings(merged.column_values("x")), (std::vector<std::string>{"1", "two"}));
  EXPECT_TRUE(is_null(merged.value(0, "y")));
  // Peak metadata missing in one input becomes null.
  auto notes = internal::to_values(*merged.peaks().metadata()->column(0));
  EXPECT_TRUE(is_null(notes[0]));
  EXPECT_EQ(std::get<std::string>(notes[1]), "n");
}

TEST(SpecId, AssignsZeroPaddedIds) {
  MSDataset dataset(nullptr, PeakSeries::from_spectra(std::vector<std::vector<std::pair<double, double>>>(12)));
  EXPECT_TRUE(processing::set_spec_id(dataset, "SpecID", "SP"));
  auto ids = as_strings(dataset.column_values("SpecID"));
  EXPECT_EQ(ids.front(), "SP01");
  EXPECT_EQ(ids.back(), "SP12");
  EXPECT_FALSE(processing::set_spec_id(dataset, "SpecID", "X"));
  EXPECT_TRUE(processing::set_spec_id(dataset, "SpecID", "X", true, 0));
  EXPECT_EQ(as_strings(dataset.column_values("SpecID")).front(), "X00");
}

}  // namespace
}  // namespace msentity
