// Ports of tests/TestCore/TestPeakSeries.py.

#include <gtest/gtest.h>

#include "test_helpers.hpp"

namespace msentity {
namespace {

using testing::make_table;
using testing::sample_peak_series;
using testing::strings;

TEST(PeakSeries, ShapeAndLayout) {
  PeakSeries series = sample_peak_series();
  EXPECT_EQ(series.size(), 3);
  EXPECT_EQ(series.n_peaks_total(), 6);
  EXPECT_EQ(series.lengths(), (std::vector<int64_t>{3, 2, 1}));
  EXPECT_EQ(series.offsets(), (std::vector<int64_t>{0, 3, 5, 6}));
  EXPECT_EQ(series.peak_indices(), (std::vector<int64_t>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(series.mz(), (std::vector<double>{100, 101, 102, 200, 201, 300}));
  EXPECT_EQ(series.intensity(), (std::vector<double>{10, 30, 20, 5, 15, 7}));
  EXPECT_EQ(series.n_peaks(1), 2);
  EXPECT_EQ(series.metadata_columns(), (std::vector<std::string>{"annotation", "group"}));
}

TEST(PeakSeries, ValidatesInput) {
  EXPECT_THROW(PeakSeries({1.0, 2.0, 3.0}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(PeakSeries({1.0, 2.0}, {0, 2}), std::invalid_argument);
  EXPECT_THROW(PeakSeries({1.0, 2.0, 3.0, 4.0}, {0, 2, 1}), std::invalid_argument);
  EXPECT_THROW(PeakSeries({1.0, 2.0}, {0, 1}, make_table({{"a", strings({"x", "y"})}})),
               std::invalid_argument);
  EXPECT_THROW(PeakSeries({1.0, 2.0}, {0, 1}, nullptr, std::nullopt, std::vector<int64_t>{1}),
               std::out_of_range);
}

TEST(PeakSeries, SpectrumAccess) {
  PeakSeries series = sample_peak_series();
  Spectrum spectrum = series[1];
  EXPECT_EQ(spectrum.size(), 2);
  Peak peak = spectrum[1];
  EXPECT_EQ(peak.mz, 201.0);
  EXPECT_EQ(peak.intensity, 15.0);
  EXPECT_EQ(std::get<std::string>(peak.metadata.at("annotation")), "e");
  EXPECT_EQ(std::get<int64_t>(peak.metadata.at("group")), 2);
  EXPECT_THROW(series[3], std::out_of_range);
  EXPECT_THROW(spectrum.mz(2), std::out_of_range);
  auto metadata = spectrum.metadata();
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->num_rows(), 2);
}

TEST(PeakSeries, ViewsShareStorage) {
  PeakSeries series = sample_peak_series();
  PeakSeries sliced = series.slice(1, 3);
  EXPECT_EQ(sliced.size(), 2);
  EXPECT_EQ(sliced.mz(), (std::vector<double>{200, 201, 300}));
  PeakSeries picked = series.select({2, 0});
  EXPECT_EQ(picked.offsets(), (std::vector<int64_t>{0, 1, 4}));
  EXPECT_EQ(picked.mz(), (std::vector<double>{300, 100, 101, 102}));
  EXPECT_EQ(series.select({-1}).mz(), (std::vector<double>{300}));

  auto metadata = picked.metadata();
  EXPECT_EQ(testing::as_strings(internal::to_values(*metadata->column(0))),
            (std::vector<std::string>{"f", "a", "b", "c"}));

  picked.normalize();  // in place on shared storage
  EXPECT_EQ(series.intensity(), (std::vector<double>{10.0 / 30, 1.0, 20.0 / 30, 5, 15, 1.0}));
}

TEST(PeakSeries, CopyIsIndependent) {
  PeakSeries series = sample_peak_series();
  PeakSeries copy = series.select({1}).copy();
  EXPECT_EQ(copy.size(), 1);
  EXPECT_EQ(copy.offsets(), (std::vector<int64_t>{0, 2}));
  copy.normalize(100.0);
  EXPECT_EQ(copy.intensity(), (std::vector<double>{5.0 / 15 * 100, 100.0}));
  EXPECT_EQ(series.intensity()[3], 5.0);
  EXPECT_EQ(copy.metadata()->num_rows(), 2);
}

TEST(PeakSeries, ResetView) {
  PeakSeries series = sample_peak_series().select({1});
  PeakSeries full = series.full_view();
  EXPECT_EQ(series.size(), 1);
  EXPECT_EQ(full.size(), 3);
  series.reset_view();
  EXPECT_EQ(series.size(), 3);
}

TEST(PeakSeries, SetMetadataColumn) {
  PeakSeries series = sample_peak_series();
  PeakSeries view = series.select({1});
  view.set_metadata_column("score", std::vector<Value>{0.5, 0.25});
  EXPECT_EQ(view.metadata_columns().back(), "score");
  // Other rows of the shared table are null.
  auto all = series.full_view();
  all.set_metadata_columns({"score"});
  auto values = internal::to_values(*all.metadata()->column(0));
  EXPECT_TRUE(is_null(values[0]));
  EXPECT_EQ(std::get<double>(values[3]), 0.5);
  EXPECT_THROW(view.set_metadata_column("bad", std::vector<Value>{1.0}), std::invalid_argument);

  PeakSeries plain({1, 2, 3, 4}, {0, 2});
  plain.set_metadata_column("label", Value(std::string("x")));
  EXPECT_EQ(plain.metadata()->num_rows(), 2);
}

TEST(PeakSeries, NormalizeSkipsEmptyAndNonPositiveSpectra) {
  PeakSeries series({1, 0, 2, 0, 3, 4}, {0, 2, 2, 3});
  series.normalize();
  EXPECT_EQ(series.intensity(), (std::vector<double>{0, 0, 1.0}));
  PeakSeries normalized = sample_peak_series().normalized(2.0);
  EXPECT_DOUBLE_EQ(normalized.intensity()[1], 2.0);
}

TEST(PeakSeries, SortByIntensityDescending) {
  PeakSeries sorted = sample_peak_series().sorted_by_intensity(false);
  EXPECT_EQ(sorted.data(), (std::vector<double>{101, 30, 102, 20, 100, 10, 201, 15, 200, 5, 300, 7}));
  EXPECT_EQ(testing::as_strings(internal::to_values(*sorted.metadata()->column(0))),
            (std::vector<std::string>{"b", "c", "a", "e", "d", "f"}));
}

TEST(PeakSeries, SortByMzReturnsPermutation) {
  PeakSeries series({202, 20, 200, 10, 201, 30, 101, 15, 100, 5, 300, 7}, {0, 3, 5, 6},
                    make_table({{"annotation", strings({"a", "b", "c", "d", "e", "f"})}}));
  std::vector<int64_t> permutation;
  series.sort_by_mz(true, &permutation);
  EXPECT_EQ(permutation, (std::vector<int64_t>{1, 2, 0, 4, 3, 5}));
  EXPECT_EQ(series.mz(), (std::vector<double>{200, 201, 202, 100, 101, 300}));
  EXPECT_EQ(testing::as_strings(internal::to_values(*series.metadata()->column(0))),
            (std::vector<std::string>{"b", "c", "a", "e", "d", "f"}));
}

TEST(PeakSeries, DescendingSortReversesStableOrderOfTies) {
  // Descending = stable ascending order reversed.
  PeakSeries series({1, 5, 2, 5, 3, 1}, {0, 3});
  std::vector<int64_t> permutation;
  series.sort_by_intensity(false, &permutation);
  EXPECT_EQ(permutation, (std::vector<int64_t>{1, 0, 2}));
}

TEST(PeakSeries, Reorder) {
  PeakSeries series = sample_peak_series();
  EXPECT_EQ(series.reorder({2, 0, 1}).index(), (std::vector<int64_t>{2, 0, 1}));
  EXPECT_THROW(series.reorder({0, 0, 1}), std::invalid_argument);
}

TEST(PeakSeries, ReplaceDataForVisibleView) {
  std::vector<double> data;
  for (int i = 0; i < 12; ++i) {
    data.push_back(100.0 + i);
    data.push_back(10.0 + i);
  }
  PeakSeries series(data, {0, 4, 7, 10, 12});
  PeakSeries view = series.select({0, 2});
  view.replace_data({1, 1, 2, 2, 3, 3}, {0, 1, 3},
                    make_table({{"note", strings({"x", "y", "z"})}}));
  EXPECT_EQ(view.offsets(), (std::vector<int64_t>{0, 1, 3}));
  EXPECT_EQ(view.full_view().lengths(), (std::vector<int64_t>{1, 0, 2, 0}));
  EXPECT_EQ(view.metadata_columns(), (std::vector<std::string>{"note"}));
  EXPECT_THROW(view.replace_data({1, 1}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(view.replace_data({1, 1}, {0, 1, 2}), std::invalid_argument);
}

TEST(PeakSeries, FromSpectra) {
  PeakSeries series = PeakSeries::from_spectra({{{1, 2}, {3, 4}}, {}, {{5, 6}}});
  EXPECT_EQ(series.offsets(), (std::vector<int64_t>{0, 2, 2, 3}));
  EXPECT_EQ(series.mz(), (std::vector<double>{1, 3, 5}));
}

TEST(Peak, NeutralLoss) {
  Peak peak(100.0, 1.0, {{"PrecursorMZ", 150.5}});
  EXPECT_DOUBLE_EQ(peak.neutral_loss(), 50.5);
  EXPECT_THROW(Peak(1, 1).neutral_loss(), KeyError);
}

}  // namespace
}  // namespace msentity
