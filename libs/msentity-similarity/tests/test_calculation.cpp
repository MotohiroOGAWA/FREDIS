#include <gtest/gtest.h>

#include <cmath>

#include "internal/table.hpp"
#include "test_helpers.hpp"

namespace msentity::similarity {
namespace {

using mssim_test::msentity_data;

MSDataset spectra(const std::vector<std::vector<std::pair<double, double>>>& peaks) {
  return MSDataset(nullptr, PeakSeries::from_spectra(peaks));
}

MSDataset sample() { return io::read_msp(msentity_data("sample.msp")); }

// ---- definition on hand-made spectra ------------------------------------------

TEST(Score, CosineAndReverseCosine) {
  // a: bins 100, 200; b: bins 100, 300 (unit intensities).
  MSDataset a = spectra({{{100.0, 1.0}, {200.0, 1.0}}});
  MSDataset b = spectra({{{100.0, 1.0}, {300.0, 1.0}}});
  EXPECT_FLOAT_EQ(pair_scores(a, {0}, b, {0})[0], 0.5f);  // 1 / (√2 √2)
  Options reverse;
  reverse.method = Method::ReverseCosine;
  // Only a's matched peak counts in its norm: 1 / (√1 √2).
  EXPECT_FLOAT_EQ(pair_scores(a, {0}, b, {0}, reverse)[0], static_cast<float>(1 / std::sqrt(2.0)));
}

TEST(Score, BinsSumIntensityAndClipNegative) {
  // 100.001 and 100.009 share bin floor(x / 0.01) = 10000; the -5 becomes 0.
  MSDataset a = spectra({{{100.001, 1.0}, {100.009, 1.0}, {150.0, -5.0}}});
  MSDataset b = spectra({{{100.005, 2.0}}});
  EXPECT_FLOAT_EQ(pair_scores(a, {0}, b, {0})[0], 1.0f);
  Options coarse;
  coarse.bin_width = 1000.0;  // everything in one bin
  EXPECT_FLOAT_EQ(pair_scores(a, {0}, b, {0}, coarse)[0], 1.0f);
}

TEST(Score, IntensityExponentAndEmptySpectra) {
  MSDataset a = spectra({{{100.0, 4.0}, {200.0, 1.0}}, {}});
  MSDataset b = spectra({{{100.0, 1.0}}});
  Options sqrt_intensity;
  sqrt_intensity.intensity_exponent = 0.5;  // 4 → 2, 1 → 1: 2 / √5
  EXPECT_FLOAT_EQ(pair_scores(a, {0}, b, {0}, sqrt_intensity)[0],
                  static_cast<float>(2 / std::sqrt(5.0)));
  EXPECT_EQ(pair_scores(a, {1}, b, {0})[0], 0.0f);  // empty spectrum
}

TEST(Score, Validation) {
  MSDataset a = spectra({{{100.0, 1.0}}});
  Options bad;
  bad.bin_width = 0;
  EXPECT_THROW(pair_scores(a, {0}, a, {0}, bad), std::invalid_argument);
  bad = {};
  bad.intensity_exponent = -1;
  EXPECT_THROW(pair_scores(a, {0}, a, {0}, bad), std::invalid_argument);
  EXPECT_THROW(pair_scores(a, {0, 0}, a, {0}), std::invalid_argument);
  EXPECT_THROW(pair_scores(a, {1}, a, {0}), std::out_of_range);
  EXPECT_THROW(parse_method("dot"), std::invalid_argument);
  EXPECT_EQ(parse_method("reverse_cosine"), Method::ReverseCosine);
  EXPECT_EQ(method_name(Method::Cosine), "cosine");
}

// ---- reference values for msentity/tests/data/sample.msp -----------------------

TEST(Reference, AllPairsMatrixOfSampleMsp) {
  const float expected[5][5] = {
      {1.0f, 0.7569831609725952f, 0.85833740234375f, 0.6929314732551575f, 0.6837769150733948f},
      {0.7569831609725952f, 1.0f, 0.3559565544128418f, 0.22130076587200165f, 0.9926462769508362f},
      {0.85833740234375f, 0.3559565544128418f, 1.0f, 0.9309672117233276f, 0.25937944650650024f},
      {0.6929314732551575f, 0.22130076587200165f, 0.9309672117233276f, 1.0f, 0.1344955712556839f},
      {0.6837769150733948f, 0.9926462769508362f, 0.25937944650650024f, 0.1344955712556839f, 1.0f},
  };
  MSDataset ds = sample();
  auto matrix = all_pairs_matrix(ds, ds);
  ASSERT_EQ(matrix.size(), 25u);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) EXPECT_NEAR(matrix[i * 5 + j], expected[i][j], 1e-6) << i << "," << j;
  }
}

TEST(Reference, MethodsAndBinning) {
  struct Case {
    Method method;
    double bin_width, exponent;
    float s01, s10, s24, s42;
  };
  const Case cases[] = {
      {Method::Cosine, 0.5, 0.5, 0.8930210471153259f, 0.8930210471153259f, 0.5522544980049133f,
       0.5522544980049133f},
      {Method::ReverseCosine, 0.01, 1.0, 0.7586880922317505f, 0.7569831609725952f,
       0.31010058522224426f, 0.25987136363983154f},
      {Method::ReverseCosine, 0.5, 0.5, 0.9125574827194214f, 0.8930210471153259f,
       0.691342294216156f, 0.5648825168609619f},
  };
  MSDataset ds = sample();
  for (const auto& c : cases) {
    Options options;
    options.method = c.method;
    options.bin_width = c.bin_width;
    options.intensity_exponent = c.exponent;
    auto scores = pair_scores(ds, {0, 1, 2, 4}, ds, {1, 0, 4, 2}, options);
    EXPECT_NEAR(scores[0], c.s01, 1e-6);
    EXPECT_NEAR(scores[1], c.s10, 1e-6);
    EXPECT_NEAR(scores[2], c.s24, 1e-6);
    EXPECT_NEAR(scores[3], c.s42, 1e-6);
  }
}

TEST(Reference, LibrarySearchThreshold) {
  MSDataset ds = sample();
  auto table = library_search(ds, ds, 0.9);
  auto index1 = internal::to_values(*table->column(0));
  auto index2 = internal::to_values(*table->column(1));
  std::vector<std::pair<int64_t, int64_t>> pairs;
  for (std::size_t i = 0; i < index1.size(); ++i) {
    pairs.emplace_back(std::get<int64_t>(index1[i]), std::get<int64_t>(index2[i]));
  }
  EXPECT_EQ(pairs, (std::vector<std::pair<int64_t, int64_t>>{
                       {0, 0}, {1, 1}, {1, 4}, {2, 2}, {2, 3}, {3, 2}, {3, 3}, {4, 1}, {4, 4}}));
  EXPECT_EQ(table->schema()->field(2)->type()->id(), arrow::Type::FLOAT);
}

// ---- library search and key matching --------------------------------------------

TEST(LibrarySearch, ThreadsDoNotChangeResultsAndProgressIsReported) {
  // 60 × 60 pairs is enough to use several threads.
  std::vector<std::vector<std::pair<double, double>>> peaks;
  for (int s = 0; s < 60; ++s) {
    peaks.push_back({{100.0 + s % 7, 1.0}, {150.0 + s % 5, 0.5}, {200.0 + s % 3, 0.25}});
  }
  MSDataset ds = spectra(peaks);
  Options one, many;
  one.threads = 1;
  many.threads = 4;
  int64_t last_done = -1, last_total = -1;
  auto a = library_search(ds, ds, 0.5, one);
  auto b = library_search(ds, ds, 0.5, many, [&](int64_t done, int64_t total) {
    last_done = done;
    last_total = total;
  });
  EXPECT_TRUE(a->Equals(*b));
  EXPECT_EQ(last_done, 3600);
  EXPECT_EQ(last_total, 3600);
  EXPECT_THROW(library_search(ds, ds, 1.5), std::invalid_argument);
}

TEST(ByKey, PairsEqualKeysInFirstDatasetOrder) {
  MSDataset ds1 = spectra({{{100, 1}}, {{200, 1}}, {{300, 1}}, {{400, 1}}});
  ds1.set_column("ID", std::vector<Value>{std::string("a"), std::string("b"), std::monostate{},
                                          std::string("d")});
  MSDataset ds2 = spectra({{{400, 1}}, {{100, 1}}, {{999, 1}}});
  ds2.set_column("Key", std::vector<Value>{std::string("d"), std::string("a"), std::string("x")});
  auto table = similarity_by_key(ds1, ds2, "ID", "Key");
  EXPECT_EQ(internal::column_names(*table),
            (std::vector<std::string>{"ID", "Key", "index1", "index2", "cosine_similarity"}));
  EXPECT_EQ(table->num_rows(), 2);
  auto ids = internal::to_values(*table->column(0));
  EXPECT_EQ(std::get<std::string>(ids[0]), "a");
  EXPECT_EQ(std::get<std::string>(ids[1]), "d");
  auto scores = internal::to_values(*table->column(4));
  EXPECT_EQ(std::get<double>(scores[0]), 1.0);

  // Same key name: one key column. Integer and float keys compare by value.
  ds1.set_column("N", std::vector<Value>{int64_t{1}, int64_t{2}, int64_t{3}, int64_t{4}});
  ds2.set_column("N", std::vector<Value>{4.0, 1.0, 7.0});
  auto numeric = similarity_by_key(ds1, ds2, "N", "N");
  EXPECT_EQ(internal::column_names(*numeric),
            (std::vector<std::string>{"N", "index1", "index2", "cosine_similarity"}));
  EXPECT_EQ(numeric->num_rows(), 2);
}

TEST(ByKey, Errors) {
  MSDataset ds = spectra({{{100, 1}}, {{200, 1}}});
  ds.set_column("ID", std::vector<Value>{std::string("a"), std::string("a")});
  EXPECT_THROW(similarity_by_key(ds, ds, "ID", "ID"), std::invalid_argument);  // duplicate
  EXPECT_THROW(similarity_by_key(ds, ds, "Missing", "ID"), KeyError);
  ds.set_column("index1", Value(int64_t{1}));
  EXPECT_THROW(similarity_by_key(ds, ds, "index1", "index1"), std::invalid_argument);  // reserved
}

}  // namespace
}  // namespace msentity::similarity
