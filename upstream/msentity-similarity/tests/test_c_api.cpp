#include <gtest/gtest.h>

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <nlohmann/json.hpp>

#include <cstring>
#include <string>
#include <vector>

#include "msentity/similarity/c/msentity_similarity_c.h"
#include "test_helpers.hpp"

namespace {

using mssim_test::msentity_data;
using mssim_test::own_data;
using mssim_test::TempDir;

std::vector<uint8_t> sample_bytes() {
  return msentity::io::msds_to_bytes(msentity::io::read_msp(msentity_data("sample.msp")));
}

struct Result {
  mssim_result* handle = nullptr;
  ~Result() { mssim_free(handle); }
};

std::vector<int64_t> int_column(const mssim_result* result, const char* column) {
  int64_t n = mssim_size(result);
  std::vector<msentity_value> values(static_cast<std::size_t>(n));
  EXPECT_EQ(mssim_get_column(result, column, values.data(), n), MSENTITY_OK) << mssim_last_error();
  std::vector<int64_t> out;
  for (const auto& v : values) out.push_back(v.int_value);
  mssim_values_clear(values.data(), n);
  return out;
}

TEST(CApi, AllPairsAndPairScores) {
  auto bytes = sample_bytes();
  mssim_options options;
  mssim_default_options(&options);
  EXPECT_EQ(options.method, MSSIM_METHOD_COSINE);
  EXPECT_DOUBLE_EQ(options.bin_width, 0.01);

  std::vector<float> matrix(25);
  ASSERT_EQ(mssim_all_pairs(bytes.data(), bytes.size(), bytes.data(), bytes.size(), &options,
                            matrix.data(), 25),
            MSENTITY_OK)
      << mssim_last_error();
  EXPECT_NEAR(matrix[1], 0.7569831609725952f, 1e-6);
  EXPECT_EQ(mssim_all_pairs(bytes.data(), bytes.size(), bytes.data(), bytes.size(), nullptr,
                            matrix.data(), 24),
            MSENTITY_ERR_INVALID_ARGUMENT);

  int64_t index1[] = {0, 2}, index2[] = {1, 4};
  float scores[2];
  options.method = MSSIM_METHOD_REVERSE_COSINE;
  ASSERT_EQ(mssim_pair_scores(bytes.data(), bytes.size(), index1, bytes.data(), bytes.size(),
                              index2, 2, &options, scores),
            MSENTITY_OK);
  EXPECT_NEAR(scores[0], 0.7586880922317505f, 1e-6);
  EXPECT_NEAR(scores[1], 0.31010058522224426f, 1e-6);

  options.method = 7;
  EXPECT_EQ(mssim_pair_scores(bytes.data(), bytes.size(), index1, bytes.data(), bytes.size(),
                              index2, 2, &options, scores),
            MSENTITY_ERR_INVALID_ARGUMENT);
  index1[0] = 9;
  EXPECT_EQ(mssim_pair_scores(bytes.data(), bytes.size(), index1, bytes.data(), bytes.size(),
                              index2, 2, nullptr, scores),
            MSENTITY_ERR_OUT_OF_RANGE);
  uint8_t junk[] = {1, 2, 3};
  EXPECT_EQ(mssim_all_pairs(junk, 3, bytes.data(), bytes.size(), nullptr, matrix.data(), 25),
            MSENTITY_ERR_FORMAT);
  EXPECT_NE(std::strlen(mssim_last_error()), 0u);
}

void count_progress(int64_t done, int64_t total, void* user_data) {
  auto* last = static_cast<int64_t*>(user_data);
  last[0] = done;
  last[1] = total;
}

TEST(CApi, LibrarySearchResult) {
  TempDir tmp;
  auto bytes = sample_bytes();
  int64_t progress[2] = {-1, -1};
  Result result;
  ASSERT_EQ(mssim_library_search(bytes.data(), bytes.size(), bytes.data(), bytes.size(), 0.9,
                                 nullptr, 1, "q.msp", nullptr, count_progress, progress,
                                 &result.handle),
            MSENTITY_OK)
      << mssim_last_error();
  EXPECT_EQ(progress[0], 25);
  EXPECT_EQ(progress[1], 25);
  EXPECT_EQ(mssim_size(result.handle), 9);
  ASSERT_EQ(mssim_n_columns(result.handle), 5);
  char* name = nullptr;
  ASSERT_EQ(mssim_column_name(result.handle, 2, &name), MSENTITY_OK);
  EXPECT_STREQ(name, "cosine_similarity");
  mssim_string_free(name);
  EXPECT_EQ(mssim_column_name(result.handle, 5, &name), MSENTITY_ERR_OUT_OF_RANGE);
  EXPECT_EQ(int_column(result.handle, "index2"), (std::vector<int64_t>{0, 1, 4, 2, 3, 2, 3, 1, 4}));
  msentity_value cell[9];
  EXPECT_EQ(mssim_get_column(result.handle, "missing", cell, 9), MSENTITY_ERR_KEY);

  char* json = nullptr;
  ASSERT_EQ(mssim_get_metadata_json(result.handle, &json), MSENTITY_OK);
  auto metadata = nlohmann::json::parse(json);
  mssim_string_free(json);
  EXPECT_EQ(metadata["mode"], "library_search");
  EXPECT_EQ(metadata["sources"][0]["path"], "q.msp");

  ASSERT_EQ(mssim_has_matched_data(result.handle), 1);
  EXPECT_EQ(mssim_matched_size(result.handle, 1), 5);
  uint8_t* matched = nullptr;
  int64_t matched_size = 0;
  ASSERT_EQ(mssim_matched_dataset(result.handle, 2, &matched, &matched_size), MSENTITY_OK);
  auto reference = msentity::io::msds_from_bytes(std::vector<uint8_t>(matched, matched + matched_size));
  mssim_bytes_free(matched);
  EXPECT_EQ(reference.size(), 5);
  int64_t sources[5];
  ASSERT_EQ(mssim_source_indices(result.handle, 1, sources, 5), MSENTITY_OK);
  EXPECT_EQ(sources[4], 4);
  int64_t d1 = -1, d2 = -1;
  ASSERT_EQ(mssim_match_positions(result.handle, 2, &d1, &d2), MSENTITY_OK);
  EXPECT_EQ(d1, 1);
  // Matched spectra are stored in order of first appearance (index2: 0, 1, 4, ...).
  EXPECT_EQ(d2, 2);
  int64_t reference_sources[5];
  ASSERT_EQ(mssim_source_indices(result.handle, 2, reference_sources, 5), MSENTITY_OK);
  EXPECT_EQ(reference_sources[d2], 4);
  EXPECT_EQ(mssim_match_positions(result.handle, 9, &d1, &d2), MSENTITY_ERR_OUT_OF_RANGE);
  EXPECT_EQ(mssim_matched_dataset(result.handle, 3, &matched, &matched_size),
            MSENTITY_ERR_INVALID_ARGUMENT);

  // Filter (index1 == 1), sort, save and load.
  auto index1 = int_column(result.handle, "index1");
  std::vector<uint8_t> mask;
  for (auto v : index1) mask.push_back(v == 1 ? 1 : 0);
  Result filtered, sorted, loaded;
  ASSERT_EQ(mssim_filter(result.handle, mask.data(), mask.size(), &filtered.handle), MSENTITY_OK);
  const char* columns[] = {"cosine_similarity"};
  int32_t ascending[] = {1};
  ASSERT_EQ(mssim_sort_by(filtered.handle, columns, ascending, 1, &sorted.handle), MSENTITY_OK);
  EXPECT_EQ(int_column(sorted.handle, "index2"), (std::vector<int64_t>{4, 1}));
  std::string path = tmp.file("filtered.mssim");
  ASSERT_EQ(mssim_save(sorted.handle, path.c_str()), MSENTITY_OK) << mssim_last_error();
  ASSERT_EQ(mssim_load(path.c_str(), &loaded.handle), MSENTITY_OK);
  EXPECT_EQ(mssim_size(loaded.handle), 2);
  EXPECT_EQ(mssim_matched_size(loaded.handle, 1), 1);
  EXPECT_EQ(mssim_matched_size(loaded.handle, 2), 2);

  double summary[7];
  ASSERT_EQ(mssim_describe_scores(loaded.handle, summary), MSENTITY_OK);
  EXPECT_EQ(summary[0], 2);
  int64_t counts[4];
  double edges[5];
  ASSERT_EQ(mssim_histogram(loaded.handle, 4, counts, edges), MSENTITY_OK);
  EXPECT_EQ(counts[3], 2);
  EXPECT_DOUBLE_EQ(edges[4], 1.0);

  ArrowArrayStream stream;
  ASSERT_EQ(mssim_table_arrow(loaded.handle, &stream), MSENTITY_OK);
  auto table = arrow::ImportRecordBatchReader(&stream).ValueOrDie()->ToTable().ValueOrDie();
  EXPECT_EQ(table->num_rows(), 2);
}

TEST(CApi, ByKeyLoadAndErrors) {
  TempDir tmp;
  auto ds = msentity::io::read_msp(msentity_data("sample.msp"));
  msentity::processing::set_spec_id(ds, "SpecID", "SP");
  auto bytes = msentity::io::msds_to_bytes(ds);
  Result result;
  ASSERT_EQ(mssim_similarity_by_key(bytes.data(), bytes.size(), bytes.data(), bytes.size(),
                                    nullptr, nullptr, nullptr, 0, nullptr, nullptr, &result.handle),
            MSENTITY_OK)
      << mssim_last_error();
  EXPECT_EQ(mssim_size(result.handle), 5);
  EXPECT_EQ(mssim_has_matched_data(result.handle), 0);
  int64_t d1, d2;
  EXPECT_EQ(mssim_match_positions(result.handle, 0, &d1, &d2), MSENTITY_ERR_INVALID_ARGUMENT);
  ASSERT_EQ(mssim_set_metadata_json(result.handle, R"({"note": "x"})"), MSENTITY_OK);
  EXPECT_EQ(mssim_set_metadata_json(result.handle, "[1]"), MSENTITY_ERR_INVALID_ARGUMENT);
  std::string csv = tmp.file("t.csv");
  EXPECT_EQ(mssim_export_table(result.handle, csv.c_str()), MSENTITY_OK);

  Result reference;
  ASSERT_EQ(mssim_load(own_data("similarity.mssim").c_str(), &reference.handle), MSENTITY_OK);
  EXPECT_EQ(mssim_size(reference.handle), 16);
  Result missing;
  EXPECT_EQ(mssim_load(tmp.file("none.mssim").c_str(), &missing.handle), MSENTITY_ERR_IO);
  EXPECT_EQ(mssim_save(nullptr, csv.c_str()), MSENTITY_ERR_INVALID_ARGUMENT);
  EXPECT_STREQ(mssim_version(), "0.1.0");
}

}  // namespace
