#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cmath>

#include "internal/table.hpp"
#include "test_helpers.hpp"

namespace msentity::similarity {
namespace {

using mssim_test::msentity_data;
using mssim_test::own_data;
using mssim_test::read_text;
using mssim_test::TempDir;

std::shared_ptr<arrow::Table> scores_table(const std::vector<int64_t>& index1,
                                           const std::vector<int64_t>& index2,
                                           const std::vector<double>& scores) {
  arrow::Int64Builder b1, b2;
  arrow::DoubleBuilder b3;
  internal::check(b1.AppendValues(index1));
  internal::check(b2.AppendValues(index2));
  internal::check(b3.AppendValues(scores));
  return arrow::Table::Make(arrow::schema({arrow::field("index1", arrow::int64()),
                                           arrow::field("index2", arrow::int64()),
                                           arrow::field("cosine_similarity", arrow::float64())}),
                            {internal::unwrap(b1.Finish()), internal::unwrap(b2.Finish()),
                             internal::unwrap(b3.Finish())},
                            static_cast<int64_t>(scores.size()));
}

TEST(SimilarityDataset, ReadsReferenceFile) {
  // tests/data/similarity.mssim: schema version 2 with embedded matched data.
  auto result = SimilarityDataset::load(own_data("similarity.mssim"));
  EXPECT_EQ(result.size(), 16);
  ASSERT_TRUE(result.has_matched_data());
  EXPECT_EQ(result.matched_datasets()->first.size(), 6);
  EXPECT_EQ(result.matched_source_indices()->first.size(), 6u);
  auto [query, reference] = result.match_records(1);
  EXPECT_EQ(query.size(), 1);
  EXPECT_EQ(reference.size(), 1);
  auto metadata = nlohmann::json::parse(result.metadata_json());
  EXPECT_TRUE(metadata.is_object());
}

TEST(SimilarityDataset, Validation) {
  EXPECT_THROW(SimilarityDataset(nullptr), std::invalid_argument);
  EXPECT_THROW(SimilarityDataset(scores_table({0}, {0}, {1.5})), std::invalid_argument);
  EXPECT_THROW(SimilarityDataset(scores_table({0}, {0}, {std::nan("")})), std::invalid_argument);
  EXPECT_THROW(SimilarityDataset(scores_table({0}, {0}, {0.5}), "[1]"), std::invalid_argument);
  auto table = scores_table({0}, {0}, {0.5});
  EXPECT_THROW(SimilarityDataset(internal::select_columns(*table, {"index1", "cosine_similarity"})),
               std::invalid_argument);
  MSDataset one(nullptr, PeakSeries::from_spectra({{{1, 1}}}));
  // Matched data without data_index columns.
  EXPECT_THROW(SimilarityDataset(table, "{}", SimilarityDataset::DatasetPair{one, one},
                                 SimilarityDataset::IndexPair{{0}, {0}}),
               std::invalid_argument);
  EXPECT_THROW(SimilarityDataset::with_matched_data(scores_table({3}, {0}, {0.5}), "{}", one, one),
               std::invalid_argument);
}

TEST(SimilarityDataset, SaveCompactsMatchedData) {
  TempDir tmp;
  MSDataset ds = io::read_msp(msentity_data("sample.msp"));
  auto result = calculate_library_search(ds, ds, 0.9);  // 9 pairs over 5 spectra
  ASSERT_TRUE(result.has_matched_data());
  EXPECT_EQ(result.matched_datasets()->first.size(), 5);

  // Keep the pairs of spectrum 1 only: (1,1) and (1,4).
  auto index1 = result.column_values("index1");
  std::vector<bool> mask;
  for (const auto& v : index1) mask.push_back(std::get<int64_t>(v) == 1);
  auto filtered = result.filter(mask);
  EXPECT_EQ(filtered.size(), 2);
  auto metadata = nlohmann::json::parse(filtered.metadata_json());
  EXPECT_EQ(metadata["row_count"], 2);
  EXPECT_EQ(metadata["source_row_count"], 9);

  filtered.save(tmp.file("filtered.mssim"));
  auto loaded = SimilarityDataset::load(tmp.file("filtered.mssim"));
  EXPECT_EQ(loaded.size(), 2);
  EXPECT_EQ(loaded.matched_datasets()->first.size(), 1);   // spectrum 1
  EXPECT_EQ(loaded.matched_datasets()->second.size(), 2);  // spectra 1 and 4
  EXPECT_EQ(loaded.matched_source_indices()->first, (std::vector<int64_t>{1}));
  EXPECT_EQ(loaded.matched_source_indices()->second, (std::vector<int64_t>{1, 4}));
  auto [query, reference] = loaded.match_records(1);
  EXPECT_EQ(std::get<std::string>(reference.value(0, "Name")),
            std::get<std::string>(ds.value(4, "Name")));
}

TEST(SimilarityDataset, LightweightRoundTripAndErrors) {
  TempDir tmp;
  SimilarityDataset result(scores_table({0, 1}, {1, 0}, {0.25, 1.0}), R"({"method": "test", "名前": "値"})");
  result.save(tmp.file("dir/light.mssim"));
  auto loaded = SimilarityDataset::load(tmp.file("dir/light.mssim"));
  EXPECT_FALSE(loaded.has_matched_data());
  EXPECT_TRUE(loaded.table()->Equals(*result.table()));
  EXPECT_EQ(loaded.metadata_json(), result.metadata_json());
  EXPECT_THROW(result.save(tmp.file("wrong.ext")), std::invalid_argument);
  EXPECT_THROW(loaded.match_records(0), std::invalid_argument);
  io::write_msds(MSDataset(nullptr, PeakSeries::from_spectra({{{1, 1}}})), tmp.file("plain.mssim"));
  EXPECT_THROW(SimilarityDataset::load(tmp.file("plain.mssim")), FormatError);
}

TEST(SimilarityDataset, SortDescribeHistogram) {
  SimilarityDataset result(scores_table({0, 1, 2, 3}, {3, 2, 1, 0}, {0.2, 1.0, 0.6, 0.6}));
  auto sorted = result.sort_by({"cosine_similarity", "index1"}, {false, true});
  std::vector<int64_t> order;
  for (const auto& v : sorted.column_values("index1")) order.push_back(std::get<int64_t>(v));
  EXPECT_EQ(order, (std::vector<int64_t>{1, 2, 3, 0}));

  ScoreSummary s = result.describe_scores();
  EXPECT_EQ(s.count, 4);
  EXPECT_DOUBLE_EQ(s.mean, 0.6);
  EXPECT_DOUBLE_EQ(s.median, 0.6);
  EXPECT_DOUBLE_EQ(s.q1, 0.5);   // linear: 0.2 + (0.6 - 0.2) * 0.75
  EXPECT_DOUBLE_EQ(s.q3, 0.7);   // 0.6 + (1.0 - 0.6) * 0.25
  EXPECT_DOUBLE_EQ(s.min, 0.2);
  EXPECT_DOUBLE_EQ(s.max, 1.0);

  Histogram h = result.histogram(5);
  EXPECT_EQ(h.counts, (std::vector<int64_t>{0, 1, 0, 2, 1}));  // 1.0 is in the last bin
  EXPECT_EQ(h.edges.size(), 6u);
  EXPECT_THROW(result.histogram(0), std::invalid_argument);
  EXPECT_TRUE(std::isnan(SimilarityDataset(scores_table({}, {}, {})).describe_scores().mean));
}

TEST(SimilarityDataset, ExportTable) {
  TempDir tmp;
  SimilarityDataset result(scores_table({0, 1}, {1, 0}, {0.25, 1.0}));
  result.export_table(tmp.file("t.csv"));
  EXPECT_EQ(read_text(tmp.file("t.csv")), "index1,index2,cosine_similarity\n0,1,0.25\n1,0,1.0\n");
  result.export_table(tmp.file("t.tsv"));
  EXPECT_EQ(read_text(tmp.file("t.tsv")), "index1\tindex2\tcosine_similarity\n0\t1\t0.25\n1\t0\t1.0\n");
  result.export_table(tmp.file("t.parquet"));
  EXPECT_GT(read_text(tmp.file("t.parquet")).size(), 4u);
  EXPECT_THROW(result.export_table(tmp.file("t.xlsx")), std::invalid_argument);
}

TEST(Calculate, MetadataDescribesTheCalculation) {
  MSDataset ds = io::read_msp(msentity_data("sample.msp"));
  processing::set_spec_id(ds, "SpecID", "SP");
  Options options;
  options.method = Method::ReverseCosine;
  auto result = calculate_similarity(ds, ds.select({4, 3, 2}), "SpecID", "SpecID", options, false,
                                     "a.msp", "b.msp");
  EXPECT_EQ(result.size(), 3);
  EXPECT_FALSE(result.has_matched_data());
  auto metadata = nlohmann::json::parse(result.metadata_json());
  EXPECT_EQ(metadata["mode"], "by_key");
  EXPECT_EQ(metadata["algorithm"], "binned_reverse_cosine_similarity_by_key");
  EXPECT_EQ(metadata["parameters"]["key1"], "SpecID");
  EXPECT_EQ(metadata["sources"][1]["path"], "b.msp");
  EXPECT_EQ(metadata["sources"][1]["n_rows"], 3);
  EXPECT_EQ(metadata["row_count"], 3);
  EXPECT_EQ(metadata["matched_data"], "lightweight");
  EXPECT_NE(metadata["created_at"].get<std::string>().find("+00:00"), std::string::npos);

  auto search = calculate_library_search(ds, ds, 0.0, {}, false);
  auto m = nlohmann::json::parse(search.metadata_json());
  EXPECT_EQ(m["candidate_pair_count"], 25);
  EXPECT_EQ(m["parameters"]["threshold"], 0.0);
}

}  // namespace
}  // namespace msentity::similarity
