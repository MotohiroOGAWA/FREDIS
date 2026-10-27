#include <gtest/gtest.h>

#include "internal/hdf5.hpp"
#include "test_helpers.hpp"

namespace msentity::io {
namespace {

using msentity::testing::as_strings;
using msentity::testing::make_table;
using msentity::testing::sample_dataset;
using msentity::testing::strings;
using msentity::testing::TempDir;

void expect_same(const MSDataset& a, const MSDataset& b) {
  EXPECT_EQ(a.size(), b.size());
  EXPECT_EQ(a.columns(), b.columns());
  EXPECT_EQ(a.description(), b.description());
  EXPECT_EQ(a.attributes(), b.attributes());
  EXPECT_EQ(a.tags(), b.tags());
  EXPECT_TRUE(a.metadata()->Equals(*b.metadata()));
  EXPECT_EQ(a.peaks().offsets(), b.peaks().offsets());
  EXPECT_EQ(a.peaks().data(), b.peaks().data());
  EXPECT_EQ(a.peaks().metadata_columns(), b.peaks().metadata_columns());
  if (a.peaks().has_metadata()) EXPECT_TRUE(a.peaks().metadata()->Equals(*b.peaks().metadata()));
}

TEST(Msds, ReadsReferenceSample) {
  MSDataset dataset = read_msds(msentity::testing::data_path("sample.msds"));
  EXPECT_EQ(dataset.size(), 6);
  EXPECT_EQ(dataset.columns().front(), "IDENTIFIER");
  EXPECT_EQ(dataset.peaks().offsets()[1], 8);
  EXPECT_DOUBLE_EQ(dataset.peaks().mz().front(), 91.0542);
  // Row labels from the schema metadata are not exposed as a column.
  EXPECT_EQ(internal::column_index(*dataset.spectrum_metadata_ref(), "__index_level_0__"), -1);
}

TEST(Msds, RoundTripWithTypesAndPeakMetadata) {
  TempDir tmp;
  MSDataset dataset = sample_dataset();
  dataset.set_description("説明 with UTF-8");
  dataset.set_attributes({{"source", "unit"}, {"キー", "値"}});
  dataset.set_tags({"a", "タグ"});
  dataset.set_column("flag", std::vector<Value>{true, false, std::monostate{}});
  dataset.set_column("count", std::vector<Value>{int64_t{1}, int64_t{2}, int64_t{3}});
  write_msds(dataset, tmp.file("a.msds"));
  MSDataset loaded = read_msds(tmp.file("a.msds"));
  expect_same(dataset, loaded);

  MSDatasetMeta meta = read_msds_meta(tmp.file("a.msds"));
  EXPECT_EQ(meta.description, "説明 with UTF-8");
  EXPECT_EQ(meta.attributes, dataset.attributes());
  EXPECT_EQ(meta.tags, dataset.tags());

  MSDataset without_peak_metadata = read_msds(tmp.file("a.msds"), false);
  EXPECT_FALSE(without_peak_metadata.peaks().has_metadata());
}

TEST(Msds, SaveViewVersusRawStorage) {
  TempDir tmp;
  MSDataset view = sample_dataset().select({2, 0});
  view.set_columns({"SMILES", "name"});

  write_msds(view, tmp.file("view.msds"));
  MSDataset saved_view = read_msds(tmp.file("view.msds"));
  expect_same(view, saved_view);
  EXPECT_EQ(saved_view.peaks().index(), (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(saved_view.spectrum_metadata_ref()->num_columns(), 2);

  MsdsWriteOptions raw;
  raw.save_view = false;
  write_msds(view, tmp.file("raw.msds"), raw);
  MSDataset saved_raw = read_msds(tmp.file("raw.msds"));
  expect_same(view, saved_raw);
  EXPECT_EQ(saved_raw.peaks().index(), (std::vector<int64_t>{2, 0}));
  EXPECT_EQ(saved_raw.spectrum_metadata_ref()->num_columns(), 3);
}

TEST(Msds, AppendConcatenatesGroups) {
  TempDir tmp;
  MSDataset dataset = sample_dataset();
  MsdsWriteOptions append;
  append.append = true;
  write_msds(dataset.slice(0, 1), tmp.file("a.msds"), append);  // creates the file
  write_msds(dataset.slice(1, 3), tmp.file("a.msds"), append);
  MSDataset loaded = read_msds(tmp.file("a.msds"));
  EXPECT_EQ(loaded.size(), 3);
  EXPECT_EQ(as_strings(loaded.column_values("name")),
            (std::vector<std::string>{"spec1", "spec2", "spec3"}));
  EXPECT_EQ(loaded.description(), "test dataset");

  write_msds(dataset.slice(0, 1), tmp.file("a.msds"));  // overwrite
  EXPECT_EQ(read_msds(tmp.file("a.msds")).size(), 1);
}

TEST(Msds, SplitsLargeTablesIntoParts) {
  TempDir tmp;
  MsdsWriteOptions options;
  options.max_part_bytes = 1500;
  std::vector<Value> names;
  for (int i = 0; i < 200; ++i) names.emplace_back("spectrum name number " + std::to_string(i));
  std::vector<std::vector<std::pair<double, double>>> spectra(200, {{1.0, 2.0}});
  MSDataset dataset(make_table({{"name", names}}), PeakSeries::from_spectra(spectra));
  write_msds(dataset, tmp.file("parts.msds"), options);

  auto file = internal::h5::open_file(tmp.file("parts.msds"), internal::h5::OpenMode::ReadOnly);
  auto group = internal::h5::open_group(file.id(), "dataset_0");
  EXPECT_TRUE(*internal::h5::read_bool_attribute(group.id(), "spectrum_metadata_parquet__chunked"));
  EXPECT_GT(*internal::h5::read_int_attribute(group.id(), "spectrum_metadata_parquet__num_parts"), 1);
  expect_same(dataset, read_msds(tmp.file("parts.msds")));
}

TEST(Msds, ZeroColumnMetadataKeepsRowCount) {
  TempDir tmp;
  MSDataset dataset = read_mgf_text("BEGIN IONS\n1 2\n3 4\nEND IONS\nBEGIN IONS\n5 6\nEND IONS\n");
  ASSERT_EQ(dataset.size(), 2);
  EXPECT_TRUE(dataset.columns().empty());
  write_msds(dataset, tmp.file("nometa.msds"));
  MSDataset loaded = read_msds(tmp.file("nometa.msds"));
  EXPECT_EQ(loaded.size(), 2);
  EXPECT_EQ(loaded.peaks().offsets(), (std::vector<int64_t>{0, 2, 3}));
}

TEST(Msds, BytesRoundTrip) {
  MSDataset dataset = sample_dataset();
  auto bytes = msds_to_bytes(dataset);
  ASSERT_GT(bytes.size(), 8u);
  EXPECT_EQ(std::string(bytes.begin() + 1, bytes.begin() + 4), "HDF");
  expect_same(dataset, msds_from_bytes(bytes));
}

TEST(Msds, Errors) {
  TempDir tmp;
  EXPECT_THROW(read_msds(tmp.file("missing.msds")), FormatError);
  std::string not_hdf5 = tmp.write("x.msds", "not an hdf5 file");
  EXPECT_THROW(read_msds(not_hdf5), FormatError);
}

TEST(Hdf5, GuessChunkLength) {
  EXPECT_EQ(internal::h5::guess_chunk_1d(210250, 1), 13141u);
  EXPECT_EQ(internal::h5::guess_chunk_1d(6764, 1), 6764u);
  EXPECT_EQ(internal::h5::guess_chunk_1d(28647517, 1), 55953u);
}

}  // namespace
}  // namespace msentity::io
