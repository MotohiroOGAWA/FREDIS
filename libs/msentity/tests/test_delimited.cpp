// Ports of tests/TestIO/TestTsv.py, TestCsv.py, TestMsDataIO.py.

#include <gtest/gtest.h>

#include "test_helpers.hpp"

namespace msentity::io {
namespace {

using msentity::testing::as_strings;
using msentity::testing::read_text;
using msentity::testing::TempDir;

const char* kTypedTsv =
    "Name\tPrecursorMZ\tCharge\tEmpty\tPeak\n"
    "a\t100.5\t1\t\t10,1;20,2\n"
    "b\t\t2\t\t\n";

TEST(Tsv, KeepsTextByDefault) {
  MSDataset dataset = read_tsv_text(kTypedTsv);
  auto metadata = dataset.metadata();
  for (const auto& field : metadata->schema()->fields()) {
    EXPECT_EQ(field->type()->id(), arrow::Type::STRING) << field->name();
  }
  EXPECT_EQ(std::get<std::string>(dataset.value(0, "PrecursorMZ")), "100.5");
  EXPECT_EQ(std::get<std::string>(dataset.value(0, "Charge")), "1");
  EXPECT_EQ(std::get<std::string>(dataset.value(1, "PrecursorMZ")), "");
  EXPECT_EQ(dataset.peaks().data(), (std::vector<double>{10, 1, 20, 2}));
}

TEST(Tsv, InfersTypesWhenRequested) {
  DelimitedReadOptions options;
  options.infer_types = true;
  MSDataset dataset = read_tsv_text(kTypedTsv, options);
  ASSERT_EQ(dataset.size(), 2);
  EXPECT_EQ(dataset.columns(), (std::vector<std::string>{"Name", "PrecursorMZ", "Charge", "Empty"}));
  auto metadata = dataset.metadata();
  EXPECT_EQ(metadata->schema()->field(0)->type()->id(), arrow::Type::STRING);
  EXPECT_EQ(metadata->schema()->field(1)->type()->id(), arrow::Type::DOUBLE);
  EXPECT_EQ(metadata->schema()->field(2)->type()->id(), arrow::Type::INT64);
  // An all-empty column is an all-null float column.
  EXPECT_EQ(metadata->schema()->field(3)->type()->id(), arrow::Type::DOUBLE);
  EXPECT_TRUE(is_null(dataset.value(1, "PrecursorMZ")));
  EXPECT_EQ(dataset.peaks().offsets(), (std::vector<int64_t>{0, 2, 2}));
  EXPECT_EQ(dataset.peaks().data(), (std::vector<double>{10, 1, 20, 2}));
}

TEST(Tsv, RoundTripAndAutoDetection) {
  TempDir tmp;
  MSDataset dataset = read_tsv_text("Name\tScore\tPeak\nx\t0.1\t1,2;3,4.5\ny\t\t\n");
  write_tsv(dataset, tmp.file("out.tsv"));
  // Text columns: the empty cell stays empty.
  EXPECT_EQ(read_text(tmp.file("out.tsv")), "Name\tScore\tPeak\nx\t0.1\t1,2;3,4.5\ny\t\t\n");
  // With inferred types a missing number is written as "nan".
  DelimitedReadOptions infer;
  infer.infer_types = true;
  write_tsv(read_tsv_text("Name\tScore\tPeak\nx\t0.1\t1,2\ny\t\t\n", infer), tmp.file("typed.tsv"));
  EXPECT_EQ(read_text(tmp.file("typed.tsv")), "Name\tScore\tPeak\nx\t0.1\t1,2\ny\tnan\t\n");
  MSDataset reread = load_ms_dataset(tmp.file("out.tsv"));
  EXPECT_EQ(reread.peaks().data(), dataset.peaks().data());
  EXPECT_EQ(as_strings(reread.column_values("Name")), (std::vector<std::string>{"x", "y"}));
}

TEST(Tsv, Validation) {
  EXPECT_THROW(read_tsv_text("Name\nx\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Name\tPeak\nx\t1;2\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Name\tPeak\nx\t1,a\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Name\tName\tPeak\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Name\tPeak\nx\t1,2\textra\n"), std::invalid_argument);
}

TEST(Csv, RoundTripQuotingHeadersAndDetection) {
  TempDir tmp;
  MSDataset dataset = read_csv_text(
      "Name,Note,Peak\n"
      "\"comma, inside\",\"quote \"\"here\"\"\",\"1,2;3,4\"\n");
  EXPECT_EQ(std::get<std::string>(dataset.value(0, "Name")), "comma, inside");
  EXPECT_EQ(std::get<std::string>(dataset.value(0, "Note")), "quote \"here\"");

  DelimitedWriteOptions options;
  options.headers = std::vector<std::string>{"Note"};
  write_csv(dataset, tmp.file("out.csv"), options);
  EXPECT_EQ(read_text(tmp.file("out.csv")), "Note,Peak\n\"quote \"\"here\"\"\",\"1,2;3,4\"\n");
  EXPECT_EQ(load_ms_dataset(tmp.file("out.csv")).size(), 1);

  options.headers = std::vector<std::string>{"Peak"};
  EXPECT_THROW(write_csv(dataset, tmp.file("bad.csv"), options), std::invalid_argument);
  options.headers = std::vector<std::string>{"Unknown"};
  EXPECT_THROW(write_csv(dataset, tmp.file("bad.csv"), options), std::invalid_argument);
}

TEST(Csv, StripsBomAndNormalizesNewlines) {
  TempDir tmp;
  std::string path = tmp.write("bom.csv", "\xEF\xBB\xBFName,Peak\r\nx,\"1,2\"\r\n\r\ny,\r\n");
  MSDataset dataset = read_csv(path);
  EXPECT_EQ(dataset.columns(), (std::vector<std::string>{"Name"}));
  EXPECT_EQ(dataset.size(), 2);
}

TEST(Load, DetectsTypes) {
  EXPECT_EQ(load_ms_dataset(msentity::testing::data_path("sample.msp")).size(), 5);
  EXPECT_EQ(load_ms_dataset(msentity::testing::data_path("sample.mgf")).size(), 6);
  EXPECT_EQ(load_ms_dataset(msentity::testing::data_path("sample.msds")).size(), 6);
  TempDir tmp;
  std::string h5 = tmp.file("copy.h5");
  write_msds(read_msds(msentity::testing::data_path("sample.msds")), h5);
  EXPECT_EQ(load_ms_dataset(h5).size(), 6);
  std::string text = tmp.write("data.txt", "Name: a\nNumPeaks: 1\n1 1\n");
  EXPECT_EQ(load_ms_dataset(text, std::string("msp")).size(), 1);
  EXPECT_THROW(load_ms_dataset(text), std::invalid_argument);
  EXPECT_THROW(load_ms_dataset(text, std::string("xyz")), std::invalid_argument);

  MSDataset with_ids = load_ms_dataset(msentity::testing::data_path("sample.msp"), std::nullopt,
                                       std::string("SP"));
  EXPECT_EQ(std::get<std::string>(with_ids.value(0, "SpecID")), "SP1");
}

}  // namespace
}  // namespace msentity::io

namespace msentity::io {
namespace {

using msentity::testing::as_strings;
using msentity::testing::read_text;
using msentity::testing::TempDir;

std::vector<std::string> peak_column(const MSDataset& dataset, const std::string& name) {
  auto table = dataset.peaks().metadata();
  return as_strings(internal::to_values(*table->column(internal::column_index(*table, name))));
}

TEST(PeakCell, PeakMetadataRoundTrip) {
  TempDir tmp;
  TextReadOptions raw;
  raw.normalize_intensity = false;
  MSDataset msp = read_msp_text(
      "Name: a\nNumPeaks: 3\nmz intensity formula\n91.0542 245 C7H7+\n125.0233 1000 C6H5O3+\n185.0961 349\n\n"
      "Name: b\nNumPeaks: 1\n50 1\n\n",
      raw);
  write_tsv(msp, tmp.file("out.tsv"));
  EXPECT_EQ(read_text(tmp.file("out.tsv")),
            "Name\tNumPeaks\tPeak\n"
            "a\t3\tmz,intensity,formula;91.054199999999994,245,C7H7+;"
            "125.02330000000001,1000,C6H5O3+;185.09610000000001,349\n"
            "b\t1\t50,1\n");  // no annotation in this spectrum → plain pairs

  MSDataset back = read_tsv(tmp.file("out.tsv"));
  EXPECT_EQ(back.peaks().data(), msp.peaks().data());
  EXPECT_EQ(back.peaks().metadata_columns(), (std::vector<std::string>{"formula"}));
  EXPECT_EQ(peak_column(back, "formula"),
            (std::vector<std::string>{"C7H7+", "C6H5O3+", "", ""}));

  // CSV: the whole Peak cell is quoted because it contains commas.
  write_csv(msp, tmp.file("out.csv"));
  EXPECT_NE(read_text(tmp.file("out.csv")).find("\"mz,intensity,formula;91.054199999999994"),
            std::string::npos);
  EXPECT_EQ(peak_column(read_csv(tmp.file("out.csv")), "formula"),
            (std::vector<std::string>{"C7H7+", "C6H5O3+", "", ""}));
}

TEST(PeakCell, QuotesSpecialCharacters) {
  TempDir tmp;
  MSDataset dataset(nullptr, PeakSeries::from_spectra({{{1, 2}, {3, 4}}}));
  dataset.peaks().set_metadata_column(
      "note", std::vector<Value>{std::string("loss; H2O, \"x\""), std::string(" padded ")});
  write_tsv(dataset, tmp.file("q.tsv"));
  // Peak cell:  mz,intensity,note;1,2,"loss; H2O, ""x""";3,4," padded "
  // The cell contains '"', so the TSV field is quoted too (quotes doubled again).
  EXPECT_EQ(read_text(tmp.file("q.tsv")),
            "Peak\n\"mz,intensity,note;1,2,\"\"loss; H2O, \"\"\"\"x\"\"\"\"\"\";"
            "3,4,\"\" padded \"\"\"\n");
  write_csv(dataset, tmp.file("q.csv"));
  for (const char* name : {"q.tsv", "q.csv"}) {
    MSDataset back = load_ms_dataset(tmp.file(name));
    EXPECT_EQ(peak_column(back, "note"),
              (std::vector<std::string>{"loss; H2O, \"x\"", " padded "}))
        << name;
  }
}

TEST(PeakCell, RowsWithDifferentColumns) {
  MSDataset dataset = read_tsv_text(
      "Name\tPeak\n"
      "a\tmz,intensity,formula;1,1,X\n"
      "b\tMZ,Intensity,note,formula;2,2,n1;3,3,,Y\n"
      "c\t4,4\n");
  EXPECT_EQ(dataset.peaks().metadata_columns(), (std::vector<std::string>{"formula", "note"}));
  EXPECT_EQ(peak_column(dataset, "formula"), (std::vector<std::string>{"X", "", "Y", ""}));
  EXPECT_EQ(peak_column(dataset, "note"), (std::vector<std::string>{"", "n1", "", ""}));
}

TEST(PeakCell, Errors) {
  EXPECT_THROW(read_tsv_text("Peak\n1,2,extra\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Peak\nmz,intensity,a;1,2,x,y\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Peak\nmz,intensity,a,a;1,2\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Peak\nmz,intensity,a;1,2,\"open\n"), std::invalid_argument);
  EXPECT_THROW(read_tsv_text("Peak\nmz,intensity,a;1,2,\"x\"y\n"), std::invalid_argument);
}

TEST(PeakCell, PeakHeadersSelectColumns) {
  TempDir tmp;
  MSDataset dataset(nullptr, PeakSeries::from_spectra({{{1, 2}}}));
  dataset.peaks().set_metadata_column("a", Value(std::string("x")));
  dataset.peaks().set_metadata_column("b", Value(std::string("y")));
  DelimitedWriteOptions options;
  options.peak_headers = std::vector<std::string>{"b"};
  write_tsv(dataset, tmp.file("b.tsv"), options);
  EXPECT_EQ(read_text(tmp.file("b.tsv")), "Peak\nmz,intensity,b;1,2,y\n");
  options.peak_headers = std::vector<std::string>{};
  write_tsv(dataset, tmp.file("plain.tsv"), options);
  EXPECT_EQ(read_text(tmp.file("plain.tsv")), "Peak\n1,2\n");
}

}  // namespace
}  // namespace msentity::io
