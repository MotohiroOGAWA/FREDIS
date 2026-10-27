// Ports of tests/TestIO/TestMsp.py and TestMgf.py.

#include <gtest/gtest.h>

#include "test_helpers.hpp"

namespace msentity::io {
namespace {

using msentity::testing::as_strings;
using msentity::testing::read_text;
using msentity::testing::TempDir;

const char* kTwoSpectraMsp =
    "Name: spec1\n"
    "PrecursorMZ: 100.0\n"
    "SMILES: CCO\n"
    "NumPeaks: 3\n"
    "50.0\t10.0\n"
    "60.0\t20.0\n"
    "70.0\t30.0\n"
    "\n"
    "Name: spec2\n"
    "PrecursorMZ: 200.0\n"
    "SMILES: CCC\n"
    "NumPeaks: 2\n"
    "80.0\t40.0\n"
    "90.0\t50.0\n"
    "\n";

const char* kTwoSpectraMgf =
    "BEGIN IONS\n"
    "TITLE=spec1\n"
    "PEPMASS=100.0\n"
    "50.0 10.0\n"
    "60.0 20.0\n"
    "END IONS\n"
    "\n"
    "BEGIN IONS\n"
    "TITLE=spec2\n"
    "PEPMASS=200.0\n"
    "70.0 30.0\n"
    "END IONS\n";

TextReadOptions quiet() { return TextReadOptions{}; }

TEST(Msp, ReadsBasicFile) {
  TempDir tmp;
  MSDataset dataset = read_msp(tmp.write("a.msp", kTwoSpectraMsp));
  ASSERT_EQ(dataset.size(), 2);
  EXPECT_EQ(dataset.columns(), (std::vector<std::string>{"Name", "PrecursorMZ", "SMILES", "NumPeaks"}));
  EXPECT_EQ(std::get<std::string>(dataset.value(1, "Name")), "spec2");
  EXPECT_EQ(dataset[0].n_peaks(), 3);
  EXPECT_EQ(dataset[1].n_peaks(), 2);
  // Intensities are normalized to the base peak by default.
  EXPECT_EQ(dataset.peaks().intensity(), (std::vector<double>{10.0 / 30, 20.0 / 30, 1.0, 0.8, 1.0}));
}

TEST(Msp, WithoutTrailingBlankLineAndRawIntensities) {
  TextReadOptions options;
  options.normalize_intensity = false;
  MSDataset dataset = read_msp_text("Name: spec1\nPrecursorMZ: 100.0\nNumPeaks: 2\n50.0\t10.0\n60.0\t20.0", options);
  ASSERT_EQ(dataset.size(), 1);
  EXPECT_EQ(dataset.peaks().intensity(), (std::vector<double>{10.0, 20.0}));
  EXPECT_THROW(read_msp_text("  \n "), std::invalid_argument);
}

TEST(Msp, ProgressCallbackAndHeaderMap) {
  TempDir tmp;
  std::string path = tmp.write("a.msp", "NAME: x\nprecursor_type: [M+H]1+\nNum Peaks: 1\n1 2\n");
  int64_t last_processed = -1, last_total = -1, last_records = -1;
  TextReadOptions options;
  options.progress_callback = [&](int64_t processed, int64_t total, int64_t, int64_t records) {
    last_processed = processed;
    last_total = total;
    last_records = records;
  };
  TextReadResult result = read_msp_with_header_map(path, options);
  EXPECT_EQ(last_processed, last_total);
  EXPECT_EQ(last_records, 1);
  EXPECT_EQ(result.header_map, (HeaderMap{{"Name", "NAME"}, {"AdductType", "precursor_type"},
                                          {"NumPeaks", "Num Peaks"}}));
  EXPECT_EQ(std::get<std::string>(result.dataset.value(0, "AdductType")), "[M+H]+");
}

TEST(Msp, SpecIdPrefix) {
  TextReadOptions options;
  options.spec_id_prefix = "SP";
  MSDataset dataset = read_msp_text(kTwoSpectraMsp, options);
  EXPECT_EQ(as_strings(dataset.column_values("SpecID")), (std::vector<std::string>{"SP1", "SP2"}));
}

TEST(Msp, PeakHeaderColumnsAndCustomParser) {
  const char* text =
      "Name: a\nNumPeaks: 2\nmz intensity frag\n50 10 \"x;y\"\n60 20\n\n";
  MSDataset dataset = read_msp_text(text);
  EXPECT_EQ(dataset.peaks().metadata_columns(), (std::vector<std::string>{"frag"}));
  auto values = internal::to_values(*dataset.peaks().metadata()->column(0));
  EXPECT_EQ(as_strings(values), (std::vector<std::string>{"x;y", ""}));

  TextReadOptions options;
  options.normalize_intensity = false;
  options.peak_parser = [](const std::string& line) {
    PeakEntry entry;
    auto comma = line.find(',');
    entry.mz = std::stod(line.substr(0, comma));
    entry.intensity = std::stod(line.substr(comma + 1));
    entry.metadata = {{"source", "custom"}};
    return entry;
  };
  dataset = read_msp_text("Name: a\nNumPeaks: 1\n5,6\n", options);
  EXPECT_EQ(dataset.peaks().data(), (std::vector<double>{5, 6}));
  EXPECT_EQ(dataset.peaks().metadata_columns(), (std::vector<std::string>{"source"}));
}

TEST(Msp, DuplicateMetaSkipsRecordAndWritesErrorLog) {
  TempDir tmp;
  TextReadOptions options;
  options.error_log_level = ErrorLogLevel::Detail;
  options.error_log_file = tmp.file("errors.txt");
  MSDataset dataset = read_msp_text(
      "Name: bad\nNAME: again\nNumPeaks: 1\n1 1\n\nName: good\nNumPeaks: 1\n2 2\n\n", options);
  ASSERT_EQ(dataset.size(), 1);
  EXPECT_EQ(std::get<std::string>(dataset.value(0, "Name")), "good");
  std::string log = read_text(tmp.file("errors.txt"));
  EXPECT_NE(log.find("[ERROR] Line (00002) NAME: again | Duplicate meta key"), std::string::npos);
  EXPECT_NE(log.find("[CONTEXT]"), std::string::npos);

  options.allow_duplicate_cols = true;
  options.error_log_level = ErrorLogLevel::None;
  dataset = read_msp_text("Name: a\nNAME: b\nNumPeaks: 0\n\n", options);
  EXPECT_EQ(std::get<std::string>(dataset.value(0, "Name1")), "b");
}

TEST(Msp, WriteBasicHeaderMapAndSelection) {
  TempDir tmp;
  MSDataset dataset = read_msp_text(kTwoSpectraMsp);
  write_msp(dataset, tmp.file("out.msp"));
  EXPECT_EQ(read_text(tmp.file("out.msp")),
            "Name: spec1\nPrecursorMZ: 100.0\nSMILES: CCO\nNumPeaks: 3\n"
            "50.0\t0.3333333333333333\n60.0\t0.6666666666666666\n70.0\t1.0\n\n"
            "Name: spec2\nPrecursorMZ: 200.0\nSMILES: CCC\nNumPeaks: 2\n"
            "80.0\t0.8\n90.0\t1.0\n\n");

  TextWriteOptions options;
  options.headers = std::vector<std::string>{"SMILES", "Missing"};
  options.header_map = {{"SMILES", "Structure"}};
  options.delimiter = " ";
  write_msp(dataset.slice(0, 1), tmp.file("sel.msp"), options);
  EXPECT_EQ(read_text(tmp.file("sel.msp")),
            "Structure: CCO\nNumPeaks: 3\n50.0 0.3333333333333333\n60.0 0.6666666666666666\n70.0 1.0\n\n");
}

TEST(Msp, WritePeakHeadersAndRoundTrip) {
  TempDir tmp;
  MSDataset dataset = read_msp_text(kTwoSpectraMsp);
  dataset.peaks().set_metadata_column(
      "ann", std::vector<Value>{std::string("a\"q"), std::string(""), std::string("c"),
                                std::monostate{}, std::monostate{}});
  write_msp(dataset, tmp.file("ann.msp"));
  std::string text = read_text(tmp.file("ann.msp"));
  EXPECT_NE(text.find("NumPeaks: 3\nmz\tintensity\tann\n50.0\t0.3333333333333333\t\"a'q\"\n60.0\t0.6666666666666666\n"),
            std::string::npos);
  // The second spectrum has no annotations, so no peak header is written.
  EXPECT_NE(text.find("NumPeaks: 2\n80.0\t0.8\n"), std::string::npos);

  MSDataset reread = read_msp(tmp.file("ann.msp"));
  EXPECT_EQ(reread.size(), 2);
  EXPECT_EQ(reread.peaks().data(), dataset.peaks().data());
  EXPECT_EQ(as_strings(internal::to_values(*reread.peaks().metadata()->column(0))),
            (std::vector<std::string>{"a'q", "", "c", "", ""}));
}

TEST(Mgf, ReadsBasicFileAndIgnoresOutsideLines) {
  TempDir tmp;
  MSDataset dataset = read_mgf(tmp.write("a.mgf", std::string("junk=1\n") + kTwoSpectraMgf));
  ASSERT_EQ(dataset.size(), 2);
  EXPECT_EQ(dataset.columns(), (std::vector<std::string>{"TITLE", "PEPMASS"}));
  EXPECT_EQ(dataset[0].n_peaks(), 2);
  EXPECT_EQ(dataset.peaks().intensity(), (std::vector<double>{0.5, 1.0, 1.0}));
}

TEST(Mgf, WithoutEndIonsAndHeaderColumns) {
  MSDataset dataset = read_mgf_text(
      "BEGIN IONS\nTITLE=a\nmz intensity ann\n1 2 x\nBEGIN IONS\nTITLE=b\n3 4\n");
  ASSERT_EQ(dataset.size(), 2);
  EXPECT_EQ(dataset.peaks().metadata_columns(), (std::vector<std::string>{"ann"}));
  EXPECT_THROW(read_mgf_text(""), std::invalid_argument);
}

TEST(Mgf, WriteAndRoundTrip) {
  TempDir tmp;
  MSDataset dataset = read_mgf_text(kTwoSpectraMgf);
  write_mgf(dataset, tmp.file("out.mgf"));
  EXPECT_EQ(read_text(tmp.file("out.mgf")),
            "BEGIN IONS\nTITLE=spec1\nPEPMASS=100.0\n50.0\t0.5\n60.0\t1.0\nEND IONS\n\n"
            "BEGIN IONS\nTITLE=spec2\nPEPMASS=200.0\n70.0\t1.0\nEND IONS\n\n");
  MSDataset reread = read_mgf(tmp.file("out.mgf"));
  EXPECT_EQ(reread.peaks().data(), dataset.peaks().data());
  EXPECT_EQ(as_strings(reread.column_values("TITLE")), (std::vector<std::string>{"spec1", "spec2"}));
}

TEST(SampleFiles, ReadSampleMspAndMgf) {
  MSDataset msp = read_msp(msentity::testing::data_path("sample.msp"));
  EXPECT_EQ(msp.size(), 5);
  EXPECT_EQ(std::get<std::string>(msp.value(0, "AdductType")), "[M+H]+");
  MSDataset mgf = read_mgf(msentity::testing::data_path("sample.mgf"));
  EXPECT_EQ(mgf.size(), 6);
}

}  // namespace
}  // namespace msentity::io
