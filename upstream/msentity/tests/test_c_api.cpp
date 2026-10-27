// Tests of the C API (include/msentity/c/msentity_c.h), the interface used by
// the Python, Java, and C# bindings.

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/table.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "msentity/c/msentity_c.h"
#include "test_helpers.hpp"

namespace {

std::string example_msp() { return std::string(MSENTITY_EXAMPLES_DIR) + "/example.msp"; }

// RAII owner for a dataset handle.
struct Handle {
  msentity_dataset* ptr = nullptr;
  ~Handle() { msentity_free(ptr); }
  msentity_dataset** out() { return &ptr; }
};

std::string take_string(char* text) {
  std::string out = text;
  msentity_string_free(text);
  return out;
}

msentity_value text_value(const char* text) {
  msentity_value v{};
  v.type = MSENTITY_TYPE_STRING;
  v.string_value = const_cast<char*>(text);
  return v;
}

TEST(CApi, LoadAndInspect) {
  Handle d;
  ASSERT_EQ(msentity_load(example_msp().c_str(), nullptr, nullptr, 0, d.out()), MSENTITY_OK);
  EXPECT_EQ(msentity_size(d.ptr), 2);
  EXPECT_EQ(msentity_n_peaks_total(d.ptr), 7);
  EXPECT_EQ(msentity_n_columns(d.ptr), 6);
  char* name = nullptr;
  ASSERT_EQ(msentity_column_name(d.ptr, 0, &name), MSENTITY_OK);
  EXPECT_EQ(take_string(name), "Name");
  EXPECT_EQ(msentity_has_column(d.ptr, "NumPeaks"), 1);

  msentity_value v{};
  ASSERT_EQ(msentity_get_value(d.ptr, 1, "CollisionEnergy", &v), MSENTITY_OK);
  EXPECT_EQ(v.type, MSENTITY_TYPE_STRING);
  EXPECT_STREQ(v.string_value, "20.0");
  msentity_values_clear(&v, 1);

  std::vector<msentity_value> names(2);
  ASSERT_EQ(msentity_get_column(d.ptr, "Name", names.data(), 2), MSENTITY_OK);
  EXPECT_STREQ(names[1].string_value, "MassSpecID0000002");
  msentity_values_clear(names.data(), 2);
}

TEST(CApi, ErrorsBecomeStatusCodes) {
  Handle d;
  EXPECT_EQ(msentity_load("/nonexistent/file.msds", nullptr, nullptr, 0, d.out()), MSENTITY_ERR_IO);
  EXPECT_NE(std::string(msentity_last_error()), "");
  EXPECT_EQ(msentity_load("data.unknown", nullptr, nullptr, 0, d.out()),
            MSENTITY_ERR_INVALID_ARGUMENT);
  ASSERT_EQ(msentity_load(example_msp().c_str(), nullptr, nullptr, 0, d.out()), MSENTITY_OK);
  EXPECT_STREQ(msentity_last_error(), "");
  msentity_value v{};
  EXPECT_EQ(msentity_get_value(d.ptr, 0, "Missing", &v), MSENTITY_ERR_KEY);
  EXPECT_EQ(msentity_get_value(d.ptr, 5, "Name", &v), MSENTITY_ERR_OUT_OF_RANGE);
  std::vector<msentity_value> wrong(3);
  EXPECT_EQ(msentity_get_column(d.ptr, "Name", wrong.data(), 3), MSENTITY_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(msentity_get_value(nullptr, 0, "Name", &v), MSENTITY_ERR_INVALID_ARGUMENT);
}

TEST(CApi, ViewsShareData) {
  Handle d, v, copy;
  ASSERT_EQ(msentity_load(example_msp().c_str(), nullptr, nullptr, 0, d.out()), MSENTITY_OK);
  int64_t positions[] = {1};
  ASSERT_EQ(msentity_select(d.ptr, positions, 1, v.out()), MSENTITY_OK);
  EXPECT_EQ(msentity_size(v.ptr), 1);
  msentity_value flag{};
  flag.type = MSENTITY_TYPE_BOOL;
  flag.bool_value = 1;
  ASSERT_EQ(msentity_set_column_scalar(v.ptr, "Checked", &flag), MSENTITY_OK);

  Handle full;
  ASSERT_EQ(msentity_full_view(d.ptr, 1, full.out()), MSENTITY_OK);
  std::vector<msentity_value> checked(2);
  ASSERT_EQ(msentity_get_column(full.ptr, "Checked", checked.data(), 2), MSENTITY_OK);
  EXPECT_EQ(checked[0].type, MSENTITY_TYPE_NULL);
  EXPECT_EQ(checked[1].type, MSENTITY_TYPE_BOOL);

  ASSERT_EQ(msentity_copy(v.ptr, copy.out()), MSENTITY_OK);
  std::vector<int64_t> offsets(2);
  ASSERT_EQ(msentity_offsets(copy.ptr, offsets.data(), 2), MSENTITY_OK);
  EXPECT_EQ(offsets, (std::vector<int64_t>{0, 3}));

  Handle sorted, filtered, sliced;
  ASSERT_EQ(msentity_sort_by(d.ptr, "CollisionEnergy", 1, sorted.out()), MSENTITY_OK);
  msentity_value first{};
  ASSERT_EQ(msentity_get_value(sorted.ptr, 0, "Name", &first), MSENTITY_OK);
  EXPECT_STREQ(first.string_value, "MassSpecID0000002");
  msentity_values_clear(&first, 1);
  uint8_t mask[] = {1, 0};
  ASSERT_EQ(msentity_filter(d.ptr, mask, 2, filtered.out()), MSENTITY_OK);
  EXPECT_EQ(msentity_size(filtered.ptr), 1);
  ASSERT_EQ(msentity_slice(d.ptr, 1, 2, sliced.out()), MSENTITY_OK);
  EXPECT_EQ(msentity_n_peaks_total(sliced.ptr), 3);
}

TEST(CApi, PeaksAndPeakMetadata) {
  Handle d;
  ASSERT_EQ(msentity_load(example_msp().c_str(), nullptr, nullptr, 0, d.out()), MSENTITY_OK);
  std::vector<double> mz(7), intensity(7);
  ASSERT_EQ(msentity_peaks(d.ptr, mz.data(), intensity.data(), 7), MSENTITY_OK);
  EXPECT_DOUBLE_EQ(mz[0], 91.0542);
  EXPECT_DOUBLE_EQ(intensity[0], 0.245);
  int64_t n = 0;
  ASSERT_EQ(msentity_spectrum_size(d.ptr, 1, &n), MSENTITY_OK);
  EXPECT_EQ(n, 3);
  std::vector<double> spectrum_mz(3);
  ASSERT_EQ(msentity_spectrum_peaks(d.ptr, 1, spectrum_mz.data(), nullptr, 3), MSENTITY_OK);
  EXPECT_DOUBLE_EQ(spectrum_mz[2], 246.1125);

  EXPECT_EQ(msentity_n_peak_columns(d.ptr), 1);
  std::vector<msentity_value> formula(7);
  ASSERT_EQ(msentity_get_peak_column(d.ptr, "formula", formula.data(), 7), MSENTITY_OK);
  EXPECT_STREQ(formula[1].string_value, "C6H5O3+");
  msentity_values_clear(formula.data(), 7);

  ASSERT_EQ(msentity_normalize(d.ptr, 100.0), MSENTITY_OK);
  ASSERT_EQ(msentity_sort_peaks(d.ptr, 1, 0), MSENTITY_OK);
  ASSERT_EQ(msentity_spectrum_peaks(d.ptr, 0, nullptr, intensity.data(), 4), MSENTITY_OK);
  EXPECT_DOUBLE_EQ(intensity[0], 100.0);
}

TEST(CApi, BuildEditAndSave) {
  msentity::testing::TempDir tmp;
  double data[] = {100.0, 1.0, 150.0, 0.5, 90.0, 1.0};
  int64_t offsets[] = {0, 2, 3};
  Handle d;
  ASSERT_EQ(msentity_from_peaks(data, 3, offsets, 3, d.out()), MSENTITY_OK);
  msentity_value names[] = {text_value("a"), text_value("b")};
  ASSERT_EQ(msentity_set_column(d.ptr, "Name", names, 2), MSENTITY_OK);
  msentity_value score{};
  score.type = MSENTITY_TYPE_FLOAT;
  score.float_value = 0.5;
  ASSERT_EQ(msentity_set_value(d.ptr, 1, "Score", &score), MSENTITY_OK);
  ASSERT_EQ(msentity_set_description(d.ptr, "built in C"), MSENTITY_OK);
  ASSERT_EQ(msentity_set_attribute(d.ptr, "source", "test"), MSENTITY_OK);
  int32_t added = 0;
  ASSERT_EQ(msentity_add_tag(d.ptr, "c-api", &added), MSENTITY_OK);
  EXPECT_EQ(added, 1);
  int32_t assigned = 0;
  ASSERT_EQ(msentity_set_spec_id(d.ptr, nullptr, "SP", 0, 1, &assigned), MSENTITY_OK);
  EXPECT_EQ(assigned, 1);
  msentity_value notes[] = {text_value("x"), text_value(""), text_value("y")};
  ASSERT_EQ(msentity_set_peak_column(d.ptr, "note", notes, 3), MSENTITY_OK);

  std::string path = tmp.file("built.msds");
  ASSERT_EQ(msentity_save(d.ptr, path.c_str(), nullptr, 1, 0), MSENTITY_OK);
  Handle back;
  ASSERT_EQ(msentity_load(path.c_str(), nullptr, nullptr, 0, back.out()), MSENTITY_OK);
  char* description = nullptr;
  ASSERT_EQ(msentity_get_description(back.ptr, &description), MSENTITY_OK);
  EXPECT_EQ(take_string(description), "built in C");
  char* key = nullptr;
  char* value = nullptr;
  ASSERT_EQ(msentity_attribute(back.ptr, 0, &key, &value), MSENTITY_OK);
  EXPECT_EQ(take_string(key), "source");
  EXPECT_EQ(take_string(value), "test");
  char* tag = nullptr;
  ASSERT_EQ(msentity_tag(back.ptr, 0, &tag), MSENTITY_OK);
  EXPECT_EQ(take_string(tag), "c-api");
  msentity_value id{};
  ASSERT_EQ(msentity_get_value(back.ptr, 1, "SpecID", &id), MSENTITY_OK);
  EXPECT_STREQ(id.string_value, "SP2");
  msentity_values_clear(&id, 1);

  for (const char* ext : {"msp", "mgf", "tsv", "csv"}) {
    std::string out = tmp.file(std::string("built.") + ext);
    ASSERT_EQ(msentity_save(d.ptr, out.c_str(), nullptr, 1, 0), MSENTITY_OK) << ext;
    Handle reread;
    ASSERT_EQ(msentity_load(out.c_str(), nullptr, nullptr, 0, reread.out()), MSENTITY_OK) << ext;
    EXPECT_EQ(msentity_size(reread.ptr), 2) << ext;
  }

  const msentity_dataset* parts[] = {d.ptr, back.ptr};
  Handle joined;
  ASSERT_EQ(msentity_concat(parts, 2, joined.out()), MSENTITY_OK);
  EXPECT_EQ(msentity_size(joined.ptr), 4);
}

TEST(CApi, MsdsBytesRoundTrip) {
  Handle d, back;
  ASSERT_EQ(msentity_load(example_msp().c_str(), nullptr, nullptr, 0, d.out()), MSENTITY_OK);
  uint8_t* bytes = nullptr;
  int64_t size = 0;
  ASSERT_EQ(msentity_to_msds_bytes(d.ptr, &bytes, &size), MSENTITY_OK);
  ASSERT_GT(size, 8);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(bytes) + 1, 3), "HDF");
  ASSERT_EQ(msentity_from_msds_bytes(bytes, size, back.out()), MSENTITY_OK);
  msentity_bytes_free(bytes);
  EXPECT_EQ(msentity_size(back.ptr), 2);
  EXPECT_EQ(msentity_n_peaks_total(back.ptr), 7);
  EXPECT_EQ(msentity_from_msds_bytes(reinterpret_cast<const uint8_t*>("not hdf5"), 8, back.out()),
            MSENTITY_ERR_FORMAT);
}

TEST(CApi, ReadText) {
  Handle d;
  ASSERT_EQ(msentity_read_text("Name\tID\tPeak\nA\t007\t1,2\n", "tsv", 1, d.out()), MSENTITY_OK);
  msentity_value v{};
  ASSERT_EQ(msentity_get_value(d.ptr, 0, "ID", &v), MSENTITY_OK);
  EXPECT_EQ(v.type, MSENTITY_TYPE_INT);
  EXPECT_EQ(v.int_value, 7);
  EXPECT_EQ(msentity_read_text("x", "xml", 0, d.out()), MSENTITY_ERR_INVALID_ARGUMENT);
}

TEST(CApi, ArrowExport) {
  Handle d;
  ASSERT_EQ(msentity_load(example_msp().c_str(), nullptr, nullptr, 0, d.out()), MSENTITY_OK);
  ArrowArrayStream stream{};
  ASSERT_EQ(msentity_metadata_arrow(d.ptr, &stream), MSENTITY_OK);
  auto reader = arrow::ImportRecordBatchReader(&stream).ValueOrDie();
  auto table = reader->ToTable().ValueOrDie();
  EXPECT_EQ(table->num_rows(), 2);
  EXPECT_EQ(table->num_columns(), 6);

  ArrowArrayStream peak_stream{};
  ASSERT_EQ(msentity_peak_metadata_arrow(d.ptr, &peak_stream), MSENTITY_OK);
  auto peaks = arrow::ImportRecordBatchReader(&peak_stream).ValueOrDie()->ToTable().ValueOrDie();
  EXPECT_EQ(peaks->num_rows(), 7);
  EXPECT_EQ(peaks->schema()->field(0)->name(), "formula");
}

}  // namespace
