#pragma once

#include <arrow/api.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "internal/table.hpp"
#include "msentity/msentity.hpp"

namespace msentity::testing {

inline std::string data_path(const std::string& name) {
  return std::string(MSENTITY_TEST_DATA_DIR) + "/" + name;
}

// Table from (name, values) columns; types are inferred like set_column.
inline std::shared_ptr<arrow::Table> make_table(
    const std::vector<std::pair<std::string, std::vector<Value>>>& columns) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  int64_t rows = columns.empty() ? 0 : static_cast<int64_t>(columns.front().second.size());
  for (const auto& [name, values] : columns) {
    auto array = internal::values_to_array(values);
    fields.push_back(arrow::field(name, array->type()));
    arrays.push_back(array);
  }
  return arrow::Table::Make(arrow::schema(fields), arrays, rows);
}

inline std::vector<Value> strings(std::initializer_list<const char*> values) {
  std::vector<Value> out;
  for (const char* v : values) out.emplace_back(std::string(v));
  return out;
}

inline std::vector<Value> ints(std::initializer_list<int64_t> values) {
  return std::vector<Value>(values.begin(), values.end());
}

inline std::vector<Value> doubles(std::initializer_list<double> values) {
  return std::vector<Value>(values.begin(), values.end());
}

inline std::vector<std::string> as_strings(const std::vector<Value>& values) {
  std::vector<std::string> out;
  for (const auto& v : values) out.push_back(to_display_string(v));
  return out;
}

// Fixture shared by the PeakSeries / MSDataset tests.
inline PeakSeries sample_peak_series(const std::string& group_column = "group") {
  return PeakSeries({100.0, 10.0, 101.0, 30.0, 102.0, 20.0, 200.0, 5.0, 201.0, 15.0, 300.0, 7.0},
                    {0, 3, 5, 6},
                    make_table({{"annotation", strings({"a", "b", "c", "d", "e", "f"})},
                                {group_column, ints({1, 1, 1, 2, 2, 3})}}));
}

inline MSDataset sample_dataset() {
  auto metadata = make_table({{"name", strings({"spec1", "spec2", "spec3"})},
                              {"precursor_mz", doubles({150.0, 250.0, 350.0})},
                              {"SMILES", strings({"CCO", "CCC", "CCN"})}});
  return MSDataset(metadata, sample_peak_series("peak_group"), std::nullopt, "test dataset",
                   {{"source", "unit-test"}}, {"tag1"});
}

class TempDir {
 public:
  TempDir() {
    std::random_device random;
    path_ = std::filesystem::temp_directory_path() /
            ("msentity-test-" + std::to_string(random()) + std::to_string(random()));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  std::string file(const std::string& name) const { return (path_ / name).string(); }
  std::string write(const std::string& name, const std::string& text) const {
    std::ofstream(file(name), std::ios::binary) << text;
    return file(name);
  }

 private:
  std::filesystem::path path_;
};

inline std::string read_text(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(stream), {});
}

}  // namespace msentity::testing
