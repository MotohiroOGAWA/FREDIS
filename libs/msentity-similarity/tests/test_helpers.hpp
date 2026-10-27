#pragma once

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include <msentity/msentity.hpp>
#include <msentity/similarity/similarity.hpp>

namespace mssim_test {

inline std::string msentity_data(const std::string& name) {
  return std::string(MSENTITY_TEST_DATA_DIR) + "/" + name;
}
inline std::string own_data(const std::string& name) {
  return std::string(MSSIM_TEST_DATA_DIR) + "/" + name;
}
inline std::string example(const std::string& name) {
  return std::string(MSENTITY_EXAMPLES_DIR) + "/" + name;
}

class TempDir {
 public:
  TempDir() {
    std::random_device random;
    path_ = std::filesystem::temp_directory_path() /
            ("mssim-test-" + std::to_string(random()) + std::to_string(random()));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  std::string file(const std::string& name) const { return (path_ / name).string(); }

 private:
  std::filesystem::path path_;
};

inline std::string read_text(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(stream), {});
}

}  // namespace mssim_test
