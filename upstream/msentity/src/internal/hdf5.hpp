#pragma once

// Thin RAII wrapper over the HDF5 C API, writing the types of the MSDS
// specification (docs/format/msds.md §2):
//   string attribute/dataset -> variable-length UTF-8 string
//   bool attribute           -> enum(int8) {FALSE=0, TRUE=1}
//   integer attribute        -> 64-bit little-endian integer

#include <hdf5.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace msentity::internal::h5 {

class Handle {
 public:
  using Closer = herr_t (*)(hid_t);
  Handle() = default;
  Handle(hid_t id, Closer closer);
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept;
  Handle& operator=(Handle&& other) noexcept;
  ~Handle();

  hid_t id() const { return id_; }
  explicit operator bool() const { return id_ >= 0; }
  void close();

 private:
  hid_t id_ = -1;
  Closer closer_ = nullptr;
};

// Suppress HDF5's default error printing; errors become exceptions.
void silence_errors();

enum class OpenMode { ReadOnly, ReadWrite, Truncate, ReadWriteOrCreate };
Handle open_file(const std::string& path, OpenMode mode);
// Read-only file opened from an in-memory image.
Handle open_image(const void* data, std::size_t size);
// Writable in-memory file; retrieve its bytes with file_image().
Handle create_memory_file();
std::vector<uint8_t> file_image(const Handle& file);

Handle open_group(hid_t location, const std::string& name);
Handle create_group(hid_t location, const std::string& name);
bool link_exists(hid_t location, const std::string& name);
void delete_link(hid_t location, const std::string& name);
std::vector<std::string> member_names(hid_t group);

// Attributes.
bool attribute_exists(hid_t object, const std::string& name);
void delete_attribute(hid_t object, const std::string& name);
std::optional<std::string> read_string_attribute(hid_t object, const std::string& name);
std::optional<int64_t> read_int_attribute(hid_t object, const std::string& name);
std::optional<bool> read_bool_attribute(hid_t object, const std::string& name);
void write_string_attribute(hid_t object, const std::string& name, const std::string& value);
void write_int_attribute(hid_t object, const std::string& name, int64_t value);
void write_bool_attribute(hid_t object, const std::string& name, bool value);

// Datasets.
std::vector<hsize_t> dataset_shape(hid_t location, const std::string& name);
std::vector<double> read_double_dataset(hid_t location, const std::string& name);
std::vector<int64_t> read_int64_dataset(hid_t location, const std::string& name);
std::vector<uint8_t> read_bytes_dataset(hid_t location, const std::string& name);
std::vector<std::string> read_string_dataset(hid_t location, const std::string& name);

Handle write_double_dataset(hid_t location, const std::string& name, const double* data,
                            const std::vector<hsize_t>& shape);
Handle write_int64_dataset(hid_t location, const std::string& name,
                           const std::vector<int64_t>& data);
// 1-D uint8 dataset; with gzip, chunked by guess_chunk_1d.
Handle write_bytes_dataset(hid_t location, const std::string& name, const uint8_t* data,
                           std::size_t size, bool gzip);
Handle write_string_dataset(hid_t location, const std::string& name,
                            const std::vector<std::string>& values);
Handle write_scalar_string_dataset(hid_t location, const std::string& name,
                                   const std::string& value);

// Chunk length for a 1-D dataset (algorithm in docs/format/msds.md §4.2).
hsize_t guess_chunk_1d(hsize_t length, std::size_t type_size);

}  // namespace msentity::internal::h5
