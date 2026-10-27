#include "internal/hdf5.hpp"

#include <hdf5_hl.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "msentity/value.hpp"

namespace msentity::internal::h5 {

namespace {

[[noreturn]] void fail(const std::string& what) { throw FormatError("HDF5 error: " + what); }

void ensure(herr_t status, const std::string& what) {
  if (status < 0) fail(what);
}

Handle make(hid_t id, Handle::Closer closer, const std::string& what) {
  if (id < 0) fail(what);
  return Handle(id, closer);
}

Handle vlen_utf8_type() {
  Handle type = make(H5Tcopy(H5T_C_S1), H5Tclose, "copy string type");
  ensure(H5Tset_size(type.id(), H5T_VARIABLE), "set variable string size");
  ensure(H5Tset_cset(type.id(), H5T_CSET_UTF8), "set UTF-8 charset");
  return type;
}

Handle bool_enum_type() {
  Handle type = make(H5Tenum_create(H5T_NATIVE_INT8), H5Tclose, "create bool enum");
  int8_t value = 0;
  ensure(H5Tenum_insert(type.id(), "FALSE", &value), "insert FALSE");
  value = 1;
  ensure(H5Tenum_insert(type.id(), "TRUE", &value), "insert TRUE");
  return type;
}

hssize_t npoints(hid_t space) {
  hssize_t n = H5Sget_simple_extent_npoints(space);
  if (n < 0) fail("get number of points");
  return n;
}

std::string trim_fixed_string(const char* data, std::size_t size, H5T_str_t pad) {
  std::size_t length = strnlen(data, size);
  std::string value(data, length);
  if (pad == H5T_STR_SPACEPAD) {
    while (!value.empty() && value.back() == ' ') value.pop_back();
  }
  return value;
}

// Reads a 1-element string from an attribute or dataset.
template <typename ReadFn>
std::string read_string_object(hid_t file_type, hid_t space, ReadFn read) {
  if (H5Tget_class(file_type) != H5T_STRING) fail("expected a string value");
  if (npoints(space) != 1) fail("expected a scalar string");
  if (H5Tis_variable_str(file_type) > 0) {
    Handle mem_type = vlen_utf8_type();
    char* buffer = nullptr;
    ensure(read(mem_type.id(), &buffer), "read variable-length string");
    std::string value = buffer != nullptr ? std::string(buffer) : std::string();
    H5free_memory(buffer);
    return value;
  }
  std::size_t size = H5Tget_size(file_type);
  std::vector<char> buffer(size + 1, '\0');
  Handle mem_type = make(H5Tcopy(file_type), H5Tclose, "copy fixed string type");
  ensure(read(mem_type.id(), buffer.data()), "read fixed-length string");
  return trim_fixed_string(buffer.data(), size, H5Tget_strpad(file_type));
}

// Reads one integer-like value (integer, enum, float) stored with any type.
template <typename ReadFn>
std::optional<int64_t> read_integer_object(hid_t file_type, hid_t space, ReadFn read) {
  if (npoints(space) != 1) fail("expected a scalar number");
  H5T_class_t cls = H5Tget_class(file_type);
  if (cls == H5T_INTEGER) {
    int64_t value = 0;
    ensure(read(H5T_NATIVE_INT64, &value), "read integer");
    return value;
  }
  if (cls == H5T_FLOAT) {
    double value = 0;
    ensure(read(H5T_NATIVE_DOUBLE, &value), "read float");
    return static_cast<int64_t>(value);
  }
  if (cls == H5T_ENUM) {
    Handle native = make(H5Tget_native_type(file_type, H5T_DIR_ASCEND), H5Tclose,
                         "native enum type");
    std::size_t size = H5Tget_size(native.id());
    Handle super = make(H5Tget_super(native.id()), H5Tclose, "enum base type");
    bool is_signed = H5Tget_sign(super.id()) == H5T_SGN_2;
    unsigned char raw[8] = {0};
    if (size > sizeof(raw)) fail("enum too large");
    ensure(read(native.id(), raw), "read enum");
    int64_t value = 0;
    switch (size) {
      case 1: value = is_signed ? static_cast<int8_t>(raw[0]) : raw[0]; break;
      case 2: { int16_t v; std::memcpy(&v, raw, 2); value = is_signed ? v : static_cast<uint16_t>(v); break; }
      case 4: { int32_t v; std::memcpy(&v, raw, 4); value = is_signed ? v : static_cast<uint32_t>(v); break; }
      default: std::memcpy(&value, raw, 8); break;
    }
    return value;
  }
  return std::nullopt;
}

}  // namespace

Handle::Handle(hid_t id, Closer closer) : id_(id), closer_(closer) {}

Handle::Handle(Handle&& other) noexcept : id_(other.id_), closer_(other.closer_) {
  other.id_ = -1;
}

Handle& Handle::operator=(Handle&& other) noexcept {
  if (this != &other) {
    close();
    id_ = other.id_;
    closer_ = other.closer_;
    other.id_ = -1;
  }
  return *this;
}

Handle::~Handle() { close(); }

void Handle::close() {
  if (id_ >= 0 && closer_ != nullptr) closer_(id_);
  id_ = -1;
}

void silence_errors() { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }

Handle open_file(const std::string& path, OpenMode mode) {
  silence_errors();
  switch (mode) {
    case OpenMode::ReadOnly:
      return make(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose,
                  "cannot open " + path);
    case OpenMode::ReadWrite:
      return make(H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose,
                  "cannot open " + path + " for writing");
    case OpenMode::Truncate:
      return make(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose,
                  "cannot create " + path);
    case OpenMode::ReadWriteOrCreate:
      if (std::filesystem::exists(path)) return open_file(path, OpenMode::ReadWrite);
      return open_file(path, OpenMode::Truncate);
  }
  fail("invalid open mode");
}

Handle open_image(const void* data, std::size_t size) {
  silence_errors();
  // flags = 0: the library copies the buffer and opens it read-only.
  return make(H5LTopen_file_image(const_cast<void*>(data), size, 0), H5Fclose,
              "not a valid HDF5 file image");
}

Handle create_memory_file() {
  silence_errors();
  static std::atomic<uint64_t> counter{0};
  Handle fapl = make(H5Pcreate(H5P_FILE_ACCESS), H5Pclose, "create fapl");
  ensure(H5Pset_fapl_core(fapl.id(), 1 << 20, 0), "set core driver");
  std::string name = "msentity-memory-" + std::to_string(counter.fetch_add(1)) + ".h5";
  return make(H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl.id()), H5Fclose,
              "create in-memory file");
}

std::vector<uint8_t> file_image(const Handle& file) {
  ensure(H5Fflush(file.id(), H5F_SCOPE_GLOBAL), "flush in-memory file");
  auto size = H5Fget_file_image(file.id(), nullptr, 0);
  if (size < 0) fail("get file image size");
  std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
  if (H5Fget_file_image(file.id(), bytes.data(), bytes.size()) < 0) fail("get file image");
  return bytes;
}

Handle open_group(hid_t location, const std::string& name) {
  return make(H5Gopen2(location, name.c_str(), H5P_DEFAULT), H5Gclose,
              "cannot open group '" + name + "'");
}

Handle create_group(hid_t location, const std::string& name) {
  return make(H5Gcreate2(location, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
              H5Gclose, "cannot create group '" + name + "'");
}

bool link_exists(hid_t location, const std::string& name) {
  htri_t exists = H5Lexists(location, name.c_str(), H5P_DEFAULT);
  if (exists < 0) fail("check link '" + name + "'");
  return exists > 0;
}

void delete_link(hid_t location, const std::string& name) {
  ensure(H5Ldelete(location, name.c_str(), H5P_DEFAULT), "delete '" + name + "'");
}

std::vector<std::string> member_names(hid_t group) {
  std::vector<std::string> names;
  // H5Literate / H5L_info_t map to the newest API version (H5Literate2 on
  // HDF5 >= 1.12) and also exist on HDF5 1.10.
  auto callback = [](hid_t, const char* name, const H5L_info_t*, void* data) -> herr_t {
    static_cast<std::vector<std::string>*>(data)->emplace_back(name);
    return 0;
  };
  ensure(H5Literate(group, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, callback, &names),
         "iterate group");
  return names;
}

bool attribute_exists(hid_t object, const std::string& name) {
  htri_t exists = H5Aexists(object, name.c_str());
  if (exists < 0) fail("check attribute '" + name + "'");
  return exists > 0;
}

void delete_attribute(hid_t object, const std::string& name) {
  if (attribute_exists(object, name)) {
    ensure(H5Adelete(object, name.c_str()), "delete attribute '" + name + "'");
  }
}

std::optional<std::string> read_string_attribute(hid_t object, const std::string& name) {
  if (!attribute_exists(object, name)) return std::nullopt;
  Handle attr = make(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose, "open attribute");
  Handle type = make(H5Aget_type(attr.id()), H5Tclose, "attribute type");
  Handle space = make(H5Aget_space(attr.id()), H5Sclose, "attribute space");
  return read_string_object(type.id(), space.id(), [&](hid_t mem_type, void* buffer) {
    return H5Aread(attr.id(), mem_type, buffer);
  });
}

std::optional<int64_t> read_int_attribute(hid_t object, const std::string& name) {
  if (!attribute_exists(object, name)) return std::nullopt;
  Handle attr = make(H5Aopen(object, name.c_str(), H5P_DEFAULT), H5Aclose, "open attribute");
  Handle type = make(H5Aget_type(attr.id()), H5Tclose, "attribute type");
  Handle space = make(H5Aget_space(attr.id()), H5Sclose, "attribute space");
  return read_integer_object(type.id(), space.id(), [&](hid_t mem_type, void* buffer) {
    return H5Aread(attr.id(), mem_type, buffer);
  });
}

std::optional<bool> read_bool_attribute(hid_t object, const std::string& name) {
  auto value = read_int_attribute(object, name);
  if (!value) return std::nullopt;
  return *value != 0;
}

namespace {

Handle create_attribute(hid_t object, const std::string& name, hid_t type) {
  delete_attribute(object, name);
  Handle space = make(H5Screate(H5S_SCALAR), H5Sclose, "scalar space");
  return make(H5Acreate2(object, name.c_str(), type, space.id(), H5P_DEFAULT, H5P_DEFAULT),
              H5Aclose, "create attribute '" + name + "'");
}

}  // namespace

void write_string_attribute(hid_t object, const std::string& name, const std::string& value) {
  Handle type = vlen_utf8_type();
  Handle attr = create_attribute(object, name, type.id());
  const char* data = value.c_str();
  ensure(H5Awrite(attr.id(), type.id(), &data), "write attribute '" + name + "'");
}

void write_int_attribute(hid_t object, const std::string& name, int64_t value) {
  Handle attr = create_attribute(object, name, H5T_STD_I64LE);
  ensure(H5Awrite(attr.id(), H5T_NATIVE_INT64, &value), "write attribute '" + name + "'");
}

void write_bool_attribute(hid_t object, const std::string& name, bool value) {
  Handle type = bool_enum_type();
  Handle attr = create_attribute(object, name, type.id());
  int8_t raw = value ? 1 : 0;
  ensure(H5Awrite(attr.id(), type.id(), &raw), "write attribute '" + name + "'");
}

std::vector<hsize_t> dataset_shape(hid_t location, const std::string& name) {
  Handle dataset = make(H5Dopen2(location, name.c_str(), H5P_DEFAULT), H5Dclose,
                        "open dataset '" + name + "'");
  Handle space = make(H5Dget_space(dataset.id()), H5Sclose, "dataset space");
  int rank = H5Sget_simple_extent_ndims(space.id());
  if (rank < 0) fail("dataset rank");
  std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
  if (rank > 0) H5Sget_simple_extent_dims(space.id(), dims.data(), nullptr);
  return dims;
}

namespace {

template <typename T>
std::vector<T> read_numeric(hid_t location, const std::string& name, hid_t mem_type) {
  Handle dataset = make(H5Dopen2(location, name.c_str(), H5P_DEFAULT), H5Dclose,
                        "open dataset '" + name + "'");
  Handle space = make(H5Dget_space(dataset.id()), H5Sclose, "dataset space");
  std::vector<T> values(static_cast<std::size_t>(npoints(space.id())));
  if (!values.empty()) {
    ensure(H5Dread(dataset.id(), mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
           "read dataset '" + name + "'");
  }
  return values;
}

}  // namespace

std::vector<double> read_double_dataset(hid_t location, const std::string& name) {
  return read_numeric<double>(location, name, H5T_NATIVE_DOUBLE);
}

std::vector<int64_t> read_int64_dataset(hid_t location, const std::string& name) {
  return read_numeric<int64_t>(location, name, H5T_NATIVE_INT64);
}

std::vector<uint8_t> read_bytes_dataset(hid_t location, const std::string& name) {
  Handle dataset = make(H5Dopen2(location, name.c_str(), H5P_DEFAULT), H5Dclose,
                        "open dataset '" + name + "'");
  Handle type = make(H5Dget_type(dataset.id()), H5Tclose, "dataset type");
  Handle space = make(H5Dget_space(dataset.id()), H5Sclose, "dataset space");
  std::size_t count = static_cast<std::size_t>(npoints(space.id()));
  if (H5Tget_class(type.id()) == H5T_INTEGER && H5Tget_size(type.id()) == 1) {
    std::vector<uint8_t> bytes(count);
    if (count > 0) {
      ensure(H5Dread(dataset.id(), H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     bytes.data()),
             "read bytes '" + name + "'");
    }
    return bytes;
  }
  // Opaque/void or other fixed-size types: return the raw element bytes.
  std::vector<uint8_t> bytes(count * H5Tget_size(type.id()));
  if (!bytes.empty()) {
    ensure(H5Dread(dataset.id(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, bytes.data()),
           "read raw bytes '" + name + "'");
  }
  return bytes;
}

std::vector<std::string> read_string_dataset(hid_t location, const std::string& name) {
  Handle dataset = make(H5Dopen2(location, name.c_str(), H5P_DEFAULT), H5Dclose,
                        "open dataset '" + name + "'");
  Handle type = make(H5Dget_type(dataset.id()), H5Tclose, "dataset type");
  Handle space = make(H5Dget_space(dataset.id()), H5Sclose, "dataset space");
  if (H5Tget_class(type.id()) != H5T_STRING) fail("'" + name + "' is not a string dataset");
  std::size_t count = static_cast<std::size_t>(npoints(space.id()));
  std::vector<std::string> values;
  values.reserve(count);
  if (count == 0) return values;
  if (H5Tis_variable_str(type.id()) > 0) {
    Handle mem_type = vlen_utf8_type();
    std::vector<char*> buffer(count, nullptr);
    ensure(H5Dread(dataset.id(), mem_type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()),
           "read strings '" + name + "'");
    for (char* item : buffer) values.emplace_back(item != nullptr ? item : "");
#if H5_VERSION_GE(1, 12, 0)
    H5Treclaim(mem_type.id(), space.id(), H5P_DEFAULT, buffer.data());
#else
    H5Dvlen_reclaim(mem_type.id(), space.id(), H5P_DEFAULT, buffer.data());
#endif
    return values;
  }
  std::size_t size = H5Tget_size(type.id());
  std::vector<char> buffer(count * size);
  ensure(H5Dread(dataset.id(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()),
         "read fixed strings '" + name + "'");
  for (std::size_t i = 0; i < count; ++i) {
    values.push_back(trim_fixed_string(buffer.data() + i * size, size, H5Tget_strpad(type.id())));
  }
  return values;
}

Handle write_double_dataset(hid_t location, const std::string& name, const double* data,
                            const std::vector<hsize_t>& shape) {
  Handle space = make(H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr),
                      H5Sclose, "dataspace");
  Handle dataset = make(H5Dcreate2(location, name.c_str(), H5T_IEEE_F64LE, space.id(),
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                        H5Dclose, "create dataset '" + name + "'");
  hsize_t total = 1;
  for (hsize_t d : shape) total *= d;
  if (total > 0) {
    ensure(H5Dwrite(dataset.id(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
           "write dataset '" + name + "'");
  }
  return dataset;
}

Handle write_int64_dataset(hid_t location, const std::string& name,
                           const std::vector<int64_t>& data) {
  hsize_t dims[1] = {data.size()};
  Handle space = make(H5Screate_simple(1, dims, nullptr), H5Sclose, "dataspace");
  Handle dataset = make(H5Dcreate2(location, name.c_str(), H5T_STD_I64LE, space.id(),
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                        H5Dclose, "create dataset '" + name + "'");
  if (!data.empty()) {
    ensure(H5Dwrite(dataset.id(), H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()),
           "write dataset '" + name + "'");
  }
  return dataset;
}

Handle write_bytes_dataset(hid_t location, const std::string& name, const uint8_t* data,
                           std::size_t size, bool gzip) {
  hsize_t dims[1] = {size};
  Handle space = make(H5Screate_simple(1, dims, nullptr), H5Sclose, "dataspace");
  Handle dcpl = make(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "dataset creation plist");
  if (gzip && size > 0) {
    hsize_t chunk[1] = {guess_chunk_1d(size, 1)};
    ensure(H5Pset_chunk(dcpl.id(), 1, chunk), "set chunk");
    ensure(H5Pset_deflate(dcpl.id(), 4), "set gzip");  // gzip level 4
  }
  Handle dataset = make(H5Dcreate2(location, name.c_str(), H5T_STD_U8LE, space.id(),
                                   H5P_DEFAULT, dcpl.id(), H5P_DEFAULT),
                        H5Dclose, "create dataset '" + name + "'");
  if (size > 0) {
    ensure(H5Dwrite(dataset.id(), H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
           "write dataset '" + name + "'");
  }
  return dataset;
}

Handle write_string_dataset(hid_t location, const std::string& name,
                            const std::vector<std::string>& values) {
  Handle type = vlen_utf8_type();
  hsize_t dims[1] = {values.size()};
  Handle space = make(H5Screate_simple(1, dims, nullptr), H5Sclose, "dataspace");
  Handle dataset = make(H5Dcreate2(location, name.c_str(), type.id(), space.id(), H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT),
                        H5Dclose, "create dataset '" + name + "'");
  std::vector<const char*> pointers;
  pointers.reserve(values.size());
  for (const auto& value : values) pointers.push_back(value.c_str());
  if (!pointers.empty()) {
    ensure(H5Dwrite(dataset.id(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, pointers.data()),
           "write dataset '" + name + "'");
  }
  return dataset;
}

Handle write_scalar_string_dataset(hid_t location, const std::string& name,
                                   const std::string& value) {
  Handle type = vlen_utf8_type();
  Handle space = make(H5Screate(H5S_SCALAR), H5Sclose, "scalar space");
  Handle dataset = make(H5Dcreate2(location, name.c_str(), type.id(), space.id(), H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT),
                        H5Dclose, "create dataset '" + name + "'");
  const char* data = value.c_str();
  ensure(H5Dwrite(dataset.id(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &data),
         "write dataset '" + name + "'");
  return dataset;
}

hsize_t guess_chunk_1d(hsize_t length, std::size_t type_size) {
  constexpr double kChunkBase = 16 * 1024;
  constexpr double kChunkMin = 8 * 1024;
  constexpr double kChunkMax = 1024 * 1024;
  double chunk = length == 0 ? 1024.0 : static_cast<double>(length);
  double dset_size = chunk * static_cast<double>(type_size);
  double target = kChunkBase * std::pow(2.0, std::log10(dset_size / (1024.0 * 1024.0)));
  if (target > kChunkMax) {
    target = kChunkMax;
  } else if (target < kChunkMin) {
    target = kChunkMin;
  }
  while (true) {
    double chunk_bytes = chunk * static_cast<double>(type_size);
    if ((chunk_bytes < target || std::abs(chunk_bytes - target) / target < 0.5) &&
        chunk_bytes < kChunkMax) {
      break;
    }
    if (chunk == 1) break;
    chunk = std::ceil(chunk / 2.0);
  }
  return static_cast<hsize_t>(chunk);
}

}  // namespace msentity::internal::h5
