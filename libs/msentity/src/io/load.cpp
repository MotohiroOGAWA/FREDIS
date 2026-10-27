#include "msentity/io/load.hpp"

#include <filesystem>
#include <stdexcept>

#include "internal/text.hpp"
#include "msentity/io/msds.hpp"
#include "msentity/processing/id.hpp"

namespace msentity::io {

MSDataset load_ms_dataset(const std::string& path, const std::optional<std::string>& file_type,
                          const std::optional<std::string>& spec_id_prefix,
                          const ReadProgressCallback& progress_callback,
                          bool infer_types) {
  std::string type;
  if (file_type) {
    type = internal::to_lower_ascii(*file_type);
  } else {
    std::string suffix = internal::to_lower_ascii(std::filesystem::path(path).extension().string());
    if (suffix == ".msp") {
      type = "msp";
    } else if (suffix == ".mgf") {
      type = "mgf";
    } else if (suffix == ".msds" || suffix == ".hdf5" || suffix == ".h5") {
      type = "msds";
    } else if (suffix == ".tsv" || suffix == ".csv") {
      type = suffix.substr(1);
    } else {
      throw std::invalid_argument(
          "Cannot determine file type from extension. Please specify 'file_type' parameter.");
    }
  }

  MSDataset dataset;
  if (type == "msp" || type == "mgf") {
    TextReadOptions options;
    options.spec_id_prefix = spec_id_prefix;
    options.progress_callback = progress_callback;
    dataset = type == "msp" ? read_msp(path, options) : read_mgf(path, options);
  } else if (type == "msds") {
    dataset = read_msds(path);
  } else if (type == "tsv" || type == "csv") {
    DelimitedReadOptions options;
    options.spec_id_prefix = spec_id_prefix;
    options.infer_types = infer_types;
    dataset = type == "tsv" ? read_tsv(path, options) : read_csv(path, options);
  } else {
    throw std::invalid_argument("Unsupported file type. Use 'msp', 'mgf', 'msds', 'tsv' or 'csv'.");
  }

  if (!dataset.has_column("SpecID") && spec_id_prefix) {
    processing::set_spec_id(dataset, "SpecID", *spec_id_prefix);
  }
  return dataset;
}

}  // namespace msentity::io
