// msentity-tool: command-line access to the C++ library.
//
//   msentity-tool dump <input> [--type T] [--spec-id-prefix P] [--infer-types]
//                                                   dataset as JSON
//   msentity-tool convert <input> <output> [--type T] [--no-save-view] [--infer-types]
//                         [--no-peak-metadata]
//   msentity-tool meta <input.msds>
//
// "dump" prints every value of a dataset, which makes it easy to compare files.

#include <arrow/api.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "msentity/msentity.hpp"

namespace {

using nlohmann::ordered_json;

ordered_json value_json(const msentity::Value& value) {
  struct Visitor {
    ordered_json operator()(std::monostate) const { return nullptr; }
    ordered_json operator()(bool v) const { return v; }
    ordered_json operator()(int64_t v) const { return v; }
    ordered_json operator()(double v) const {
      if (std::isnan(v)) return "NaN";
      if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
      return v;
    }
    ordered_json operator()(const std::string& v) const { return v; }
  };
  return std::visit(Visitor{}, value);
}

ordered_json table_json(const std::shared_ptr<arrow::Table>& table) {
  ordered_json columns = ordered_json::object();
  if (!table) return nullptr;
  for (int c = 0; c < table->num_columns(); ++c) {
    ordered_json values = ordered_json::array();
    for (const auto& chunk : table->column(c)->chunks()) {
      for (int64_t i = 0; i < chunk->length(); ++i) {
        msentity::Value value;
        if (chunk->IsNull(i)) {
          value = std::monostate{};
        } else {
          auto scalar = chunk->GetScalar(i).ValueOrDie();
          switch (chunk->type_id()) {
            case arrow::Type::BOOL:
              value = std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value;
              break;
            case arrow::Type::INT8: case arrow::Type::INT16: case arrow::Type::INT32:
            case arrow::Type::INT64: case arrow::Type::UINT8: case arrow::Type::UINT16:
            case arrow::Type::UINT32: case arrow::Type::UINT64:
              value = static_cast<int64_t>(std::stoll(scalar->ToString()));
              break;
            case arrow::Type::FLOAT:
              value = static_cast<double>(std::static_pointer_cast<arrow::FloatScalar>(scalar)->value);
              break;
            case arrow::Type::DOUBLE:
              value = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
              break;
            default:
              value = scalar->ToString();
          }
        }
        values.push_back(value_json(value));
      }
    }
    columns[table->schema()->field(c)->name()] = values;
  }
  return columns;
}

ordered_json dataset_json(const msentity::MSDataset& dataset) {
  ordered_json out;
  out["n_spectra"] = dataset.size();
  out["columns"] = dataset.columns();
  out["description"] = dataset.description();
  ordered_json attributes = ordered_json::array();
  for (const auto& [key, value] : dataset.attributes()) attributes.push_back({key, value});
  out["attributes"] = attributes;
  out["tags"] = dataset.tags();
  out["metadata"] = table_json(dataset.metadata());

  const auto& peaks = dataset.peaks();
  out["offsets"] = peaks.offsets();
  ordered_json mz = ordered_json::array();
  ordered_json intensity = ordered_json::array();
  for (double v : peaks.mz()) mz.push_back(value_json(v));
  for (double v : peaks.intensity()) intensity.push_back(value_json(v));
  out["mz"] = mz;
  out["intensity"] = intensity;
  out["peak_metadata_columns"] = peaks.metadata_columns();
  out["peak_metadata"] = table_json(peaks.metadata());
  return out;
}

std::string extension(const std::string& path) {
  std::string ext = std::filesystem::path(path).extension().string();
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext;
}

void write_dataset(const msentity::MSDataset& dataset, const std::string& path, bool save_view,
                   bool peak_metadata) {
  msentity::io::TextWriteOptions text;
  msentity::io::DelimitedWriteOptions delimited;
  if (!peak_metadata) {
    text.peak_headers = std::vector<std::string>{};
    delimited.peak_headers = std::vector<std::string>{};
  }
  std::string ext = extension(path);
  if (ext == ".msds" || ext == ".h5" || ext == ".hdf5") {
    msentity::io::MsdsWriteOptions options;
    options.save_view = save_view;
    msentity::io::write_msds(dataset, path, options);
  } else if (ext == ".msp") {
    msentity::io::write_msp(dataset, path, text);
  } else if (ext == ".mgf") {
    msentity::io::write_mgf(dataset, path, text);
  } else if (ext == ".tsv") {
    msentity::io::write_tsv(dataset, path, delimited);
  } else if (ext == ".csv") {
    msentity::io::write_csv(dataset, path, delimited);
  } else {
    throw std::invalid_argument("unsupported output extension: " + ext);
  }
}

int usage() {
  std::cerr << "usage:\n"
               "  msentity-tool dump <input> [--type T] [--spec-id-prefix P] [--infer-types]\n"

               "  msentity-tool convert <input> <output> [--type T] [--no-save-view] [--infer-types]\n"
               "                       [--no-peak-metadata]\n"

               "  msentity-tool meta <input.msds>\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) return usage();
  std::optional<std::string> type;
  std::optional<std::string> spec_id_prefix;
  bool save_view = true;
  bool infer_types = false;
  bool peak_metadata = true;
  std::vector<std::string> positional;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--type" && i + 1 < args.size()) {
      type = args[++i];
    } else if (args[i] == "--spec-id-prefix" && i + 1 < args.size()) {
      spec_id_prefix = args[++i];
    } else if (args[i] == "--no-peak-metadata") {
      peak_metadata = false;
    } else if (args[i] == "--infer-types") {
      infer_types = true;
    } else if (args[i] == "--no-save-view") {
      save_view = false;
    } else if (args[i] == "--save-view") {
      save_view = true;
    } else {
      positional.push_back(args[i]);
    }
  }

  try {
    const std::string& command = args[0];
    if (command == "dump" && positional.size() == 1) {
      auto dataset = msentity::io::load_ms_dataset(positional[0], type, spec_id_prefix, {}, infer_types);
      std::cout << dataset_json(dataset).dump() << "\n";
    } else if (command == "convert" && positional.size() == 2) {
      auto dataset = msentity::io::load_ms_dataset(positional[0], type, spec_id_prefix, {}, infer_types);
      write_dataset(dataset, positional[1], save_view, peak_metadata);
    } else if (command == "meta" && positional.size() == 1) {
      auto meta = msentity::io::read_msds_meta(positional[0]);
      ordered_json out;
      out["description"] = meta.description;
      ordered_json attributes = ordered_json::array();
      for (const auto& [key, value] : meta.attributes) attributes.push_back({key, value});
      out["attributes"] = attributes;
      out["tags"] = meta.tags;
      std::cout << out.dump() << "\n";
    } else {
      return usage();
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
