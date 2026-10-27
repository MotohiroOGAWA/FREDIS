// mssim-tool: command-line access to msentity-similarity.
//
//   mssim-tool search <query> <reference> <output.mssim> [--threshold T] [--lightweight]
//   mssim-tool by-key <input1> <input2> <output.mssim> [--key1 K] [--key2 K] [--embed]
//   mssim-tool info <input.mssim>                         size, columns, metadata, scores as JSON
//   mssim-tool export <input.mssim> <output.csv|.tsv|.parquet>
//
// search and by-key also take [--method cosine|reverse_cosine] [--bin-width W]
// [--exponent E] [--threads N]. Inputs are any format msentity reads
// (.msp, .mgf, .msds, .tsv, .csv).

#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "msentity/msentity.hpp"
#include "msentity/similarity/similarity.hpp"

namespace {

using nlohmann::ordered_json;
namespace sim = msentity::similarity;

int usage() {
  std::cerr << "usage:\n"
               "  mssim-tool search <query> <reference> <output.mssim> [--threshold T] [--lightweight]\n"
               "  mssim-tool by-key <input1> <input2> <output.mssim> [--key1 K] [--key2 K] [--embed]\n"
               "  mssim-tool info <input.mssim>\n"
               "  mssim-tool export <input.mssim> <output.csv|.tsv|.parquet>\n"
               "options for search and by-key:\n"
               "  --method cosine|reverse_cosine  --bin-width W  --exponent E  --threads N\n";
  return 2;
}

ordered_json info_json(const sim::SimilarityDataset& result) {
  ordered_json out;
  out["size"] = result.size();
  out["columns"] = result.columns();
  out["has_matched_data"] = result.has_matched_data();
  if (result.has_matched_data()) {
    out["matched_sizes"] = {result.matched_datasets()->first.size(),
                            result.matched_datasets()->second.size()};
  }
  out["metadata"] = ordered_json::parse(result.metadata_json());
  if (result.size() > 0) {
    auto s = result.describe_scores();
    out["scores"] = {{"count", s.count}, {"mean", s.mean},     {"q1", s.q1}, {"median", s.median},
                     {"q3", s.q3},       {"min", s.min},       {"max", s.max}};
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) return usage();
  sim::Options options;
  double threshold = 0.8;
  bool lightweight = false;
  bool embed = false;
  std::string key1 = "SpecID";
  std::string key2 = "SpecID";
  std::vector<std::string> positional;

  try {
    for (std::size_t i = 1; i < args.size(); ++i) {
      bool has_value = i + 1 < args.size();
      if (args[i] == "--threshold" && has_value) {
        threshold = std::stod(args[++i]);
      } else if (args[i] == "--method" && has_value) {
        options.method = sim::parse_method(args[++i]);
      } else if (args[i] == "--bin-width" && has_value) {
        options.bin_width = std::stod(args[++i]);
      } else if (args[i] == "--exponent" && has_value) {
        options.intensity_exponent = std::stod(args[++i]);
      } else if (args[i] == "--threads" && has_value) {
        options.threads = std::stoi(args[++i]);
      } else if (args[i] == "--key1" && has_value) {
        key1 = args[++i];
      } else if (args[i] == "--key2" && has_value) {
        key2 = args[++i];
      } else if (args[i] == "--lightweight") {
        lightweight = true;
      } else if (args[i] == "--embed") {
        embed = true;
      } else if (args[i].rfind("--", 0) == 0) {
        return usage();
      } else {
        positional.push_back(args[i]);
      }
    }

    const std::string& command = args[0];
    if (command == "search" && positional.size() == 3) {
      auto query = msentity::io::load_ms_dataset(positional[0]);
      auto reference = msentity::io::load_ms_dataset(positional[1]);
      auto result = sim::calculate_library_search(query, reference, threshold, options,
                                                  !lightweight, positional[0], positional[1]);
      result.save(positional[2]);
      std::cerr << result.size() << " pairs written to " << positional[2] << "\n";
    } else if (command == "by-key" && positional.size() == 3) {
      auto ds1 = msentity::io::load_ms_dataset(positional[0]);
      auto ds2 = msentity::io::load_ms_dataset(positional[1]);
      auto result = sim::calculate_similarity(ds1, ds2, key1, key2, options, embed, positional[0],
                                              positional[1]);
      result.save(positional[2]);
      std::cerr << result.size() << " pairs written to " << positional[2] << "\n";
    } else if (command == "info" && positional.size() == 1) {
      std::cout << info_json(sim::SimilarityDataset::load(positional[0])).dump(2) << "\n";
    } else if (command == "export" && positional.size() == 2) {
      sim::SimilarityDataset::load(positional[0]).export_table(positional[1]);
    } else {
      return usage();
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
