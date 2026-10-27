#include "msentity/similarity/calculation.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "internal/table.hpp"

namespace msentity::similarity {

namespace {

// A spectrum as a sparse binned vector: (bin, summed intensity) sorted by bin.
struct BinnedSpectrum {
  std::vector<std::pair<int64_t, double>> bins;
  double norm = 0.0;  // sum of squared bin values
};

void validate(const Options& options) {
  if (!std::isfinite(options.bin_width) || options.bin_width <= 0) {
    throw std::invalid_argument("bin_width must be positive");
  }
  if (!std::isfinite(options.intensity_exponent) || options.intensity_exponent <= 0) {
    throw std::invalid_argument("intensity_exponent must be positive");
  }
  if (options.threads < 0) throw std::invalid_argument("threads must be >= 0");
}

BinnedSpectrum bin_spectrum(const Spectrum& spectrum, const Options& options) {
  BinnedSpectrum result;
  std::vector<std::pair<int64_t, double>> peaks;
  peaks.reserve(static_cast<std::size_t>(spectrum.size()));
  for (int64_t i = 0; i < spectrum.size(); ++i) {
    double intensity = std::max(0.0, spectrum.intensity(i));
    if (options.intensity_exponent != 1.0) {
      intensity = std::pow(intensity, options.intensity_exponent);
    }
    auto bin = static_cast<int64_t>(std::floor(spectrum.mz(i) / options.bin_width));
    peaks.emplace_back(bin, intensity);
  }
  std::stable_sort(peaks.begin(), peaks.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& peak : peaks) {
    if (!result.bins.empty() && result.bins.back().first == peak.first) {
      result.bins.back().second += peak.second;
    } else {
      result.bins.push_back(peak);
    }
  }
  for (const auto& bin : result.bins) result.norm += bin.second * bin.second;
  return result;
}

float score(const BinnedSpectrum& a, const BinnedSpectrum& b, Method method) {
  double dot = 0.0;
  double matched_norm_a = 0.0;
  std::size_t i = 0, j = 0;
  while (i < a.bins.size() && j < b.bins.size()) {
    if (a.bins[i].first < b.bins[j].first) {
      ++i;
    } else if (b.bins[j].first < a.bins[i].first) {
      ++j;
    } else {
      dot += a.bins[i].second * b.bins[j].second;
      matched_norm_a += a.bins[i].second * a.bins[i].second;
      ++i;
      ++j;
    }
  }
  double norm_a = method == Method::ReverseCosine ? matched_norm_a : a.norm;
  double denominator = std::sqrt(norm_a * b.norm);
  double value = denominator > 0 ? dot / denominator : 0.0;
  if (!(value >= 0.0)) value = std::isnan(value) ? value : 0.0;  // keep NaN, clip negatives
  if (value > 1.0) value = 1.0;
  return static_cast<float>(value);
}

std::vector<BinnedSpectrum> bin_all(const MSDataset& dataset, const Options& options) {
  std::vector<BinnedSpectrum> binned(static_cast<std::size_t>(dataset.size()));
  for (int64_t i = 0; i < dataset.size(); ++i) {
    binned[static_cast<std::size_t>(i)] = bin_spectrum(dataset.peaks().spectrum(i), options);
  }
  return binned;
}

unsigned thread_count(const Options& options, int64_t work) {
  unsigned threads = options.threads > 0 ? static_cast<unsigned>(options.threads)
                                         : std::max(1u, std::thread::hardware_concurrency());
  if (work < 2048) threads = 1;  // not worth starting threads
  return threads;
}

// Run body(begin, end) over [0, n) split into contiguous ranges, one per thread.
template <typename Body>
void parallel_ranges(int64_t n, unsigned threads, Body&& body) {
  if (threads <= 1 || n <= 1) {
    body(int64_t{0}, n);
    return;
  }
  std::vector<std::thread> workers;
  int64_t chunk = (n + threads - 1) / threads;
  for (unsigned t = 0; t < threads; ++t) {
    int64_t begin = static_cast<int64_t>(t) * chunk;
    int64_t end = std::min(n, begin + chunk);
    if (begin >= end) break;
    workers.emplace_back([&body, begin, end] { body(begin, end); });
  }
  for (auto& worker : workers) worker.join();
}

void check_positions(const std::vector<int64_t>& index, int64_t size, const char* name) {
  for (int64_t position : index) {
    if (position < 0 || position >= size) {
      throw std::out_of_range(std::string(name) + " contains an invalid position: " +
                              std::to_string(position));
    }
  }
}

// Key of a metadata value for equality: numbers (and booleans) compare by
// value across integer/float types, text by content.
std::optional<std::string> match_key(const Value& value) {
  if (is_null(value)) return std::nullopt;
  if (const auto* d = std::get_if<double>(&value)) {
    if (std::isnan(*d)) return std::nullopt;
    return "n:" + format_float(*d);
  }
  if (const auto* i = std::get_if<int64_t>(&value)) return "n:" + format_float(static_cast<double>(*i));
  if (const auto* b = std::get_if<bool>(&value)) return "n:" + format_float(*b ? 1.0 : 0.0);
  return "s:" + std::get<std::string>(value);
}

std::shared_ptr<arrow::Array> int64_array(const std::vector<int64_t>& values) {
  arrow::Int64Builder builder;
  internal::check(builder.AppendValues(values));
  return internal::unwrap(builder.Finish());
}

std::shared_ptr<arrow::Array> float_array(const std::vector<float>& values) {
  arrow::FloatBuilder builder;
  internal::check(builder.AppendValues(values));
  return internal::unwrap(builder.Finish());
}

}  // namespace

std::string method_name(Method method) {
  return method == Method::Cosine ? "cosine" : "reverse_cosine";
}

Method parse_method(const std::string& name) {
  if (name == "cosine") return Method::Cosine;
  if (name == "reverse_cosine") return Method::ReverseCosine;
  throw std::invalid_argument("method must be 'cosine' or 'reverse_cosine'");
}

std::vector<float> pair_scores(const MSDataset& ds1, const std::vector<int64_t>& index1,
                               const MSDataset& ds2, const std::vector<int64_t>& index2,
                               const Options& options) {
  validate(options);
  if (index1.size() != index2.size()) {
    throw std::invalid_argument("index1 and index2 must have the same length");
  }
  check_positions(index1, ds1.size(), "index1");
  check_positions(index2, ds2.size(), "index2");

  // Bin only the spectra that are used.
  auto bin_used = [&](const MSDataset& dataset, const std::vector<int64_t>& index) {
    std::unordered_map<int64_t, BinnedSpectrum> binned;
    for (int64_t position : index) {
      if (binned.count(position) == 0) {
        binned.emplace(position, bin_spectrum(dataset.peaks().spectrum(position), options));
      }
    }
    return binned;
  };
  auto binned1 = bin_used(ds1, index1);
  auto binned2 = bin_used(ds2, index2);

  const int64_t n = static_cast<int64_t>(index1.size());
  std::vector<float> scores(static_cast<std::size_t>(n));
  parallel_ranges(n, thread_count(options, n), [&](int64_t begin, int64_t end) {
    for (int64_t k = begin; k < end; ++k) {
      auto u = static_cast<std::size_t>(k);
      scores[u] = score(binned1.at(index1[u]), binned2.at(index2[u]), options.method);
    }
  });
  return scores;
}

std::vector<float> all_pairs_matrix(const MSDataset& ds1, const MSDataset& ds2,
                                    const Options& options) {
  validate(options);
  auto binned1 = bin_all(ds1, options);
  auto binned2 = bin_all(ds2, options);
  const int64_t n1 = ds1.size(), n2 = ds2.size();
  std::vector<float> matrix(static_cast<std::size_t>(n1 * n2));
  parallel_ranges(n1, thread_count(options, n1 * n2), [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      for (int64_t j = 0; j < n2; ++j) {
        matrix[static_cast<std::size_t>(i * n2 + j)] =
            score(binned1[static_cast<std::size_t>(i)], binned2[static_cast<std::size_t>(j)],
                  options.method);
      }
    }
  });
  return matrix;
}

std::shared_ptr<arrow::Table> similarity_by_key(const MSDataset& ds1, const MSDataset& ds2,
                                                const std::string& key1,
                                                const std::string& key2,
                                                const Options& options) {
  validate(options);
  if (!ds1.has_column(key1)) throw KeyError("ds1 does not contain key column: " + key1);
  if (!ds2.has_column(key2)) throw KeyError("ds2 does not contain key column: " + key2);
  for (const auto& key : {key1, key2}) {
    if (key == "index1" || key == "index2" || key == "cosine_similarity") {
      throw std::invalid_argument(
          "Key column names must not collide with result columns: index1, index2, "
          "cosine_similarity");
    }
  }
  auto values1 = ds1.column_values(key1);
  auto values2 = ds2.column_values(key2);

  // Non-missing keys must be unique on each side.
  auto unique_keys = [](const std::vector<Value>& values, const std::string& side,
                        const std::string& key) {
    std::unordered_map<std::string, int64_t> positions;
    std::vector<std::string> duplicates;
    for (std::size_t i = 0; i < values.size(); ++i) {
      auto k = match_key(values[i]);
      if (!k) continue;
      auto [it, inserted] = positions.emplace(*k, static_cast<int64_t>(i));
      if (!inserted) {
        std::string shown = to_display_string(values[i]);
        if (std::find(duplicates.begin(), duplicates.end(), shown) == duplicates.end() &&
            duplicates.size() < 5) {
          duplicates.push_back(shown);
        }
      }
    }
    if (!duplicates.empty()) {
      std::string examples;
      for (const auto& d : duplicates) examples += (examples.empty() ? "'" : ", '") + d + "'";
      throw std::invalid_argument(side + " key column '" + key +
                                  "' contains duplicate values; one-to-one matching is "
                                  "ambiguous; examples: [" + examples + "]");
    }
    return positions;
  };
  unique_keys(values1, "ds1", key1);
  auto right = unique_keys(values2, "ds2", key2);

  std::vector<int64_t> index1, index2;
  std::vector<Value> matched1, matched2;
  for (std::size_t i = 0; i < values1.size(); ++i) {
    auto k = match_key(values1[i]);
    if (!k) continue;
    auto it = right.find(*k);
    if (it == right.end()) continue;
    index1.push_back(static_cast<int64_t>(i));
    index2.push_back(it->second);
    matched1.push_back(values1[i]);
    matched2.push_back(values2[static_cast<std::size_t>(it->second)]);
  }
  auto scores = pair_scores(ds1, index1, ds2, index2, options);

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  auto add = [&](const std::string& name, std::shared_ptr<arrow::Array> array) {
    fields.push_back(arrow::field(name, array->type()));
    arrays.push_back(std::move(array));
  };
  add(key1, internal::values_to_array(matched1));
  if (key2 != key1) add(key2, internal::values_to_array(matched2));
  add("index1", int64_array(index1));
  add("index2", int64_array(index2));
  add("cosine_similarity", float_array(scores));
  return arrow::Table::Make(arrow::schema(fields), arrays, static_cast<int64_t>(index1.size()));
}

std::shared_ptr<arrow::Table> library_search(const MSDataset& query, const MSDataset& reference,
                                             double threshold, const Options& options,
                                             const ProgressCallback& progress) {
  validate(options);
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1) {
    throw std::invalid_argument("threshold must be a finite value between 0 and 1");
  }
  const int64_t n_query = query.size(), n_reference = reference.size();
  const int64_t total = n_query * n_reference;
  auto binned_query = bin_all(query, options);
  auto binned_reference = bin_all(reference, options);

  // Hits of each query row, filled in parallel, joined in order.
  std::vector<std::vector<std::pair<int64_t, float>>> hits(static_cast<std::size_t>(n_query));
  const unsigned threads = thread_count(options, total);
  const int64_t batch = std::max<int64_t>(1, 4096 / std::max<int64_t>(1, n_reference) + 1) * threads;
  if (progress) progress(0, total);
  for (int64_t start = 0; start < n_query; start += batch) {
    int64_t stop = std::min(n_query, start + batch);
    parallel_ranges(stop - start, threads, [&](int64_t begin, int64_t end) {
      for (int64_t q = start + begin; q < start + end; ++q) {
        auto& row = hits[static_cast<std::size_t>(q)];
        for (int64_t r = 0; r < n_reference; ++r) {
          float s = score(binned_query[static_cast<std::size_t>(q)],
                          binned_reference[static_cast<std::size_t>(r)], options.method);
          if (s >= threshold) row.emplace_back(r, s);
        }
      }
    });
    if (progress) progress(stop * n_reference, total);
  }

  std::vector<int64_t> index1, index2;
  std::vector<float> scores;
  for (int64_t q = 0; q < n_query; ++q) {
    for (const auto& [r, s] : hits[static_cast<std::size_t>(q)]) {
      index1.push_back(q);
      index2.push_back(r);
      scores.push_back(s);
    }
  }
  return arrow::Table::Make(
      arrow::schema({arrow::field("index1", arrow::int64()), arrow::field("index2", arrow::int64()),
                     arrow::field("cosine_similarity", arrow::float32())}),
      {int64_array(index1), int64_array(index2), float_array(scores)},
      static_cast<int64_t>(scores.size()));
}

}  // namespace msentity::similarity
