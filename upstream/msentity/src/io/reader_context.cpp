#include "io/reader_context.hpp"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "internal/table.hpp"
#include "internal/text.hpp"

namespace msentity::io::detail {

namespace fs = std::filesystem;

ReaderContext::ReaderContext(std::string file_path, std::string file_type_name,
                             const TextReadOptions& options, int64_t file_size)
    : file_path_(std::move(file_path)),
      file_type_name_(std::move(file_type_name)),
      options_(options),
      item_parser_(options.canonicalize_column, options.canonicalize_adduct_type),
      file_size_(file_size) {
  error_file_path_ = resolve_error_log_path(options.error_log_file);
}

std::string ReaderContext::resolve_error_log_path(
    const std::optional<std::string>& error_log_file) const {
  if (!error_log_file) {
    std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d%H%M%S", std::localtime(&now));
    fs::path path(file_path_);
    std::string stem = (path.parent_path() / path.stem()).string();
    if (path.parent_path().empty()) stem = path.stem().string();
    std::string base = stem + "_error_" + stamp;
    std::string candidate = base + ".txt";
    for (int count = 1; fs::exists(candidate); ++count) {
      candidate = base + "_" + std::to_string(count) + ".txt";
    }
    return candidate;
  }
  fs::path directory = fs::path(*error_log_file).parent_path();
  if (!directory.empty() && !fs::exists(directory)) {
    throw std::invalid_argument("Directory '" + directory.string() + "' does not exist.");
  }
  return *error_log_file;
}

void ReaderContext::update(std::string_view line, std::size_t raw_bytes) {
  ++line_count_;
  processed_size_ += static_cast<int64_t>(raw_bytes);
  if (options_.error_context_lines > 0) {
    recent_lines_.emplace_back(line);
    while (recent_lines_.size() > options_.error_context_lines) recent_lines_.pop_front();
  }
  emit_progress(false);
}

void ReaderContext::emit_progress(bool force) {
  if (!options_.progress_callback) return;
  auto now = std::chrono::steady_clock::now();
  if (!force && now - last_progress_ < std::chrono::milliseconds(100)) return;
  last_progress_ = now;
  options_.progress_callback(std::min(processed_size_, file_size_), file_size_, success_count_,
                             record_count_);
}

void ReaderContext::reset_record() {
  meta_.clear();
  peak_mz_.clear();
  peak_intensity_.clear();
  peak_meta_names_.clear();
  peak_meta_.clear();
  error_texts_.clear();
  error_flag_ = false;
  recent_lines_.clear();
}

bool ReaderContext::has_record_content() const { return !meta_.empty() || !peak_mz_.empty(); }

void ReaderContext::set_record() {
  // Spectrum metadata: new columns are back-filled with "".
  for (const auto& [key, value] : meta_) {
    if (col_index_.count(key) == 0) {
      col_index_[key] = col_values_.size();
      col_values_.emplace_back(static_cast<std::size_t>(success_count_), std::string());
    }
  }
  for (const auto& column : col_index_) {
    const std::string& key = column.first;
    auto it = std::find_if(meta_.begin(), meta_.end(),
                           [&](const auto& item) { return item.first == key; });
    col_values_[column.second].push_back(it != meta_.end() ? it->second : std::string());
  }

  // Peaks: new peak metadata columns are back-filled with "".
  for (const auto& name : peak_meta_names_) {
    if (all_peak_meta_.count(name) == 0) {
      all_peak_meta_[name] = std::vector<std::string>(all_mz_.size());
      all_peak_meta_names_.push_back(name);
    }
  }
  const std::size_t length = peak_mz_.size();
  all_mz_.insert(all_mz_.end(), peak_mz_.begin(), peak_mz_.end());
  all_intensity_.insert(all_intensity_.end(), peak_intensity_.begin(), peak_intensity_.end());
  for (auto& [name, values] : all_peak_meta_) {
    auto it = peak_meta_.find(name);
    if (it != peak_meta_.end()) {
      values.insert(values.end(), it->second.begin(), it->second.end());
    } else {
      values.insert(values.end(), length, std::string());
    }
  }
  offsets_.push_back(static_cast<int64_t>(all_mz_.size()));
}

void ReaderContext::cleanup_column_names() {
  all_col_names_.erase(std::remove_if(all_col_names_.begin(), all_col_names_.end(),
                                      [&](const std::string& name) {
                                        return col_index_.count(name) == 0;
                                      }),
                       all_col_names_.end());
  header_map_.erase(
      std::remove_if(header_map_.begin(), header_map_.end(),
                     [&](const auto& item) {
                       return std::find(all_col_names_.begin(), all_col_names_.end(),
                                        item.first) == all_col_names_.end();
                     }),
      header_map_.end());
}

void ReaderContext::update_record() {
  try_write_errors();
  if (has_record_content()) {
    if (!error_flag_) {
      set_record();
      ++success_count_;
    }
    ++record_count_;
  }
  cleanup_column_names();
  reset_record();
}

std::pair<std::string, std::string> ReaderContext::add_meta(std::string_view key,
                                                            std::string_view value) {
  auto parsed = item_parser_.parse_item_pair(key, value);
  std::string parsed_key = std::move(parsed.first);
  std::string parsed_value = std::move(parsed.second);
  auto find = [&](const std::string& name) {
    return std::find_if(meta_.begin(), meta_.end(),
                        [&](const auto& item) { return item.first == name; });
  };
  auto existing = find(parsed_key);
  if (existing != meta_.end()) {
    if (!options_.allow_duplicate_cols) {
      throw std::invalid_argument("Duplicate meta key: (" + std::string(key) + " & " +
                                  existing->second + ") -> " + parsed_key);
    }
    int index = 1;
    while (find(parsed_key + std::to_string(index)) != meta_.end()) ++index;
    parsed_key += std::to_string(index);
  }
  if (col_index_.count(parsed_key) == 0) {
    auto it = std::find_if(header_map_.begin(), header_map_.end(),
                           [&](const auto& item) { return item.first == parsed_key; });
    if (it != header_map_.end()) {
      it->second = std::string(key);
    } else {
      header_map_.emplace_back(parsed_key, std::string(key));
    }
    all_col_names_.push_back(parsed_key);
  }
  meta_.emplace_back(parsed_key, parsed_value);
  return {parsed_key, parsed_value};
}

void ReaderContext::add_peak(const PeakEntry& entry) {
  const std::size_t previous = peak_mz_.size();
  for (const auto& [name, value] : entry.metadata) {
    if (peak_meta_.count(name) == 0) {
      peak_meta_[name] = std::vector<std::string>(previous);
      peak_meta_names_.push_back(name);
    }
  }
  peak_mz_.push_back(entry.mz);
  peak_intensity_.push_back(entry.intensity);
  for (auto& column : peak_meta_) {
    const std::string& name = column.first;
    auto it = std::find_if(entry.metadata.begin(), entry.metadata.end(),
                           [&](const auto& item) { return item.first == name; });
    column.second.push_back(it != entry.metadata.end() ? it->second : std::string());
  }
}

void ReaderContext::add_error_message(std::string_view message, std::string_view line_text) {
  error_flag_ = true;
  std::string m = internal::replace_all(internal::strip(message), "\n", "\\n");
  std::string l = internal::replace_all(internal::strip(line_text), "\n", "\\n");
  char number[32];
  std::snprintf(number, sizeof(number), "%05lld", static_cast<long long>(line_count_));
  error_texts_.push_back("[ERROR] Line (" + std::string(number) + ") " + l + " | " + m);
}

bool ReaderContext::try_write_errors() {
  if (!error_flag_ || options_.error_log_level == ErrorLogLevel::None) return false;
  if (options_.error_log_level == ErrorLogLevel::Detail && !recent_lines_.empty()) {
    error_texts_.push_back("[CONTEXT]");
    error_texts_.insert(error_texts_.end(), recent_lines_.begin(), recent_lines_.end());
  }
  std::ofstream stream(error_file_path_, std::ios::app | std::ios::binary);
  for (std::size_t i = 0; i < error_texts_.size(); ++i) {
    if (i > 0) stream << "\n";
    stream << error_texts_[i];
  }
  stream << "\n\n";
  return true;
}

TextReadResult ReaderContext::finish() {
  update_record();
  processed_size_ = file_size_;
  emit_progress(true);

  const int64_t n_spectra = static_cast<int64_t>(offsets_.size()) - 1;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> columns;
  for (const auto& name : all_col_names_) {
    fields.push_back(arrow::field(name, arrow::utf8()));
    columns.push_back(internal::strings_to_array(col_values_[col_index_.at(name)]));
  }
  auto spectrum_metadata = arrow::Table::Make(arrow::schema(fields), columns, n_spectra);

  std::vector<double> data(all_mz_.size() * 2);
  for (std::size_t i = 0; i < all_mz_.size(); ++i) {
    data[2 * i] = all_mz_[i];
    data[2 * i + 1] = all_intensity_[i];
  }

  std::shared_ptr<arrow::Table> peak_metadata;
  std::optional<std::vector<std::string>> peak_columns;
  if (!all_peak_meta_names_.empty()) {
    std::vector<std::shared_ptr<arrow::Field>> peak_fields;
    std::vector<std::shared_ptr<arrow::Array>> peak_arrays;
    for (const auto& name : all_peak_meta_names_) {
      peak_fields.push_back(arrow::field(name, arrow::utf8()));
      peak_arrays.push_back(internal::strings_to_array(all_peak_meta_.at(name)));
    }
    peak_metadata = arrow::Table::Make(arrow::schema(peak_fields), peak_arrays,
                                       static_cast<int64_t>(all_mz_.size()));
    peak_columns = all_peak_meta_names_;
  }

  PeakSeries peaks(std::move(data), offsets_, peak_metadata, peak_columns);
  if (options_.normalize_intensity) peaks.normalize();
  return {MSDataset(spectrum_metadata, std::move(peaks)), header_map_};
}

// ---------------------------------------------------------------------------

bool is_peak_header_line(std::string_view line) {
  auto items = internal::split_whitespace(internal::strip(line));
  return items.size() >= 2 && internal::to_lower_ascii(items[0]) == "mz" &&
         internal::to_lower_ascii(items[1]) == "intensity";
}

std::vector<std::string> split_peak_metadata_text(std::string_view metadata_text,
                                                  std::string_view line) {
  auto fields = internal::split_char(metadata_text, ';');
  long quote_start = -1;
  long quote_end = -1;
  char quote_char = '\0';
  std::string merged;
  std::vector<std::string> items;

  for (std::size_t i = 0; i < fields.size(); ++i) {
    std::string_view stripped = internal::strip(fields[i]);
    if (!stripped.empty() && stripped.front() == '"' && quote_start == -1) {
      quote_start = static_cast<long>(i);
      quote_char = '"';
    } else if (!stripped.empty() && stripped.front() == '\'' && quote_start == -1) {
      quote_start = static_cast<long>(i);
      quote_char = '\'';
    }
    if (quote_start != -1 && !stripped.empty() && stripped.back() == quote_char) {
      quote_end = static_cast<long>(i);
    }

    if (quote_start != -1 && quote_end != -1) {
      std::string joined;
      for (long k = quote_start; k <= quote_end; ++k) {
        if (k > quote_start) joined += ";";
        joined += fields[static_cast<std::size_t>(k)];
      }
      std::string_view view = internal::strip(joined);
      // str.strip(quote_char)
      std::size_t begin = 0, end = view.size();
      while (begin < end && view[begin] == quote_char) ++begin;
      while (end > begin && view[end - 1] == quote_char) --end;
      merged = std::string(view.substr(begin, end - begin));
      quote_start = -1;
      quote_end = -1;
      quote_char = '\0';
    } else if (quote_start != -1 && quote_end == -1) {
      continue;
    } else {
      merged = std::string(fields[i]);
    }

    if (quote_start == -1) {
      items.emplace_back(internal::strip(merged));
      merged.clear();
    }
  }
  if (!merged.empty()) {
    throw std::invalid_argument("Peak line '" + std::string(internal::strip(line)) +
                                "' has unmatched quotes in metadata.");
  }
  return items;
}

std::string float_conversion_error(std::string_view text) {
  return "could not convert string to float: '" + std::string(text) + "'";
}

PeakEntry parse_peak_line(std::string_view line, const std::vector<std::string>* peak_columns,
                          const std::string& auto_col_prefix) {
  static const std::vector<std::string> default_columns = {"mz", "intensity"};
  const std::vector<std::string>& columns = peak_columns ? *peak_columns : default_columns;

  auto items = internal::split_whitespace(internal::strip(line), 2);
  std::vector<std::string> meta_items;
  if (items.size() == 3) {
    meta_items = split_peak_metadata_text(items[2], line);
  } else if (items.size() != 2) {
    throw std::invalid_argument("Peak line '" + std::string(internal::strip(line)) +
                                "' does not have valid m/z and intensity values.");
  }

  PeakEntry entry;
  auto mz = internal::parse_number(items[0]);
  if (!mz) throw std::invalid_argument(float_conversion_error(items[0]));
  auto intensity = internal::parse_number(items[1]);
  if (!intensity) throw std::invalid_argument(float_conversion_error(items[1]));
  entry.mz = *mz;
  entry.intensity = *intensity;

  for (std::size_t i = 0; i < meta_items.size(); ++i) {
    std::string column = i + 2 >= columns.size()
                             ? auto_col_prefix + std::to_string(i + 3 - columns.size())
                             : columns[i + 2];
    if (meta_items[i].empty()) continue;
    bool duplicate = column == "mz" || column == "intensity" ||
                     std::any_of(entry.metadata.begin(), entry.metadata.end(),
                                 [&](const auto& item) { return item.first == column; });
    if (duplicate) {
      throw std::invalid_argument("Duplicate peak metadata column '" + column + "' in line '" +
                                  std::string(internal::strip(line)) + "'.");
    }
    entry.metadata.emplace_back(column, meta_items[i]);
  }
  return entry;
}

// ---------------------------------------------------------------------------

LineSource LineSource::from_file(const std::string& path) {
  LineSource source;
  source.buffer_ = internal::read_file(path);
  return source;
}

LineSource LineSource::from_text(const std::string& text) {
  LineSource source;
  source.buffer_ = text;
  return source;
}

bool LineSource::next(std::string_view& line, std::size_t& raw_bytes) {
  if (position_ >= buffer_.size()) return false;
  std::size_t start = position_;
  std::size_t end = start;
  while (end < buffer_.size() && buffer_[end] != '\n' && buffer_[end] != '\r') ++end;
  std::size_t next = end;
  if (next < buffer_.size()) {
    if (buffer_[next] == '\r' && next + 1 < buffer_.size() && buffer_[next + 1] == '\n') ++next;
    ++next;
  }
  line = std::string_view(buffer_).substr(start, end - start);
  raw_bytes = next - start;
  position_ = next;
  return true;
}

}  // namespace msentity::io::detail
