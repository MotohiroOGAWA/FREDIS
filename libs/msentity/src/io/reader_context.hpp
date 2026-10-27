#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io/item_parser.hpp"
#include "msentity/io/text_formats.hpp"

namespace msentity::io::detail {

// Port of msentity.io.IOContext.ReaderContext: accumulates records of a
// line-based spectrum file (MSP/MGF) column by column.
class ReaderContext {
 public:
  ReaderContext(std::string file_path, std::string file_type_name,
                const TextReadOptions& options, int64_t file_size);

  // Per-line bookkeeping (line count, progress, error context).
  void update(std::string_view line, std::size_t raw_bytes);
  // Finish the current record (stored only when it had no errors).
  void update_record();
  // Returns (canonical key, value).
  std::pair<std::string, std::string> add_meta(std::string_view key, std::string_view value);
  void add_peak(const PeakEntry& entry);
  void add_error_message(std::string_view message, std::string_view line_text);

  bool has_meta() const { return !meta_.empty(); }
  bool has_peaks() const { return !peak_mz_.empty(); }

  TextReadResult finish();

 private:
  void reset_record();
  bool has_record_content() const;
  void set_record();
  void cleanup_column_names();
  bool try_write_errors();
  void emit_progress(bool force);
  std::string resolve_error_log_path(const std::optional<std::string>& error_log_file) const;

  std::string file_path_;
  std::string file_type_name_;
  TextReadOptions options_;
  ItemParser item_parser_;
  int64_t file_size_;
  int64_t processed_size_ = 0;
  std::chrono::steady_clock::time_point last_progress_{};
  std::string error_file_path_;

  // Accumulated spectrum metadata: column name -> values.
  std::vector<std::string> all_col_names_;
  std::unordered_map<std::string, std::size_t> col_index_;
  std::vector<std::vector<std::string>> col_values_;
  HeaderMap header_map_;

  // Accumulated peaks.
  std::vector<double> all_mz_;
  std::vector<double> all_intensity_;
  std::vector<std::string> all_peak_meta_names_;
  std::unordered_map<std::string, std::vector<std::string>> all_peak_meta_;
  std::vector<int64_t> offsets_{0};

  int64_t record_count_ = 0;
  int64_t success_count_ = 0;
  int64_t line_count_ = 0;

  // Current record.
  std::vector<std::pair<std::string, std::string>> meta_;
  std::vector<double> peak_mz_;
  std::vector<double> peak_intensity_;
  std::vector<std::string> peak_meta_names_;
  std::unordered_map<std::string, std::vector<std::string>> peak_meta_;
  std::vector<std::string> error_texts_;
  bool error_flag_ = false;
  std::deque<std::string> recent_lines_;
};

// Shared peak-line helpers (IOContext.is_peak_header_line / parse_peak_line).
bool is_peak_header_line(std::string_view line);
std::vector<std::string> split_peak_metadata_text(std::string_view metadata_text,
                                                  std::string_view line);
PeakEntry parse_peak_line(std::string_view line, const std::vector<std::string>* peak_columns,
                          const std::string& auto_col_prefix);
std::string float_conversion_error(std::string_view text);

// Iterates over the lines of a file or string. Lines are returned without
// their terminator; raw_bytes includes it (for progress reporting).
class LineSource {
 public:
  static LineSource from_file(const std::string& path);
  static LineSource from_text(const std::string& text);
  bool next(std::string_view& line, std::size_t& raw_bytes);
  int64_t size() const { return static_cast<int64_t>(buffer_.size()); }

 private:
  std::string buffer_;
  std::size_t position_ = 0;
};

}  // namespace msentity::io::detail
