#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "msentity/core/ms_dataset.hpp"

namespace msentity::io {

enum class ErrorLogLevel {
  None = 0,    // Do not write an error log.
  Basic = 1,   // Line number and error message.
  Detail = 2,  // Basic plus the recent lines that led to the error.
};

// (processed_bytes, total_bytes, successful_records, total_records)
using ReadProgressCallback = std::function<void(int64_t, int64_t, int64_t, int64_t)>;
// (processed_records, total_records)
using WriteProgressCallback = std::function<void(int64_t, int64_t)>;

// Ordered (key, value) pairs. For readers: canonical column -> original
// header. For writers: column -> header written to the file.
using HeaderMap = std::vector<std::pair<std::string, std::string>>;

// One parsed peak line.
struct PeakEntry {
  double mz = 0.0;
  double intensity = 0.0;
  // Peak metadata (column, value); empty values are omitted.
  std::vector<std::pair<std::string, std::string>> metadata;
};
using PeakParser = std::function<PeakEntry(const std::string& line)>;

struct TextReadOptions {
  // Canonicalize header names ("PRECURSORMZ" -> "PrecursorMZ").
  bool canonicalize_column = true;
  // Canonicalize adduct values ("[M+H]1+" -> "[M+H]+").
  bool canonicalize_adduct_type = true;
  // Scale each spectrum so that its base peak has intensity 1.0.
  bool normalize_intensity = true;
  // Rename duplicate keys within a record (Name, Name1, ...) instead of
  // rejecting the record.
  bool allow_duplicate_cols = false;
  // When set, add a sequential SpecID column with this prefix.
  std::optional<std::string> spec_id_prefix;
  // Prefix for automatically named peak metadata columns (column1, ...).
  std::string auto_peak_col_prefix = "column";
  ErrorLogLevel error_log_level = ErrorLogLevel::None;
  // Error log path; default "<input stem>_error_<timestamp>.txt".
  std::optional<std::string> error_log_file;
  std::size_t error_context_lines = 10;
  // Custom parser for peak lines (replaces the default whitespace parser).
  PeakParser peak_parser;
  ReadProgressCallback progress_callback;
};

struct TextReadResult {
  MSDataset dataset;
  HeaderMap header_map;
};

struct TextWriteOptions {
  // Spectrum metadata columns to write (default: all visible columns).
  // "NumPeaks" is always generated from the peaks and never copied.
  std::optional<std::vector<std::string>> headers;
  // Column -> header name used in the output.
  HeaderMap header_map;
  // Peak metadata columns to write (default: all visible peak columns).
  std::optional<std::vector<std::string>> peak_headers;
  // Separator between m/z, intensity, and peak metadata.
  std::string delimiter = "\t";
  WriteProgressCallback progress_callback;
};

// ---- MSP ----------------------------------------------------------------
MSDataset read_msp(const std::string& path, const TextReadOptions& options = {});
TextReadResult read_msp_with_header_map(const std::string& path,
                                        const TextReadOptions& options = {});
MSDataset read_msp_text(const std::string& text, const TextReadOptions& options = {},
                        const std::string& source_name = "<msp_text>");
TextReadResult read_msp_text_with_header_map(const std::string& text,
                                             const TextReadOptions& options = {},
                                             const std::string& source_name = "<msp_text>");
void write_msp(const MSDataset& dataset, const std::string& path,
               const TextWriteOptions& options = {});

// ---- MGF ----------------------------------------------------------------
MSDataset read_mgf(const std::string& path, const TextReadOptions& options = {});
TextReadResult read_mgf_with_header_map(const std::string& path,
                                        const TextReadOptions& options = {});
MSDataset read_mgf_text(const std::string& text, const TextReadOptions& options = {},
                        const std::string& source_name = "<mgf_text>");
TextReadResult read_mgf_text_with_header_map(const std::string& text,
                                             const TextReadOptions& options = {},
                                             const std::string& source_name = "<mgf_text>");
void write_mgf(const MSDataset& dataset, const std::string& path,
               const TextWriteOptions& options = {});

// ---- TSV / CSV spectrum tables ----------------------------------------------
// One spectrum per row; a final "Peak" column holds "mz,intensity;..." pairs.
struct DelimitedReadOptions {
  std::optional<std::string> spec_id_prefix;
  // false (default): every metadata column is read as text exactly as
  // written; empty cells are "".
  // true: infer column types (int64 / float64 when all
  // non-empty cells are numbers, empty cells become null).
  bool infer_types = false;
};
struct DelimitedWriteOptions {
  // Spectrum metadata columns to write (default: all visible columns).
  std::optional<std::vector<std::string>> headers;
  // Peak metadata columns to write in the Peak cell (default: all visible
  // peak columns; an empty list writes plain "mz,intensity" pairs only).
  std::optional<std::vector<std::string>> peak_headers;
  WriteProgressCallback progress_callback;
};

MSDataset read_tsv(const std::string& path, const DelimitedReadOptions& options = {});
MSDataset read_tsv_text(const std::string& text, const DelimitedReadOptions& options = {},
                        const std::string& source_name = "<tsv_text>");
void write_tsv(const MSDataset& dataset, const std::string& path,
               const DelimitedWriteOptions& options = {});
MSDataset read_csv(const std::string& path, const DelimitedReadOptions& options = {});
MSDataset read_csv_text(const std::string& text, const DelimitedReadOptions& options = {},
                        const std::string& source_name = "<csv_text>");
void write_csv(const MSDataset& dataset, const std::string& path,
               const DelimitedWriteOptions& options = {});

}  // namespace msentity::io
