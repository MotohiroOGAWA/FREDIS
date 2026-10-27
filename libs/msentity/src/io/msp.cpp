#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "internal/table.hpp"
#include "internal/text.hpp"
#include "io/reader_context.hpp"
#include "io/text_writer.hpp"
#include "msentity/io/text_formats.hpp"
#include "msentity/processing/id.hpp"

namespace msentity::io {

namespace {

using detail::LineSource;
using detail::ReaderContext;

TextReadResult read_msp_lines(LineSource source, const std::string& source_name,
                              const TextReadOptions& options) {
  ReaderContext reader(source_name, "msp", options, source.size());
  bool peak_flag = false;
  std::optional<std::vector<std::string>> peak_columns;

  std::string_view line;
  std::size_t raw_bytes = 0;
  while (source.next(line, raw_bytes)) {
    reader.update(line, raw_bytes);
    try {
      std::string_view stripped = internal::strip(line);
      if (!peak_flag && stripped.empty()) continue;
      if (peak_flag && stripped.empty()) {
        reader.update_record();
        peak_flag = false;
        peak_columns.reset();
        continue;
      }
      if (peak_flag) {
        if (!peak_columns && detail::is_peak_header_line(line)) {
          std::vector<std::string> columns;
          for (auto item : internal::split_whitespace(internal::strip(line))) {
            columns.emplace_back(item);
          }
          peak_columns = std::move(columns);
          continue;
        }
        try {
          PeakEntry entry =
              options.peak_parser
                  ? options.peak_parser(std::string(line))
                  : detail::parse_peak_line(line, peak_columns ? &*peak_columns : nullptr,
                                            options.auto_peak_col_prefix);
          reader.add_peak(entry);
        } catch (const std::exception& error) {
          reader.add_error_message(error.what(), line);
        }
        continue;
      }

      std::size_t colon = line.find(':');
      if (colon == std::string_view::npos) {
        throw std::invalid_argument("Metadata line does not contain ':': " +
                                    std::string(internal::strip(line)));
      }
      auto [parsed_key, value] = reader.add_meta(line.substr(0, colon), line.substr(colon + 1));
      if (parsed_key == "NumPeaks") {
        peak_flag = true;
        peak_columns.reset();
      }
    } catch (const std::exception& error) {
      reader.add_error_message(error.what(), line);
    }
  }
  if (peak_flag || reader.has_meta() || reader.has_peaks()) reader.update_record();

  TextReadResult result = reader.finish();
  if (options.spec_id_prefix) {
    processing::set_spec_id(result.dataset, "SpecID", *options.spec_id_prefix);
  }
  return result;
}

}  // namespace

TextReadResult read_msp_with_header_map(const std::string& path,
                                        const TextReadOptions& options) {
  return read_msp_lines(LineSource::from_file(path), path, options);
}

MSDataset read_msp(const std::string& path, const TextReadOptions& options) {
  return read_msp_with_header_map(path, options).dataset;
}

TextReadResult read_msp_text_with_header_map(const std::string& text,
                                             const TextReadOptions& options,
                                             const std::string& source_name) {
  if (internal::strip(text).empty()) throw std::invalid_argument("MSP text is empty.");
  return read_msp_lines(LineSource::from_text(text), source_name, options);
}

MSDataset read_msp_text(const std::string& text, const TextReadOptions& options,
                        const std::string& source_name) {
  return read_msp_text_with_header_map(text, options, source_name).dataset;
}

void write_msp(const MSDataset& dataset, const std::string& path,
               const TextWriteOptions& options) {
  detail::write_text_format(dataset, path, options, detail::TextFormat::Msp);
}

// ---------------------------------------------------------------------------

namespace detail {

namespace {

std::string quote_peak_meta_item(const Value& value) {
  return internal::replace_all(to_display_string(value), "\"", "'");
}

const std::string& header_name(const HeaderMap& map, const std::string& column) {
  for (const auto& [key, value] : map) {
    if (key == column) return value;
  }
  return column;
}

}  // namespace

void write_text_format(const MSDataset& dataset, const std::string& path,
                       const TextWriteOptions& options, TextFormat format) {
  const auto& columns = dataset.columns();
  std::vector<std::string> headers = options.headers.value_or(columns);
  std::vector<std::string> valid_headers;
  for (const auto& column : headers) {
    if (std::find(columns.begin(), columns.end(), column) == columns.end()) continue;
    if (column == "NumPeaks") continue;
    valid_headers.push_back(column);
  }

  const auto& peaks = dataset.peaks();
  std::shared_ptr<arrow::Table> peak_metadata = peaks.metadata();
  std::vector<std::string> valid_peak_headers;
  if (peak_metadata) {
    std::vector<std::string> requested =
        options.peak_headers.value_or(internal::column_names(*peak_metadata));
    for (const auto& column : requested) {
      if (internal::column_index(*peak_metadata, column) < 0) continue;
      if (column == "mz" || column == "intensity") continue;
      valid_peak_headers.push_back(column);
    }
  }

  // Visible spectrum metadata and per-peak metadata columns, materialized once.
  auto metadata = internal::select_columns(*dataset.metadata(), valid_headers);
  std::vector<std::shared_ptr<arrow::Array>> header_arrays;
  for (int c = 0; c < metadata->num_columns(); ++c) {
    header_arrays.push_back(metadata->column(c)->chunk(0));
  }
  std::vector<std::shared_ptr<arrow::Array>> peak_arrays;
  for (const auto& column : valid_peak_headers) {
    peak_arrays.push_back(
        peak_metadata->column(internal::column_index(*peak_metadata, column))->chunk(0));
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open file for writing: " + path);
  const int64_t total = dataset.size();
  if (options.progress_callback) options.progress_callback(0, total);

  const std::string separator = format == TextFormat::Msp ? ": " : "=";
  int64_t peak_cursor = 0;  // row in the visible peak metadata table
  for (int64_t row = 0; row < total; ++row) {
    std::string record;
    if (format == TextFormat::Mgf) record += "BEGIN IONS\n";
    for (std::size_t h = 0; h < valid_headers.size(); ++h) {
      record += header_name(options.header_map, valid_headers[h]) + separator +
                to_display_string(internal::value_at(*header_arrays[h], row)) + "\n";
    }

    Spectrum spectrum = peaks.spectrum(row);
    const int64_t n_peaks = spectrum.size();
    if (format == TextFormat::Msp) record += "NumPeaks: " + std::to_string(n_peaks) + "\n";

    std::vector<std::vector<std::string>> meta_items(static_cast<std::size_t>(n_peaks));
    std::vector<bool> used(valid_peak_headers.size(), false);
    for (int64_t p = 0; p < n_peaks; ++p) {
      for (std::size_t c = 0; c < peak_arrays.size(); ++c) {
        std::string item = quote_peak_meta_item(internal::value_at(*peak_arrays[c], peak_cursor + p));
        if (!item.empty()) used[c] = true;
        meta_items[static_cast<std::size_t>(p)].push_back(std::move(item));
      }
    }
    peak_cursor += n_peaks;

    std::vector<std::size_t> used_indices;
    for (std::size_t c = 0; c < used.size(); ++c) {
      if (used[c]) used_indices.push_back(c);
    }
    if (!used_indices.empty()) {
      record += "mz" + options.delimiter + "intensity";
      for (std::size_t c : used_indices) record += options.delimiter + valid_peak_headers[c];
      record += "\n";
    }
    for (int64_t p = 0; p < n_peaks; ++p) {
      std::string line = format_float(spectrum.mz(p)) + options.delimiter +
                         format_float(spectrum.intensity(p));
      bool any = false;
      std::string meta_text;
      for (std::size_t k = 0; k < used_indices.size(); ++k) {
        const std::string& item = meta_items[static_cast<std::size_t>(p)][used_indices[k]];
        if (!item.empty()) any = true;
        if (k > 0) meta_text += "\" ; \"";
        meta_text += item;
      }
      if (any) line += options.delimiter + "\"" + meta_text + "\"";
      record += line + "\n";
    }
    record += format == TextFormat::Msp ? "\n" : "END IONS\n\n";
    out << record;
    if (options.progress_callback) options.progress_callback(row + 1, total);
  }
  if (!out) throw std::runtime_error("failed to write " + path);
}

}  // namespace detail

}  // namespace msentity::io
