#include <stdexcept>

#include "internal/text.hpp"
#include "io/reader_context.hpp"
#include "io/text_writer.hpp"
#include "msentity/io/text_formats.hpp"
#include "msentity/processing/id.hpp"

namespace msentity::io {

namespace {

using detail::LineSource;
using detail::ReaderContext;

void add_peak_line(ReaderContext& reader, std::string_view line,
                   const std::optional<std::vector<std::string>>& peak_columns,
                   const TextReadOptions& options) {
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
}

std::vector<std::string> header_columns(std::string_view line) {
  std::vector<std::string> columns;
  for (auto item : internal::split_whitespace(internal::strip(line))) columns.emplace_back(item);
  return columns;
}

TextReadResult read_mgf_lines(LineSource source, const std::string& source_name,
                              const TextReadOptions& options) {
  ReaderContext reader(source_name, "mgf", options, source.size());
  bool in_ions_block = false;
  bool peak_flag = false;
  std::optional<std::vector<std::string>> peak_columns;

  std::string_view line;
  std::size_t raw_bytes = 0;
  while (source.next(line, raw_bytes)) {
    reader.update(line, raw_bytes);
    try {
      std::string_view stripped = internal::strip(line);
      std::string upper = internal::to_upper_ascii(stripped);
      if (stripped.empty()) continue;

      if (upper == "BEGIN IONS") {
        if (in_ions_block && (reader.has_meta() || reader.has_peaks())) reader.update_record();
        in_ions_block = true;
        peak_flag = false;
        peak_columns.reset();
        continue;
      }
      if (upper == "END IONS") {
        if (in_ions_block) reader.update_record();
        in_ions_block = false;
        peak_flag = false;
        peak_columns.reset();
        continue;
      }
      if (!in_ions_block) continue;

      if (peak_flag) {
        if (!peak_columns && detail::is_peak_header_line(line)) {
          peak_columns = header_columns(line);
          continue;
        }
        add_peak_line(reader, line, peak_columns, options);
        continue;
      }

      std::size_t equals = line.find('=');
      if (equals != std::string_view::npos) {
        reader.add_meta(line.substr(0, equals), line.substr(equals + 1));
        continue;
      }

      // First line of the peak section.
      peak_flag = true;
      if (detail::is_peak_header_line(line)) {
        peak_columns = header_columns(line);
        continue;
      }
      peak_columns.reset();
      add_peak_line(reader, line, peak_columns, options);
    } catch (const std::exception& error) {
      reader.add_error_message(error.what(), line);
    }
  }
  if (in_ions_block && (reader.has_meta() || reader.has_peaks())) reader.update_record();

  TextReadResult result = reader.finish();
  if (options.spec_id_prefix) {
    processing::set_spec_id(result.dataset, "SpecID", *options.spec_id_prefix);
  }
  return result;
}

}  // namespace

TextReadResult read_mgf_with_header_map(const std::string& path,
                                        const TextReadOptions& options) {
  return read_mgf_lines(LineSource::from_file(path), path, options);
}

MSDataset read_mgf(const std::string& path, const TextReadOptions& options) {
  return read_mgf_with_header_map(path, options).dataset;
}

TextReadResult read_mgf_text_with_header_map(const std::string& text,
                                             const TextReadOptions& options,
                                             const std::string& source_name) {
  if (internal::strip(text).empty()) throw std::invalid_argument("MGF text is empty.");
  return read_mgf_lines(LineSource::from_text(text), source_name, options);
}

MSDataset read_mgf_text(const std::string& text, const TextReadOptions& options,
                        const std::string& source_name) {
  return read_mgf_text_with_header_map(text, options, source_name).dataset;
}

void write_mgf(const MSDataset& dataset, const std::string& path,
               const TextWriteOptions& options) {
  detail::write_text_format(dataset, path, options, detail::TextFormat::Mgf);
}

}  // namespace msentity::io
