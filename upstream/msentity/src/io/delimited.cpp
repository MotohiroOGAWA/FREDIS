#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "internal/table.hpp"
#include "internal/text.hpp"
#include "msentity/io/text_formats.hpp"
#include "msentity/processing/id.hpp"

namespace msentity::io {

namespace {

const std::string kPeakColumn = "Peak";

// CSV parsing as specified in docs/format/text-formats.md §4.3, over text
// whose newlines are already normalized to "\n". Blank lines yield empty
// rows, which csv.DictReader skips.
std::vector<std::vector<std::string>> parse_csv(const std::string& text, char delimiter) {
  enum class State { StartField, Unquoted, Quoted, AfterQuote };
  std::vector<std::vector<std::string>> rows;
  std::vector<std::string> row;
  std::string field;
  State state = State::StartField;
  bool row_has_content = false;

  auto end_field = [&]() {
    row.push_back(std::move(field));
    field.clear();
    state = State::StartField;
  };
  auto end_row = [&]() {
    if (row_has_content) end_field();
    rows.push_back(std::move(row));
    row.clear();
    row_has_content = false;
    state = State::StartField;
  };

  for (std::size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    switch (state) {
      case State::StartField:
        if (c == '\n') {
          end_row();
        } else if (c == '"') {
          state = State::Quoted;
          row_has_content = true;
        } else if (c == delimiter) {
          row_has_content = true;
          end_field();
        } else {
          field.push_back(c);
          state = State::Unquoted;
          row_has_content = true;
        }
        break;
      case State::Unquoted:
        if (c == delimiter) {
          end_field();
        } else if (c == '\n') {
          end_row();
        } else {
          field.push_back(c);
        }
        break;
      case State::Quoted:
        if (c == '"') {
          if (i + 1 < text.size() && text[i + 1] == '"') {
            field.push_back('"');
            ++i;
          } else {
            state = State::AfterQuote;
          }
        } else {
          field.push_back(c);
        }
        break;
      case State::AfterQuote:
        if (c == delimiter) {
          end_field();
        } else if (c == '\n') {
          end_row();
        } else {
          field.push_back(c);  // non-strict: characters after a closing quote
        }
        break;
    }
  }
  if (row_has_content || !row.empty()) end_row();
  return rows;
}

// ---- Peak cell -------------------------------------------------------------
//
//   plain:          91.0542,245;125.0233,1000
//   with metadata:  mz,intensity,formula;91.0542,245,C7H7+;125.0233,1000,"a;b"
//
// Entries are separated by ";" and fields by ","; a field may be enclosed in
// double quotes (with "" for a literal quote) to contain "," ";" or '"'. A
// first entry whose first two fields are "mz" and "intensity" (any case)
// names the peak metadata fields that follow in every entry.

struct PeakField {
  std::string text;
  bool quoted = false;
};

// Split a Peak cell into entries of fields, honoring quotes.
std::vector<std::vector<PeakField>> split_peak_cell(const std::string& cell,
                                                    const std::string& where) {
  std::vector<std::vector<PeakField>> entries(1);
  entries.back().emplace_back();
  std::size_t i = 0;
  while (i < cell.size()) {
    PeakField& field = entries.back().back();
    char c = cell[i];
    if (c == '"' && internal::strip(field.text).empty() && !field.quoted) {
      // Quoted field: read to the closing quote.
      field.text.clear();
      field.quoted = true;
      ++i;
      bool closed = false;
      while (i < cell.size()) {
        if (cell[i] == '"') {
          if (i + 1 < cell.size() && cell[i + 1] == '"') {
            field.text.push_back('"');
            i += 2;
            continue;
          }
          ++i;
          closed = true;
          break;
        }
        field.text.push_back(cell[i++]);
      }
      if (!closed) throw std::invalid_argument("Unclosed quote in Peak value at " + where);
      // Only whitespace may follow before the next separator.
      while (i < cell.size() && internal::is_space(cell[i])) ++i;
      if (i < cell.size() && cell[i] != ',' && cell[i] != ';') {
        throw std::invalid_argument("Unexpected text after a quoted Peak field at " + where);
      }
      continue;
    }
    if (c == ',') {
      entries.back().emplace_back();
    } else if (c == ';') {
      entries.emplace_back();
      entries.back().emplace_back();
    } else {
      field.text.push_back(c);
    }
    ++i;
  }
  for (auto& entry : entries) {
    for (auto& field : entry) {
      if (!field.quoted) field.text = std::string(internal::strip(field.text));
    }
  }
  return entries;
}

struct PeakCell {
  std::vector<std::pair<double, double>> peaks;
  std::vector<std::string> columns;               // peak metadata column names
  std::vector<std::vector<std::string>> values;   // per peak, per column
};

PeakCell parse_peak_cell(const std::string& cell, int64_t row_number,
                         const std::string& format_name) {
  PeakCell result;
  if (internal::strip(cell).empty()) return result;
  const std::string row_where = format_name + " row " + std::to_string(row_number);
  auto entries = split_peak_cell(cell, row_where);
  bool first_entry = true;
  for (std::size_t k = 0; k < entries.size(); ++k) {
    const auto& fields = entries[k];
    if (fields.size() == 1 && fields[0].text.empty() && !fields[0].quoted) continue;  // empty
    std::string where = row_where + ", peak " + std::to_string(k + 1);
    if (first_entry && fields.size() >= 2 && !fields[0].quoted && !fields[1].quoted &&
        internal::to_lower_ascii(fields[0].text) == "mz" &&
        internal::to_lower_ascii(fields[1].text) == "intensity") {
      for (std::size_t f = 2; f < fields.size(); ++f) {
        const std::string& name = fields[f].text;
        if (name.empty() || name == "mz" || name == "intensity" ||
            std::find(result.columns.begin(), result.columns.end(), name) !=
                result.columns.end()) {
          throw std::invalid_argument("Invalid peak column name '" + name + "' at " + where);
        }
        result.columns.push_back(name);
      }
      first_entry = false;
      continue;
    }
    first_entry = false;
    std::size_t max_fields = 2 + result.columns.size();
    if (fields.size() < 2 || fields.size() > max_fields) {
      throw std::invalid_argument(
          "Invalid Peak value at " + where +
          (result.columns.empty()
               ? std::string(": expected 'mz,intensity'")
               : ": expected at most " + std::to_string(max_fields) + " fields"));
    }
    auto mz = internal::parse_number(fields[0].text);
    auto intensity = internal::parse_number(fields[1].text);
    if (!mz || !intensity || fields[0].quoted || fields[1].quoted) {
      throw std::invalid_argument("Invalid numeric Peak value at " + where + ": " +
                                  fields[0].text + "," + fields[1].text);
    }
    result.peaks.emplace_back(*mz, *intensity);
    std::vector<std::string> values(result.columns.size());
    for (std::size_t f = 2; f < fields.size(); ++f) values[f - 2] = fields[f].text;
    result.values.push_back(std::move(values));
  }
  return result;
}

// A peak-cell field written with quotes when it contains a separator, a
// quote, or surrounding whitespace.
std::string peak_cell_field(const std::string& text) {
  bool needs_quotes = text.find_first_of(",;\"") != std::string::npos ||
                      (!text.empty() && (internal::is_space(text.front()) ||
                                         internal::is_space(text.back())));
  if (!needs_quotes) return text;
  return "\"" + internal::replace_all(text, "\"", "\"\"") + "\"";
}

// _infer_metadata: a column whose non-empty cells are all numeric becomes
// int64 (all integers, no empty cells) or float64; otherwise a string column
// with empty cells as null. An all-empty column becomes an all-null float64.
std::shared_ptr<arrow::Array> infer_column(const std::vector<std::string>& cells) {
  bool numeric = true;
  bool all_int = true;
  bool any_empty = false;
  std::vector<double> doubles(cells.size(), 0.0);
  std::vector<int64_t> ints(cells.size(), 0);
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (cells[i].empty()) {
      any_empty = true;
      continue;
    }
    if (auto parsed = internal::parse_int64(cells[i])) {
      ints[i] = *parsed;
      doubles[i] = static_cast<double>(*parsed);
      continue;
    }
    all_int = false;
    auto parsed = internal::parse_number(cells[i]);
    // "nan" does not count as a number for inference.
    if (!parsed || std::isnan(*parsed) ||
        internal::to_lower_ascii(internal::strip(cells[i])).find('_') != std::string::npos) {
      numeric = false;
      break;
    }
    doubles[i] = *parsed;
  }

  if (numeric && all_int && !any_empty) {
    arrow::Int64Builder builder;
    internal::check(builder.AppendValues(ints));
    return internal::unwrap(builder.Finish());
  }
  if (numeric) {
    arrow::DoubleBuilder builder;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      if (cells[i].empty()) {
        internal::check(builder.AppendNull());
      } else {
        internal::check(builder.Append(doubles[i]));
      }
    }
    return internal::unwrap(builder.Finish());
  }
  arrow::StringBuilder builder;
  for (const auto& cell : cells) {
    if (cell.empty()) {
      internal::check(builder.AppendNull());
    } else {
      internal::check(builder.Append(cell));
    }
  }
  return internal::unwrap(builder.Finish());
}

MSDataset read_delimited_text(const std::string& raw_text, char delimiter,
                              const std::string& format_name, const std::string& source_name,
                              const DelimitedReadOptions& options) {
  if (internal::strip(raw_text).empty()) {
    throw std::invalid_argument(format_name + " text is empty.");
  }
  auto rows = parse_csv(internal::normalize_newlines(raw_text), delimiter);
  if (rows.empty()) throw std::invalid_argument(format_name + " header is missing.");
  const auto& fieldnames = rows.front();
  std::unordered_set<std::string> unique(fieldnames.begin(), fieldnames.end());
  if (unique.size() != fieldnames.size()) {
    throw std::invalid_argument(format_name + " column names must be unique.");
  }
  auto peak_it = std::find(fieldnames.begin(), fieldnames.end(), kPeakColumn);
  if (peak_it == fieldnames.end()) {
    throw std::invalid_argument(format_name + " must contain a '" + kPeakColumn + "' column.");
  }
  const std::size_t peak_index = static_cast<std::size_t>(peak_it - fieldnames.begin());

  std::vector<std::string> metadata_columns;
  std::vector<std::size_t> metadata_indices;
  for (std::size_t i = 0; i < fieldnames.size(); ++i) {
    if (i != peak_index) {
      metadata_columns.push_back(fieldnames[i]);
      metadata_indices.push_back(i);
    }
  }

  std::vector<std::vector<std::string>> cells(metadata_columns.size());
  std::vector<double> data;
  std::vector<int64_t> offsets{0};
  // Peak metadata columns in order of first appearance; every column has one
  // value per peak read so far ("" where a row did not provide it).
  std::vector<std::string> peak_columns;
  std::unordered_map<std::string, std::vector<std::string>> peak_values;
  int64_t row_number = 1;
  for (std::size_t r = 1; r < rows.size(); ++r) {
    const auto& row = rows[r];
    if (row.empty()) continue;  // DictReader skips blank lines
    ++row_number;
    if (row.size() > fieldnames.size()) {
      throw std::invalid_argument("Too many fields at " + format_name + " row " +
                                  std::to_string(row_number) + " in " + source_name + ".");
    }
    for (std::size_t c = 0; c < metadata_indices.size(); ++c) {
      std::size_t index = metadata_indices[c];
      cells[c].push_back(index < row.size() ? row[index] : std::string());
    }
    std::string peak_text = peak_index < row.size() ? row[peak_index] : std::string();
    PeakCell cell = parse_peak_cell(peak_text, row_number, format_name);
    const std::size_t peaks_before = data.size() / 2;
    for (const auto& [mz, intensity] : cell.peaks) {
      data.push_back(mz);
      data.push_back(intensity);
    }
    for (const auto& name : cell.columns) {
      if (peak_values.count(name) == 0) {
        peak_values[name] = std::vector<std::string>(peaks_before);
        peak_columns.push_back(name);
      }
    }
    for (auto& [name, values] : peak_values) {
      auto it = std::find(cell.columns.begin(), cell.columns.end(), name);
      for (std::size_t p = 0; p < cell.peaks.size(); ++p) {
        values.push_back(it == cell.columns.end()
                             ? std::string()
                             : cell.values[p][static_cast<std::size_t>(it - cell.columns.begin())]);
      }
    }
    offsets.push_back(static_cast<int64_t>(data.size() / 2));
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (std::size_t c = 0; c < metadata_columns.size(); ++c) {
    auto array = options.infer_types ? infer_column(cells[c])
                                     : internal::strings_to_array(cells[c]);
    fields.push_back(arrow::field(metadata_columns[c], array->type()));
    arrays.push_back(array);
  }
  const int64_t n_rows = static_cast<int64_t>(offsets.size()) - 1;
  auto table = arrow::Table::Make(arrow::schema(fields), arrays, n_rows);
  std::shared_ptr<arrow::Table> peak_table;
  if (!peak_columns.empty()) {
    std::vector<std::shared_ptr<arrow::Field>> peak_fields;
    std::vector<std::shared_ptr<arrow::Array>> peak_arrays;
    for (const auto& name : peak_columns) {
      peak_fields.push_back(arrow::field(name, arrow::utf8()));
      peak_arrays.push_back(internal::strings_to_array(peak_values.at(name)));
    }
    peak_table = arrow::Table::Make(arrow::schema(peak_fields), peak_arrays,
                                    static_cast<int64_t>(data.size() / 2));
  }
  MSDataset dataset(table, PeakSeries(std::move(data), std::move(offsets), peak_table));
  if (options.spec_id_prefix && !dataset.has_column("SpecID")) {
    processing::set_spec_id(dataset, "SpecID", *options.spec_id_prefix);
  }
  return dataset;
}

std::string strip_bom(std::string text) {
  if (text.size() >= 3 && text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3);
  return text;
}

// csv.writer with QUOTE_MINIMAL and lineterminator "\n".
std::string csv_field(const std::string& value, char delimiter) {
  bool needs_quotes = value.find_first_of(std::string{delimiter, '"', '\n', '\r'}) !=
                      std::string::npos;
  if (!needs_quotes) return value;
  return "\"" + internal::replace_all(value, "\"", "\"\"") + "\"";
}

// Value text of a cell; a missing cell of a numeric column is written "nan"
// ("nan"), other missing values are None ("").
std::string csv_value(const arrow::Array& array, int64_t row) {
  if (array.IsNull(row)) {
    switch (array.type_id()) {
      case arrow::Type::FLOAT:
      case arrow::Type::DOUBLE:
      case arrow::Type::INT8:
      case arrow::Type::INT16:
      case arrow::Type::INT32:
      case arrow::Type::INT64:
      case arrow::Type::UINT8:
      case arrow::Type::UINT16:
      case arrow::Type::UINT32:
      case arrow::Type::UINT64:
        return "nan";
      default:
        return "";
    }
  }
  Value value = internal::value_at(array, row);
  if (const auto* d = std::get_if<double>(&value)) return format_float(*d);
  return to_display_string(value);
}

void write_delimited(const MSDataset& dataset, const std::string& path, char delimiter,
                     const std::string& format_name, const DelimitedWriteOptions& options) {
  std::vector<std::string> selected = options.headers.value_or(dataset.columns());
  if (std::find(selected.begin(), selected.end(), kPeakColumn) != selected.end()) {
    throw std::invalid_argument("'" + kPeakColumn + "' is reserved for spectrum peaks in " +
                                format_name + " files.");
  }
  std::vector<std::string> missing;
  for (const auto& column : selected) {
    if (!dataset.has_column(column)) missing.push_back(column);
  }
  if (!missing.empty()) {
    std::string list;
    for (const auto& column : missing) list += (list.empty() ? "'" : ", '") + column + "'";
    throw std::invalid_argument("Unknown " + format_name + " metadata columns: [" + list + "]");
  }

  auto metadata = internal::select_columns(*dataset.metadata(), selected);

  // Peak metadata columns to write.
  std::shared_ptr<arrow::Table> peak_metadata = dataset.peaks().metadata();
  std::vector<std::string> peak_columns;
  std::vector<std::shared_ptr<arrow::Array>> peak_arrays;
  if (peak_metadata) {
    for (const auto& column :
         options.peak_headers.value_or(internal::column_names(*peak_metadata))) {
      int index = internal::column_index(*peak_metadata, column);
      if (index < 0 || column == "mz" || column == "intensity") continue;
      peak_columns.push_back(column);
      peak_arrays.push_back(peak_metadata->column(index)->chunk(0));
    }
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open file for writing: " + path);
  const int64_t total = dataset.size();
  if (options.progress_callback) options.progress_callback(0, total);

  std::string header;
  for (const auto& column : selected) header += csv_field(column, delimiter) + delimiter;
  out << header << csv_field(kPeakColumn, delimiter) << "\n";

  int64_t peak_cursor = 0;  // row in the visible peak metadata table

  for (int64_t row = 0; row < total; ++row) {
    std::string line;
    for (int c = 0; c < metadata->num_columns(); ++c) {
      line += csv_field(csv_value(*metadata->column(c)->chunk(0), row), delimiter) + delimiter;
    }
    Spectrum spectrum = dataset.peaks().spectrum(row);
    const int64_t n_peaks = spectrum.size();

    // Peak metadata values of this spectrum; only columns with a value are written.
    std::vector<std::vector<std::string>> values(static_cast<std::size_t>(n_peaks));
    std::vector<std::size_t> used;
    for (std::size_t c = 0; c < peak_arrays.size(); ++c) {
      bool any = false;
      for (int64_t p = 0; p < n_peaks; ++p) {
        std::string text = to_display_string(internal::value_at(*peak_arrays[c], peak_cursor + p));
        any = any || !text.empty();
        values[static_cast<std::size_t>(p)].push_back(std::move(text));
      }
      if (any) used.push_back(c);
    }
    peak_cursor += n_peaks;

    std::string peaks;
    if (!used.empty()) {
      peaks = "mz,intensity";
      for (std::size_t c : used) peaks += "," + peak_cell_field(peak_columns[c]);
    }
    for (int64_t p = 0; p < n_peaks; ++p) {
      if (!peaks.empty()) peaks += ";";
      peaks += internal::format_g17(spectrum.mz(p)) + "," +
               internal::format_g17(spectrum.intensity(p));
      // Trailing empty values are omitted ("mz,intensity" alone is valid).
      std::size_t last = used.size();
      while (last > 0 && values[static_cast<std::size_t>(p)][used[last - 1]].empty()) --last;
      for (std::size_t k = 0; k < last; ++k) {
        peaks += "," + peak_cell_field(values[static_cast<std::size_t>(p)][used[k]]);
      }
    }
    out << line << csv_field(peaks, delimiter) << "\n";
    if (options.progress_callback) options.progress_callback(row + 1, total);
  }
  if (!out) throw std::runtime_error("failed to write " + path);
}

}  // namespace

MSDataset read_tsv_text(const std::string& text, const DelimitedReadOptions& options,
                        const std::string& source_name) {
  return read_delimited_text(text, '\t', "TSV", source_name, options);
}

MSDataset read_tsv(const std::string& path, const DelimitedReadOptions& options) {
  return read_delimited_text(strip_bom(internal::read_file(path)), '\t', "TSV", path, options);
}

void write_tsv(const MSDataset& dataset, const std::string& path,
               const DelimitedWriteOptions& options) {
  write_delimited(dataset, path, '\t', "TSV", options);
}

MSDataset read_csv_text(const std::string& text, const DelimitedReadOptions& options,
                        const std::string& source_name) {
  return read_delimited_text(text, ',', "CSV", source_name, options);
}

MSDataset read_csv(const std::string& path, const DelimitedReadOptions& options) {
  return read_delimited_text(strip_bom(internal::read_file(path)), ',', "CSV", path, options);
}

void write_csv(const MSDataset& dataset, const std::string& path,
               const DelimitedWriteOptions& options) {
  write_delimited(dataset, path, ',', "CSV", options);
}

}  // namespace msentity::io
