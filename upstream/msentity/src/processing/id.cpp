#include "msentity/processing/id.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "internal/table.hpp"

namespace msentity::processing {

bool set_spec_id(MSDataset& dataset, const std::string& col_name, const std::string& prefix,
                 bool overwrite, int64_t start) {
  if (start < 0) {
    throw std::invalid_argument("start must be an integer greater than or equal to 0.");
  }
  if (internal::column_index(*dataset.spectrum_metadata_ref(), col_name) >= 0 && !overwrite) {
    return false;
  }
  const int64_t n = dataset.size();
  if (n == 0) {
    dataset.set_column(col_name, std::vector<Value>{});
    return true;
  }
  const std::size_t width = std::to_string(start + n - 1).size();
  std::vector<Value> ids;
  ids.reserve(static_cast<std::size_t>(n));
  for (int64_t i = start; i < start + n; ++i) {
    std::string number = std::to_string(i);
    if (number.size() < width) number.insert(0, width - number.size(), '0');
    ids.emplace_back(prefix + number);
  }
  dataset.set_column(col_name, ids);
  return true;
}

}  // namespace msentity::processing
