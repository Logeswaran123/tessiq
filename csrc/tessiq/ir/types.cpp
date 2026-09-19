// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/ir/types.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "tessiq/support/macros.h"

namespace tessiq::ir {

namespace {

struct DTypeEntry {
  DType dtype;
  std::string_view name;
  std::size_t size;
  bool is_float;
};

// One table, so a new dtype cannot be half-added: the name, the size and the float flag all come
// from the same row.
constexpr std::array<DTypeEntry, 7> kDTypes{{
    {DType::kF16, "f16", 2, true},
    {DType::kBF16, "bf16", 2, true},
    {DType::kF32, "f32", 4, true},
    {DType::kF64, "f64", 8, true},
    {DType::kI1, "i1", 1, false},
    {DType::kI32, "i32", 4, false},
    {DType::kI64, "i64", 8, false},
}};

const DTypeEntry& entry(DType dtype) {
  for (const auto& row : kDTypes) {
    if (row.dtype == dtype) {
      return row;
    }
  }
  TESSIQ_UNREACHABLE("unhandled DType");
}

} // namespace

std::string_view dtype_name(DType dtype) { return entry(dtype).name; }

std::size_t dtype_size(DType dtype) { return entry(dtype).size; }

bool is_float(DType dtype) { return entry(dtype).is_float; }

Result<DType> parse_dtype(std::string_view name) {
  for (const auto& row : kDTypes) {
    if (row.name == name) {
      return row.dtype;
    }
  }

  std::string known;
  for (const auto& row : kDTypes) {
    known += known.empty() ? "" : ", ";
    known += row.name;
  }
  return invalid_argument("unknown dtype '" + std::string(name) + "'; expected one of: " + known);
}

Result<std::int64_t> parse_int(std::string_view text) {
  std::int64_t value = 0;
  // NOLINTBEGIN(bugprone-suspicious-stringview-data-usage): from_chars takes a begin and an end
  // pointer, so the size is exactly what is being passed.
  const auto* const end = text.data() + text.size();
  const auto result = std::from_chars(text.data(), end, value);
  // NOLINTEND(bugprone-suspicious-stringview-data-usage)
  if (result.ec != std::errc{} || result.ptr != end) {
    return invalid_argument("expected an integer, got '" + std::string(text) + "'");
  }
  return value;
}

Result<double> parse_double(std::string_view text) {
  // strtod rather than stod: it reports failure through a pointer instead of an exception, so the
  // caller does not need a catch block that does nothing.
  const std::string owned(text);
  char* end = nullptr;
  const double value = std::strtod(owned.c_str(), &end);
  if (end != owned.c_str() + owned.size() || owned.empty()) {
    return invalid_argument("expected a number, got '" + std::string(text) + "'");
  }
  return value;
}

bool Shape::is_static() const {
  return std::none_of(dims_.begin(), dims_.end(), [](std::int64_t d) { return d == kDynamic; });
}

std::int64_t Shape::element_count() const {
  TESSIQ_DCHECK(is_static(), "element_count() on a shape with an unknown dimension");
  return std::accumulate(dims_.begin(), dims_.end(), std::int64_t{1}, std::multiplies<>());
}

std::string Shape::to_string() const {
  std::string text("[");
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    if (i != 0) {
      text += ',';
    }
    text += dims_[i] == kDynamic ? "?" : std::to_string(dims_[i]);
  }
  text += ']';
  return text;
}

std::string TensorType::to_string() const {
  return std::string(dtype_name(dtype_)) + shape_.to_string();
}

Result<TensorType> parse_tensor_type(std::string_view text) {
  const std::size_t open = text.find('[');
  if (open == std::string_view::npos || text.back() != ']') {
    return invalid_argument("malformed type '" + std::string(text) +
                            "'; expected something like f32[4,8]");
  }

  TESSIQ_ASSIGN_OR_RETURN(const DType dtype, parse_dtype(text.substr(0, open)));

  std::vector<std::int64_t> dims;
  std::string_view body = text.substr(open + 1, text.size() - open - 2);
  while (!body.empty()) {
    const std::size_t comma = body.find(',');
    const std::string_view token = body.substr(0, comma);
    if (token == "?") {
      dims.push_back(kDynamic);
    } else {
      auto value = parse_int(token);
      if (!value || *value < 0) {
        return invalid_argument("bad dimension '" + std::string(token) + "' in type '" +
                                std::string(text) + "'; expected a non-negative number or ?");
      }
      dims.push_back(*value);
    }
    if (comma == std::string_view::npos) {
      break;
    }
    body.remove_prefix(comma + 1);
  }

  return TensorType(dtype, Shape(std::move(dims)));
}

} // namespace tessiq::ir
