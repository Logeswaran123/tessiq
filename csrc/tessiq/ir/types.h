// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/support/result.h"

namespace tessiq::ir {

/// Element types the IR can carry.
///
/// Deliberately smaller than torch's set. An op set that has to reason about every dtype torch
/// supports is an op set nobody can write a pass against; anything missing falls back to the host.
enum class DType {
  kF16,
  kBF16,
  kF32,
  kF64,
  kI1,
  kI32,
  kI64,
};

/// Name used in the textual IR, and accepted when parsing it.
std::string_view dtype_name(DType dtype);

Result<DType> parse_dtype(std::string_view name);

/// Bytes per element.
std::size_t dtype_size(DType dtype);

/// True for the floating-point types, which is the distinction most passes actually care about.
bool is_float(DType dtype);

/// A dimension whose size is not known until run time.
///
/// Shapes carry these from the start rather than gaining them later, because a shape type that
/// cannot express "unknown" forces every dynamic-shape decision into a side table.
inline constexpr std::int64_t kDynamic = -1;

/// Tensor shape. An empty dim list is a scalar, which is distinct from a 1-element vector.
class Shape {
 public:
  Shape() = default;

  explicit Shape(std::vector<std::int64_t> dims) : dims_(std::move(dims)) {}

  [[nodiscard]] const std::vector<std::int64_t>& dims() const { return dims_; }

  [[nodiscard]] std::size_t rank() const { return dims_.size(); }

  [[nodiscard]] bool is_scalar() const { return dims_.empty(); }

  /// True if every dimension is known.
  [[nodiscard]] bool is_static() const;

  /// Elements in the tensor. Meaningless unless is_static(), so callers check first.
  [[nodiscard]] std::int64_t element_count() const;

  friend bool operator==(const Shape& a, const Shape& b) { return a.dims_ == b.dims_; }

  friend bool operator!=(const Shape& a, const Shape& b) { return !(a == b); }

 private:
  std::vector<std::int64_t> dims_;
};

/// The type of a value: element type plus shape.
///
/// Shape and dtype live on the type rather than in a side table, which is what lets a pass ask a
/// value what it is without consulting anything else, and what makes the textual form readable.
class TensorType {
 public:
  TensorType() = default;

  TensorType(DType dtype, Shape shape) : dtype_(dtype), shape_(std::move(shape)) {}

  [[nodiscard]] DType dtype() const { return dtype_; }

  [[nodiscard]] const Shape& shape() const { return shape_; }

  /// Renders as it appears in the IR: `f32[4,8]`, `i64[]`, `f32[?,16]`.
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const TensorType& a, const TensorType& b) {
    return a.dtype_ == b.dtype_ && a.shape_ == b.shape_;
  }

  friend bool operator!=(const TensorType& a, const TensorType& b) { return !(a == b); }

 private:
  DType dtype_ = DType::kF32;
  Shape shape_;
};

/// Parses `f32[4,8]`, `i64[]` or `f32[?,16]`.
Result<TensorType> parse_tensor_type(std::string_view text);

/// Parses a whole `std::string_view` as an integer, rejecting trailing characters.
///
/// The from_chars idiom takes a begin and an end pointer, which clang-tidy reads as an unbounded
/// data() call. Keeping it in one place means the suppression is written once rather than at
/// every call site.
Result<std::int64_t> parse_int(std::string_view text);

/// Parses a whole `std::string_view` as a double, rejecting trailing characters.
Result<double> parse_double(std::string_view text);

} // namespace tessiq::ir
