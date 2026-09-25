// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/support/result.h"

namespace tessiq::ir {

/// Element types the IR can carry. Smaller than torch's set on purpose; a graph using any other
/// type is not imported and runs on the host.
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

/// True for the floating-point types.
bool is_float(DType dtype);

/// A dimension whose size is not known until run time.
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

  /// Renders as the textual IR spells it: `[4,8]`, `[]`, `[?,16]`.
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const Shape& a, const Shape& b) { return a.dims_ == b.dims_; }

  friend bool operator!=(const Shape& a, const Shape& b) { return !(a == b); }

 private:
  std::vector<std::int64_t> dims_;
};

/// How a tensor's elements are laid out: the step between consecutive indices on each axis, in
/// elements. Empty means contiguous, packed row-major, which almost every value is.
using Strides = std::vector<std::int64_t>;

/// The strides a contiguous tensor of `shape` has. Empty for a rank-0 shape.
///
/// A dynamic extent counts as 1: the strides outside it are unknown anyway, and a placeholder keeps
/// the rank right.
[[nodiscard]] Strides contiguous_strides(const Shape& shape);

/// The type of a value: element type, shape and layout.
class TensorType {
 public:
  TensorType() = default;

  TensorType(DType dtype, Shape shape) : dtype_(dtype), shape_(std::move(shape)) {}

  /// A layout equal to the contiguous one is stored as contiguous, so both spellings compare
  /// equal.
  TensorType(DType dtype, Shape shape, Strides strides);

  [[nodiscard]] DType dtype() const { return dtype_; }

  [[nodiscard]] const Shape& shape() const { return shape_; }

  /// The layout, or empty when it is the contiguous one.
  [[nodiscard]] const Strides& strides() const { return strides_; }

  /// Whether the elements are packed in row-major order, which is what every kernel here reads.
  [[nodiscard]] bool is_contiguous() const { return strides_.empty(); }

  /// The same type laid out contiguously, which is what materialising this value produces.
  [[nodiscard]] TensorType contiguous() const { return {dtype_, shape_}; }

  /// Renders as it appears in the IR: `f32[4,8]`, `i64[]`, `f32[?,16]`, `f32[4,8:1,4]`.
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const TensorType& a, const TensorType& b) {
    // Layout is part of the type. Two values of the same shape and dtype in different layouts
    // cannot share a loop, and fusion decides that by comparing types.
    return a.dtype_ == b.dtype_ && a.shape_ == b.shape_ && a.strides_ == b.strides_;
  }

  friend bool operator!=(const TensorType& a, const TensorType& b) { return !(a == b); }

 private:
  DType dtype_ = DType::kF32;
  Shape shape_;
  Strides strides_;
};

/// Parses `f32[4,8]`, `i64[]`, `f32[?,16]` or a strided `f32[4,8:1,4]`.
Result<TensorType> parse_tensor_type(std::string_view text);

/// Parses a whole `std::string_view` as an integer, rejecting trailing characters.
///
/// Wraps from_chars, whose pointer arithmetic needs a clang-tidy suppression, so the suppression
/// is written once.
Result<std::int64_t> parse_int(std::string_view text);

/// Parses a whole `std::string_view` as a double, rejecting trailing characters.
Result<double> parse_double(std::string_view text);

} // namespace tessiq::ir
