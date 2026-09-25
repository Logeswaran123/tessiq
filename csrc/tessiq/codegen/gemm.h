// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// One operand of a matrix multiply, as the library needs to see it.
///
/// A matrix stored with its columns contiguous is the transpose of a row-major one over the same
/// bytes, and BLAS takes that as a flag instead of a copy. `nn.Linear` stores its weight as
/// [out, in] and multiplies by its transpose, so every weight arrives transposed.
struct GemmOperand {
  /// The IR name of the value read.
  std::string name;

  /// Whether the library has to read it transposed.
  bool transposed = false;

  /// The step between rows of the matrix as stored, which BLAS calls the leading dimension.
  std::int64_t leading = 0;
};

/// The numbers one BLAS call needs: the description's own when the shapes were known, otherwise
/// read from the tensors at the call.
struct GemmExtents {
  std::int64_t batch = 1;
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;

  /// The step between rows of each operand as stored, which BLAS calls the leading dimension.
  std::int64_t lhs_leading = 0;
  std::int64_t rhs_leading = 0;

  [[nodiscard]] std::int64_t output_elements() const { return batch * m * n; }

  /// Whether these describe a multiply BLAS can run.
  [[nodiscard]] bool are_usable() const;
};

/// A matrix multiply, described as the library's arguments rather than as a loop.
///
/// `[batch, m, k] by [batch, k, n] giving [batch, m, n]`, with batch 1 for a plain matmul. The
/// bias is present only for addmm, where it is a row of `n` values added to every row of the
/// result.
///
/// An extent is `ir::kDynamic` when the graph was traced for more than one shape. The transpose
/// flags are always known, since they follow from the layout.
struct GemmKernel {
  std::int64_t batch = 1;
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;

  GemmOperand lhs;
  GemmOperand rhs;

  /// The IR name of the bias, when there is one. A row of `n` values, broadcast over the m rows.
  std::optional<std::string> bias;

  /// The IR name of the value written.
  std::string output;

  ir::DType dtype = ir::DType::kF32;

  /// Whether any number the call needs arrives with the tensors rather than being known here.
  [[nodiscard]] bool has_dynamic_extent() const;

  /// What the description itself knows, with `ir::kDynamic` where it knows nothing.
  [[nodiscard]] GemmExtents extents() const {
    GemmExtents known;
    known.batch = batch;
    known.m = m;
    known.n = n;
    known.k = k;
    known.lhs_leading = lhs.leading;
    known.rhs_leading = rhs.leading;
    return known;
  }
};

/// Describes `node` as a matrix multiply, or nothing if it cannot be one.
///
/// Nothing when the node is not a matmul or an operand's layout is not one a transpose flag
/// describes; it then runs elsewhere. An error means the node is malformed.
Result<std::optional<GemmKernel>> extract_gemm(const ir::Node& node);

/// Source for the kernel that fills an addmm's result with its bias before the multiply.
///
/// BLAS adds `beta * C` to the product, so the bias has to be in the output before the call.
/// Doing that with torch operations would cost a host round trip per multiply.
[[nodiscard]] std::string broadcast_row_source(ir::DType dtype, std::string_view name);

/// The nodes a GEMM could be built from, in graph order.
[[nodiscard]] std::vector<const ir::Node*> gemm_nodes(const ir::Graph& graph);

} // namespace tessiq::codegen
