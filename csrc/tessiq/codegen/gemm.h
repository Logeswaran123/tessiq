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
/// A row-major matrix is either laid out with its rows contiguous or with its columns contiguous,
/// and the second is the transpose of the first over the same bytes. BLAS takes that as a flag,
/// so a transposed operand costs nothing rather than a copy. `nn.Linear` produces one every time:
/// it stores its weight as [out, in] and computes with [in, out], so every weight arrives here
/// transposed.
struct GemmOperand {
  /// The IR name of the value read.
  std::string name;

  /// Whether the library has to read it transposed.
  bool transposed = false;

  /// The step between rows of the matrix as stored, which BLAS calls the leading dimension.
  std::int64_t leading = 0;
};

/// A matrix multiply, described as the library's arguments rather than as a loop.
///
/// `[batch, m, k] by [batch, k, n] giving [batch, m, n]`, with batch 1 for a plain matmul. The
/// bias is present only for addmm, where it is a row of `n` values added to every row of the
/// result.
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

  /// Elements in each operand and in the result, for checking what the launch is handed.
  [[nodiscard]] std::int64_t lhs_elements() const { return batch * m * k; }

  [[nodiscard]] std::int64_t rhs_elements() const { return batch * k * n; }

  [[nodiscard]] std::int64_t output_elements() const { return batch * m * n; }
};

/// Describes `node` as a matrix multiply, or nothing if it cannot be one.
///
/// Nothing is not a failure: an operation that is not a matmul, or one whose operands are laid out
/// in a way BLAS cannot name with a transpose flag, has no GEMM and runs elsewhere. An error means
/// the node claims to be a matmul and contradicts itself.
Result<std::optional<GemmKernel>> extract_gemm(const ir::Node& node);

/// Source for the kernel that fills an addmm's result with its bias before the multiply.
///
/// BLAS scales the destination and adds, which is the only way it takes an accumulator, so a bias
/// has to be in the output already. Doing that with torch operations costs a host round trip per
/// multiply, which on a transformer block is more than the multiplies.
[[nodiscard]] std::string broadcast_row_source(ir::DType dtype, std::string_view name);

/// The nodes a GEMM could be built from, in graph order.
[[nodiscard]] std::vector<const ir::Node*> gemm_nodes(const ir::Graph& graph);

} // namespace tessiq::codegen
