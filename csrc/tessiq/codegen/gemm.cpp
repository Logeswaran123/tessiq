// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/gemm.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

namespace {

/// Reads a matrix operand's layout, or nothing when BLAS cannot name it with a transpose flag.
///
/// The two it can name are rows contiguous and columns contiguous. Anything else, such as a slice
/// of a larger matrix with a gap between its rows, would need the leading dimension to differ from
/// the extent, which this does not try to recognise yet.
std::optional<GemmOperand>
read_operand(const ir::Value& value, std::int64_t rows, std::int64_t columns) {
  const ir::Strides& strides = value.type().strides();
  const std::size_t rank = value.type().shape().rank();

  // Contiguous, which the type records as no layout at all.
  if (strides.empty()) {
    return GemmOperand{value.name(), false, columns};
  }
  if (strides.size() != rank || rank < 2) {
    return std::nullopt;
  }

  const std::int64_t row_step = strides[rank - 2];
  const std::int64_t column_step = strides[rank - 1];
  if (column_step == 1) {
    return GemmOperand{value.name(), false, row_step};
  }
  if (row_step == 1) {
    // Columns are contiguous, so the bytes are the transpose of this matrix and the step between
    // what BLAS will read as rows is the other stride.
    return GemmOperand{value.name(), true, column_step};
  }
  static_cast<void>(rows);
  return std::nullopt;
}

/// The trailing two extents of a shape, which are the matrix, and the product of the rest.
struct Dims {
  std::int64_t batch = 1;
  std::int64_t rows = 0;
  std::int64_t columns = 0;
};

std::optional<Dims> matrix_dims(const ir::TensorType& type) {
  const std::vector<std::int64_t>& sizes = type.shape().dims();
  if (sizes.size() < 2) {
    return std::nullopt;
  }
  Dims dims;
  dims.rows = sizes[sizes.size() - 2];
  dims.columns = sizes.back();
  for (std::size_t i = 0; i + 2 < sizes.size(); ++i) {
    dims.batch *= sizes[i];
  }
  if (dims.rows <= 0 || dims.columns <= 0 || dims.batch <= 0) {
    return std::nullopt;
  }
  return dims;
}

} // namespace

Result<std::optional<GemmKernel>> extract_gemm(const ir::Node& node) {
  const passes::OpInfo* info = passes::find_op(node.op());
  if (info == nullptr || info->category != passes::OpCategory::kMatmul) {
    return std::optional<GemmKernel>{};
  }

  // addmm takes the bias first, then the two matrices. mm and bmm take the matrices alone.
  const bool has_bias = info->canonical == "addmm";
  const std::size_t expected = has_bias ? 3U : 2U;
  if (node.operand_count() != expected || node.result_count() != 1) {
    return invalid_argument(node.op() + " takes " + std::to_string(expected) +
                            " operands and produces one result");
  }

  const ir::Value* lhs = node.operand(has_bias ? 1 : 0);
  const ir::Value* rhs = node.operand(has_bias ? 2 : 1);
  const ir::Value* out = node.result(0);
  if (lhs == nullptr || rhs == nullptr || out == nullptr) {
    return internal_error(node.op() + " has an operand or result that is not set");
  }

  const std::optional<Dims> left = matrix_dims(lhs->type());
  const std::optional<Dims> right = matrix_dims(rhs->type());
  const std::optional<Dims> result = matrix_dims(out->type());
  if (!left.has_value() || !right.has_value() || !result.has_value()) {
    return std::optional<GemmKernel>{};
  }
  if (left->columns != right->rows || left->rows != result->rows ||
      right->columns != result->columns || left->batch != right->batch ||
      left->batch != result->batch) {
    return invalid_argument(
        node.op() + " has operands that do not multiply: " + lhs->type().to_string() + " by " +
        rhs->type().to_string() + " giving " + out->type().to_string());
  }

  if (lhs->type().dtype() != rhs->type().dtype() || lhs->type().dtype() != out->type().dtype()) {
    return unsupported(node.op() + " on mixed dtypes; promotion happens before import");
  }
  if (!out->type().is_contiguous()) {
    // The result is written row by row, so a caller expecting something else would read it in the
    // wrong order.
    return std::optional<GemmKernel>{};
  }

  const std::optional<GemmOperand> a = read_operand(*lhs, left->rows, left->columns);
  const std::optional<GemmOperand> b = read_operand(*rhs, right->rows, right->columns);
  if (!a.has_value() || !b.has_value()) {
    return std::optional<GemmKernel>{};
  }

  GemmKernel kernel;
  kernel.batch = left->batch;
  kernel.m = left->rows;
  kernel.k = left->columns;
  kernel.n = right->columns;
  kernel.lhs = *a;
  kernel.rhs = *b;
  kernel.output = out->name();
  kernel.dtype = out->type().dtype();

  if (has_bias) {
    const ir::Value* bias = node.operand(0);
    if (bias == nullptr) {
      return internal_error(node.op() + " has no bias operand");
    }
    // A row of n values added to every row. Anything else, such as a full matrix, is a shape this
    // does not handle and runs elsewhere.
    const std::vector<std::int64_t>& sizes = bias->type().shape().dims();
    if (sizes.size() != 1 || sizes.front() != kernel.n || !bias->type().is_contiguous()) {
      return std::optional<GemmKernel>{};
    }
    kernel.bias = bias->name();
  }

  return std::optional<GemmKernel>{std::move(kernel)};
}

std::string broadcast_row_source(ir::DType dtype, std::string_view name) {
  const std::string_view scalar = dtype == ir::DType::kF64 ? "double" : "float";
  std::string source = "// Generated by tessiq. Do not edit.\n";
  source += "typedef " + std::string(scalar) + " scalar_t;\n\n";
  source += "extern \"C\" __global__ void " + std::string(name) + "(\n";
  source += "    const scalar_t* __restrict__ p_row,\n";
  source += "    scalar_t* __restrict__ p_out,\n";
  source += "    long long total,\n";
  source += "    long long width) {\n";
  source += "  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
  source += "  if (i >= total) return;\n";
  source += "  p_out[i] = p_row[i % width];\n}\n";
  return source;
}

std::vector<const ir::Node*> gemm_nodes(const ir::Graph& graph) {
  std::vector<const ir::Node*> nodes;
  for (const ir::Node* node : graph.nodes()) {
    const passes::OpInfo* info = passes::find_op(node->op());
    if (info != nullptr && info->category == passes::OpCategory::kMatmul) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

} // namespace tessiq::codegen
