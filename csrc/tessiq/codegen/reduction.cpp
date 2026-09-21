// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/reduction.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/codegen/program.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/shape_rules.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

namespace {

std::optional<ReduceOp> combiner(std::string_view canonical) {
  if (canonical == "sum") {
    return ReduceOp::kSum;
  }
  if (canonical == "mean") {
    return ReduceOp::kMean;
  }
  if (canonical == "amax") {
    return ReduceOp::kMax;
  }
  if (canonical == "amin") {
    return ReduceOp::kMin;
  }
  return std::nullopt;
}

std::int64_t product(const std::vector<std::int64_t>& dims, std::size_t from, std::size_t to) {
  std::int64_t total = 1;
  for (std::size_t i = from; i < to; ++i) {
    total *= dims[i];
  }
  return total;
}

} // namespace

Result<std::optional<ReductionKernel>> extract_reduction(const ir::Node& node) {
  const passes::OpInfo* info = passes::find_op(node.op());
  if (info == nullptr || info->category != passes::OpCategory::kReduction) {
    return std::optional<ReductionKernel>{};
  }
  const std::optional<ReduceOp> op = combiner(info->canonical);
  if (!op.has_value()) {
    // In the table as a reduction, but this code has no combiner for it. A new row in ops.toml
    // should fall back rather than be guessed at.
    return std::optional<ReductionKernel>{};
  }

  if (node.operand_count() != 1 || node.result_count() != 1) {
    return invalid_argument(node.op() + " takes one operand and produces one result");
  }

  // Checked rather than assumed. A node holding a null operand is a malformed graph, and reading
  // through it would be a crash in the compiler rather than a message about the graph.
  const ir::Value* source = node.operand(0);
  const ir::Value* target = node.result(0);
  if (source == nullptr || target == nullptr) {
    return internal_error(node.op() + " has an operand or result that is not set");
  }

  const ir::TensorType& input = source->type();
  const std::vector<std::int64_t>& dims = input.shape().dims();
  // The same call the shape rule makes, so the kernel reduces exactly the axes the rule
  // predicted a shape for.
  TESSIQ_ASSIGN_OR_RETURN(const std::vector<std::size_t> axes,
                          passes::reduced_axes(node.op(), node.attributes(), dims.size()));

  if (axes.empty()) {
    // A rank-0 input has nothing to reduce, so there is no loop to write.
    return std::optional<ReductionKernel>{};
  }

  // Only a contiguous range of axes becomes [outer, reduce, inner]. Reducing the first and last
  // axis of a rank-3 tensor leaves the kept axis in the middle, which this shape cannot express.
  if (axes.back() - axes.front() + 1 != axes.size()) {
    return std::optional<ReductionKernel>{};
  }

  for (const std::int64_t dim : dims) {
    if (dim == ir::kDynamic || dim < 0) {
      // The extents are baked into the launch, so a shape that is not yet known has no kernel.
      return std::optional<ReductionKernel>{};
    }
  }

  ReductionKernel kernel;
  kernel.op = *op;
  kernel.dtype = input.dtype();
  kernel.outer = product(dims, 0, axes.front());
  kernel.reduce = product(dims, axes.front(), axes.back() + 1);
  kernel.inner = product(dims, axes.back() + 1, dims.size());
  kernel.input = source->name();
  kernel.output = target->name();
  // Consecutive threads have to read consecutive addresses. With no kept axis after the reduced
  // one, a block's threads walking the reduced axis do that; with a kept axis after it, the
  // threads of one output are `inner` apart and it is the outputs that sit next to each other.
  kernel.shape =
      kernel.inner == 1 ? ReductionShape::kBlockPerOutput : ReductionShape::kThreadPerOutput;
  kernel.name = "tessiq_reduce_" + kernel.output;

  if (kernel.reduce <= 0 || kernel.outputs() <= 0) {
    // An empty reduction has a defined answer for sum and mean and no answer for max and min, and
    // the fallback already knows which. Not worth a kernel either way.
    return std::optional<ReductionKernel>{};
  }
  return std::optional<ReductionKernel>{std::move(kernel)};
}

std::int64_t ReductionKernel::blocks() const {
  if (shape == ReductionShape::kBlockPerOutput) {
    return outputs();
  }
  const auto width = static_cast<std::int64_t>(kBlockSize);
  return (outputs() + width - 1) / width;
}

std::vector<const ir::Node*> reduction_nodes(const ir::Graph& graph) {
  std::vector<const ir::Node*> nodes;
  for (const ir::Node* node : graph.nodes()) {
    const passes::OpInfo* info = passes::find_op(node->op());
    if (info != nullptr && info->category == passes::OpCategory::kReduction) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

} // namespace tessiq::codegen
