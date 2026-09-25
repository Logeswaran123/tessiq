// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/shape_rules.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/macros.h"

namespace tessiq::passes {

namespace {

/// How an op derives its result type from its operands.
enum class RuleKind {
  /// One tensor in, same type out: abs, relu, tanh.
  kUnaryElementwise,
  /// Two or more tensors in, broadcast shape and matching dtype out: add, mul, and a registered
  /// operation of any arity above one. Also covers the one-operand form the importer produces
  /// when the second argument was a Python scalar.
  kBinaryElementwise,
  /// Reduces the dimensions the node names, dropping them or setting them to 1.
  kReduction,
  /// Matrix multiply: [m,k] by [k,n] gives [m,n].
  kMatmul,
};

/// The rule for `op`, or nothing if it has none. Derived from the operation's category and arity
/// in the op table, so an operation added to ops.toml gets its rule without a second list.
std::optional<RuleKind> find_rule(std::string_view op) {
  const OpInfo* info = find_op(op);
  if (info == nullptr) {
    return std::nullopt;
  }
  switch (info->category) {
  case OpCategory::kElementwise:
    return info->arity == 1 ? RuleKind::kUnaryElementwise : RuleKind::kBinaryElementwise;
  case OpCategory::kReduction:
    return RuleKind::kReduction;
  case OpCategory::kMovement:
    // No rule yet. Each movement operation reads its attributes differently (a view takes new
    // sizes, a permute reorders them, a slice narrows one) and changes the layout. Imported graphs
    // already carry the result type.
    return std::nullopt;
  case OpCategory::kMatmul:
    return RuleKind::kMatmul;
  case OpCategory::kConvolution:
  case OpCategory::kCustom:
    // No rule for either. A convolution's extents depend on stride, padding and dilation, and a
    // custom kernel's result is its own business. Imported graphs already carry the result type.
    return std::nullopt;
  case OpCategory::kCopy:
    // Same shape out. The result is packed whatever the input's layout, which no rule here
    // computes, so this gives the shape only.
    return RuleKind::kUnaryElementwise;
  }
  TESSIQ_UNREACHABLE("unhandled OpCategory");
}

std::string operand_summary(const std::vector<ir::TensorType>& operands) {
  std::string text;
  for (const ir::TensorType& type : operands) {
    text += text.empty() ? "" : ", ";
    text += type.to_string();
  }
  return text;
}

Result<std::vector<ir::TensorType>> infer_unary(std::string_view op,
                                                const std::vector<ir::TensorType>& operands) {
  if (operands.size() != 1) {
    return invalid_argument(std::string(op) + " takes one tensor operand, got " +
                            std::to_string(operands.size()));
  }
  return std::vector<ir::TensorType>{operands.front()};
}

Result<std::vector<ir::TensorType>> infer_binary(std::string_view op,
                                                 const std::vector<ir::TensorType>& operands) {
  // The one-operand form is a tensor and a scalar attribute, so the tensor's type carries through.
  if (operands.size() == 1) {
    return std::vector<ir::TensorType>{operands.front()};
  }
  if (operands.empty()) {
    return invalid_argument(std::string(op) + " takes at least one tensor operand");
  }

  ir::Shape shape = operands.front().shape();
  for (std::size_t i = 1; i < operands.size(); ++i) {
    if (operands[i].dtype() != operands.front().dtype()) {
      // Not promoted here. AOTAutograd graphs are already promoted, so mixed dtypes mean the
      // input is not what this pass expects.
      return unsupported(std::string(op) + " on mixed dtypes " + operands.front().to_string() +
                         " and " + operands[i].to_string() + "; promotion happens before import");
    }
    TESSIQ_ASSIGN_OR_RETURN(shape, broadcast_shapes(shape, operands[i].shape()));
  }
  return std::vector<ir::TensorType>{ir::TensorType(operands.front().dtype(), std::move(shape))};
}

/// The attribute named `name`, or null if the node has none.
const ir::Attribute* attribute_or_null(const ir::AttributeMap& attributes, std::string_view name) {
  const auto it = attributes.find(name);
  return it == attributes.end() ? nullptr : &it->second;
}

/// Whether the reduced dimensions stay as 1 rather than being dropped.
///
/// torch passes keepdim positionally, so it arrives as `arg2`. The keyword spelling is accepted
/// too, because a graph written by hand is clearer with a name than with a position.
bool keeps_dims(const ir::AttributeMap& attributes) {
  for (const std::string_view name : {"arg2", "keepdim"}) {
    if (const ir::Attribute* value = attribute_or_null(attributes, name)) {
      if (const auto* flag = std::get_if<bool>(value)) {
        return *flag;
      }
    }
  }
  return false;
}

Result<std::vector<ir::TensorType>> infer_reduction(std::string_view op,
                                                    const std::vector<ir::TensorType>& operands,
                                                    const ir::AttributeMap& attributes) {
  if (operands.size() != 1) {
    return invalid_argument(std::string(op) + " takes one tensor operand, got " +
                            std::to_string(operands.size()));
  }

  const ir::TensorType& input = operands.front();
  const std::size_t rank = input.shape().rank();
  TESSIQ_ASSIGN_OR_RETURN(const std::vector<std::size_t> dims, reduced_axes(op, attributes, rank));

  const bool keepdim = keeps_dims(attributes);
  std::vector<std::int64_t> result;
  for (std::size_t i = 0; i < rank; ++i) {
    const bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
    if (!reduced) {
      result.push_back(input.shape().dims()[i]);
    } else if (keepdim) {
      result.push_back(1);
    }
  }

  // Reducing every dimension without keepdim leaves a rank-0 tensor, as torch does for sum(x).
  return std::vector<ir::TensorType>{ir::TensorType(input.dtype(), ir::Shape(std::move(result)))};
}

Result<std::vector<ir::TensorType>> infer_matmul(std::string_view op,
                                                 const std::vector<ir::TensorType>& operands) {
  if (operands.size() != 2) {
    return invalid_argument(std::string(op) + " takes two operands, got " +
                            std::to_string(operands.size()));
  }
  if (operands[0].dtype() != operands[1].dtype()) {
    return unsupported(std::string(op) + " on mixed dtypes " + operand_summary(operands));
  }
  if (operands[0].shape().rank() != 2 || operands[1].shape().rank() != 2) {
    return invalid_argument(std::string(op) + " takes two rank-2 operands, got " +
                            operand_summary(operands));
  }

  const std::int64_t inner_lhs = operands[0].shape().dims()[1];
  const std::int64_t inner_rhs = operands[1].shape().dims()[0];
  const bool both_known = inner_lhs != ir::kDynamic && inner_rhs != ir::kDynamic;
  if (both_known && inner_lhs != inner_rhs) {
    return invalid_argument(std::string(op) +
                            " inner dimensions disagree: " + operand_summary(operands));
  }

  const std::vector<std::int64_t> dims{operands[0].shape().dims()[0],
                                       operands[1].shape().dims()[1]};
  return std::vector<ir::TensorType>{ir::TensorType(operands[0].dtype(), ir::Shape(dims))};
}

} // namespace

Result<ir::Shape> broadcast_shapes(const ir::Shape& a, const ir::Shape& b) {
  const std::size_t rank = std::max(a.rank(), b.rank());
  std::vector<std::int64_t> dims(rank, 1);

  for (std::size_t i = 0; i < rank; ++i) {
    // Walking from the back, so a missing leading dimension on the shorter shape reads as 1.
    const std::size_t from_end = rank - i;
    const std::int64_t lhs = a.rank() >= from_end ? a.dims()[a.rank() - from_end] : 1;
    const std::int64_t rhs = b.rank() >= from_end ? b.dims()[b.rank() - from_end] : 1;

    if (lhs == ir::kDynamic || rhs == ir::kDynamic) {
      dims[i] = ir::kDynamic;
      continue;
    }
    if (lhs != rhs && lhs != 1 && rhs != 1) {
      return invalid_argument("cannot broadcast " + a.to_string() + " with " + b.to_string() +
                              ": dimension " + std::to_string(i) + " is " + std::to_string(lhs) +
                              " against " + std::to_string(rhs));
    }
    dims[i] = std::max(lhs, rhs);
  }

  return ir::Shape(std::move(dims));
}

bool has_shape_rule(std::string_view op) { return find_rule(op).has_value(); }

Result<std::vector<std::size_t>>
reduced_axes(std::string_view op, const ir::AttributeMap& attributes, std::size_t rank) {
  const ir::Attribute* dims = attribute_or_null(attributes, "arg1");

  std::vector<std::int64_t> requested;
  if (dims != nullptr) {
    if (const auto* list = std::get_if<std::vector<std::int64_t>>(dims)) {
      requested = *list;
    } else if (const auto* single = std::get_if<std::int64_t>(dims)) {
      requested = {*single};
    } else {
      return invalid_argument(std::string(op) +
                              " takes its axes as an integer or a list of integers");
    }
  }

  // Built with push_back rather than resized and indexed. A rank-0 tensor gives an empty vector
  // whose data pointer may be null, and GCC cannot see that the loop guard prevents the write.
  std::vector<std::size_t> axes;
  axes.reserve(rank);
  if (requested.empty()) {
    for (std::size_t i = 0; i < rank; ++i) {
      axes.push_back(i);
    }
    return axes;
  }

  const auto signed_rank = static_cast<std::int64_t>(rank);
  for (const std::int64_t dim : requested) {
    // Negative indexes count from the end, which is how torch writes the last axis.
    const std::int64_t resolved = dim < 0 ? dim + signed_rank : dim;
    if (resolved < 0 || resolved >= signed_rank) {
      return invalid_argument(std::string(op) + " reduces dimension " + std::to_string(dim) +
                              " of a rank-" + std::to_string(rank) + " tensor");
    }
    axes.push_back(static_cast<std::size_t>(resolved));
  }

  std::sort(axes.begin(), axes.end());
  if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
    return invalid_argument(std::string(op) + " reduces the same dimension twice");
  }
  return axes;
}

Result<std::vector<ir::TensorType>> infer_result_types(std::string_view op,
                                                       const std::vector<ir::TensorType>& operands,
                                                       const ir::AttributeMap& attributes) {
  const std::optional<RuleKind> rule = find_rule(op);
  TESSIQ_CHECK(rule.has_value(),
               "infer_result_types for '" + std::string(op) + "', which has no rule");

  switch (*rule) {
  case RuleKind::kUnaryElementwise:
    return infer_unary(op, operands);
  case RuleKind::kBinaryElementwise:
    return infer_binary(op, operands);
  case RuleKind::kReduction:
    return infer_reduction(op, operands, attributes);
  case RuleKind::kMatmul:
    return infer_matmul(op, operands);
  }

  TESSIQ_UNREACHABLE("unhandled RuleKind");
}

} // namespace tessiq::passes
