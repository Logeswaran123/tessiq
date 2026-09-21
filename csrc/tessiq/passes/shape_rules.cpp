// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/shape_rules.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/macros.h"

namespace tessiq::passes {

namespace {

/// How an op derives its result type from its operands.
enum class RuleKind {
  /// One tensor in, same type out: abs, relu, tanh.
  kUnaryElementwise,
  /// Two tensors in, broadcast shape and matching dtype out: add, mul.
  ///
  /// Also covers the one-operand form, which is what the importer produces when the second
  /// argument was a Python scalar and became an attribute instead.
  kBinaryElementwise,
  /// Matrix multiply: [m,k] by [k,n] gives [m,n].
  kMatmul,
};

/// The rule for `op`, derived from the generated operation table, or nothing if it has none.
///
/// There is no second list here. A rule kind is a function of what ops.toml already says: an
/// elementwise operation takes its operands' type through unchanged, and how many operands it has
/// is its arity. Keeping a separate table meant every operation added to ops.toml also had to be
/// added here, and an operation that fused correctly but had no shape rule was the failure that
/// produced.
std::optional<RuleKind> find_rule(std::string_view op) {
  const OpInfo* info = find_op(op);
  if (info == nullptr) {
    return std::nullopt;
  }
  switch (info->category) {
  case OpCategory::kElementwise:
    return info->arity == 1 ? RuleKind::kUnaryElementwise : RuleKind::kBinaryElementwise;
  case OpCategory::kMatmul:
    return RuleKind::kMatmul;
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
  if (operands.size() != 2) {
    return invalid_argument(std::string(op) + " takes one or two tensor operands, got " +
                            std::to_string(operands.size()));
  }

  if (operands[0].dtype() != operands[1].dtype()) {
    // Type promotion is torch's own set of rules, and a second copy of them here would be a second
    // copy to keep in agreement. A graph out of AOTAutograd has already been promoted, so mixed
    // dtypes mean something upstream is not what this pass expects.
    return unsupported(std::string(op) + " on mixed dtypes " + operands[0].to_string() + " and " +
                       operands[1].to_string() + "; promotion happens before import");
  }

  TESSIQ_ASSIGN_OR_RETURN(ir::Shape shape,
                          broadcast_shapes(operands[0].shape(), operands[1].shape()));
  return std::vector<ir::TensorType>{ir::TensorType(operands[0].dtype(), std::move(shape))};
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

Result<std::vector<ir::TensorType>>
infer_result_types(std::string_view op, const std::vector<ir::TensorType>& operands) {
  const std::optional<RuleKind> rule = find_rule(op);
  TESSIQ_CHECK(rule.has_value(),
               "infer_result_types for '" + std::string(op) + "', which has no rule");

  switch (*rule) {
  case RuleKind::kUnaryElementwise:
    return infer_unary(op, operands);
  case RuleKind::kBinaryElementwise:
    return infer_binary(op, operands);
  case RuleKind::kMatmul:
    return infer_matmul(op, operands);
  }

  TESSIQ_UNREACHABLE("unhandled RuleKind");
}

} // namespace tessiq::passes
