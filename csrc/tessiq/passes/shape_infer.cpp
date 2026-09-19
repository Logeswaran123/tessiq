// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <string>
#include <vector>

#include "tessiq/ir/types.h"
#include "tessiq/passes/pass.h"
#include "tessiq/passes/shape_rules.h"

namespace tessiq::passes {

namespace {

/// True when `recorded` says everything `inferred` says, or less.
///
/// A recorded dimension of `?` is a claim of ignorance, so replacing it with a known size is a
/// refinement. A recorded dimension that disagrees with a known one is a contradiction, and the
/// two cannot both describe the same value.
bool refines(const ir::TensorType& recorded, const ir::TensorType& inferred) {
  if (recorded.dtype() != inferred.dtype()) {
    return false;
  }
  if (recorded.shape().rank() != inferred.shape().rank()) {
    return false;
  }
  for (std::size_t i = 0; i < recorded.shape().rank(); ++i) {
    const std::int64_t have = recorded.shape().dims()[i];
    const std::int64_t want = inferred.shape().dims()[i];
    if (have != want && have != ir::kDynamic && want != ir::kDynamic) {
      return false;
    }
  }
  return true;
}

/// Combines what the graph recorded with what the rule derived, keeping the more precise of each
/// dimension. A rule that cannot determine a dimension must not erase one the graph already knew.
ir::TensorType merge(const ir::TensorType& recorded, const ir::TensorType& inferred) {
  std::vector<std::int64_t> dims;
  dims.reserve(recorded.shape().rank());
  for (std::size_t i = 0; i < recorded.shape().rank(); ++i) {
    const std::int64_t have = recorded.shape().dims()[i];
    dims.push_back(have == ir::kDynamic ? inferred.shape().dims()[i] : have);
  }
  return {recorded.dtype(), ir::Shape(std::move(dims))};
}

/// Derives result types from operand types, and checks the ones already recorded.
///
/// Precondition: none. A graph whose types are already complete is left untouched.
/// Postcondition: every node with a shape rule has result types consistent with its operands, and
/// no dimension is dynamic where the rule could determine it.
///
/// Ops with no rule are skipped rather than rejected. The importer records a type for every node
/// from the tracing metadata, so an op this pass has never heard of still carries a usable type,
/// and refusing the graph would make the pass unusable on any real model.
class ShapeInference final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "shape-infer"; }

  [[nodiscard]] std::string_view description() const override {
    return "propagate shapes and dtypes, and check the ones already recorded";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;

    for (ir::Node* node : graph.nodes()) {
      if (!has_shape_rule(node->op())) {
        continue;
      }

      std::vector<ir::TensorType> operands;
      operands.reserve(node->operand_count());
      for (std::size_t i = 0; i < node->operand_count(); ++i) {
        operands.push_back(node->operand(i)->type());
      }

      TESSIQ_ASSIGN_OR_RETURN(const auto inferred, infer_result_types(node->op(), operands));

      if (inferred.size() != node->result_count()) {
        return invalid_argument("'" + node->op() + "' produces " + std::to_string(inferred.size()) +
                                " result(s) but the graph has " +
                                std::to_string(node->result_count()));
      }

      for (std::size_t i = 0; i < inferred.size(); ++i) {
        ir::Value* result = node->result(i);
        const ir::TensorType& derived = inferred[i];

        if (!refines(result->type(), derived)) {
          return invalid_argument("'" + node->op() + "' result %" + result->name() +
                                  " is recorded as " + result->type().to_string() +
                                  " but its operands give " + derived.to_string());
        }

        ir::TensorType merged = merge(result->type(), derived);
        if (merged != result->type()) {
          result->set_type(std::move(merged));
          changed = true;
        }
      }
    }

    return changed;
  }
};

TESSIQ_REGISTER_PASS(ShapeInference)

} // namespace

} // namespace tessiq::passes
