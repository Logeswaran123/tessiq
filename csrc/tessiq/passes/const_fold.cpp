// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

/// Evaluates scalar arithmetic on constants at compile time.
///
/// Precondition: `const` nodes carry a `value` attribute matching their result type.
/// Postcondition: no `add`, `sub` or `mul` node that is still used has all-constant scalar
/// operands.
///
/// Two limits:
///
///  - Scalars only. Folding a tensor would put its contents in the IR as an attribute, which needs
///    a constant pool first.
///  - Only `add`, `sub` and `mul`. They are computed in double, and rounding that to a narrower
///    float type gives the same answer as computing in it. An operation where host and device can
///    round differently, such as a fused multiply-add or a transcendental, is not folded.
class ConstantFolding final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "const-fold"; }

  [[nodiscard]] std::string_view description() const override {
    return "evaluate scalar arithmetic on constants at compile time";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;

    for (ir::Node* node : graph.nodes()) {
      if (node->op() != "add" && node->op() != "sub" && node->op() != "mul") {
        continue;
      }
      if (node->operand_count() != 2 || node->result_count() != 1) {
        continue;
      }

      const ir::TensorType& type = node->result(0)->type();
      if (!type.shape().is_scalar()) {
        continue;
      }

      const ir::Value* lhs = node->operand(0);
      const ir::Value* rhs = node->operand(1);
      if (!is_const(lhs) || !is_const(rhs)) {
        continue;
      }

      // Before the node it replaces, since a definition has to come before its uses.
      ir::Node* folded = graph.insert_node("const", node);
      if (ir::is_float(type.dtype())) {
        const double result = apply(node->op(), as_double(lhs), as_double(rhs));
        folded->set_attribute("value", result);
      } else {
        const auto result = static_cast<std::int64_t>(
            apply(node->op(), static_cast<double>(as_int(lhs)), static_cast<double>(as_int(rhs))));
        folded->set_attribute("value", result);
      }
      ir::Value* value =
          folded->append_result(graph.unique_name(node->result(0)->name() + "_f"), type);

      TESSIQ_RETURN_IF_ERROR(graph.replace_all_uses(node->result(0), value));
      changed = true;
    }

    // The folded nodes are left with no users for dead-node-elimination.
    return changed;
  }

 private:
  static bool is_const(const ir::Value* value) {
    const ir::Node* producer = value->producer();
    return producer != nullptr && producer->op() == "const" &&
           producer->attribute("value") != nullptr;
  }

  static double as_double(const ir::Value* value) {
    const ir::Attribute& attribute = *value->producer()->attribute("value");
    if (const auto* held = std::get_if<double>(&attribute)) {
      return *held;
    }
    if (const auto* held = std::get_if<std::int64_t>(&attribute)) {
      return static_cast<double>(*held);
    }
    return 0.0;
  }

  static std::int64_t as_int(const ir::Value* value) {
    const ir::Attribute& attribute = *value->producer()->attribute("value");
    if (const auto* held = std::get_if<std::int64_t>(&attribute)) {
      return *held;
    }
    if (const auto* held = std::get_if<double>(&attribute)) {
      return static_cast<std::int64_t>(*held);
    }
    return 0;
  }

  static double apply(const std::string& op, double lhs, double rhs) {
    if (op == "add") {
      return lhs + rhs;
    }
    if (op == "sub") {
      return lhs - rhs;
    }
    return lhs * rhs;
  }
};

TESSIQ_REGISTER_PASS(ConstantFolding)

} // namespace

} // namespace tessiq::passes
