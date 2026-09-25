// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "tessiq/ir/types.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

/// Whether `value` is produced by a `const` node holding exactly `target`.
///
/// Matches an integer or a float attribute, since a graph may spell the constant either way. The
/// comparison is exact; 0 and 1 are representable in every dtype.
bool is_constant(const ir::Value* value, std::int64_t target) {
  const ir::Node* producer = value->producer();
  if (producer == nullptr || producer->op() != "const") {
    return false;
  }
  const ir::Attribute* attribute = producer->attribute("value");
  if (attribute == nullptr) {
    return false;
  }
  if (const auto* held = std::get_if<std::int64_t>(attribute)) {
    return *held == target;
  }
  if (const auto* held = std::get_if<double>(attribute)) {
    return *held == static_cast<double>(target);
  }
  return false;
}

/// Whether the op's second argument is the scalar constant `target`, carried as an attribute.
///
/// The importer turns a non-tensor argument into an attribute keyed by position, so `x * 1.0`
/// arrives as a one-operand node with `arg1 = 1.0` rather than as two operands.
bool has_scalar_argument(const ir::Node* node, std::int64_t target) {
  if (node->operand_count() != 1) {
    return false;
  }
  const ir::Attribute* attribute = node->attribute("arg1");
  if (attribute == nullptr) {
    return false;
  }
  if (const auto* held = std::get_if<std::int64_t>(attribute)) {
    return *held == target;
  }
  if (const auto* held = std::get_if<double>(attribute)) {
    return *held == static_cast<double>(target);
  }
  return false;
}

/// Whether `node` is the named operation, under either spelling.
///
/// Imported graphs carry `aten.mul.Tensor` and hand-written test graphs carry `mul`.
bool is_op(const ir::Node* node, std::string_view name) {
  std::string_view op = node->op();
  if (op == name) {
    return true;
  }
  constexpr std::string_view kPrefix = "aten.";
  if (op.substr(0, kPrefix.size()) != kPrefix) {
    return false;
  }
  op.remove_prefix(kPrefix.size());
  if (op.substr(0, name.size()) != name) {
    return false;
  }
  op.remove_prefix(name.size());
  return op == ".Tensor" || op == ".default" || op == ".Scalar";
}

/// One algebraic identity.
struct Identity {
  std::string_view op;
  /// The value the other side must hold for the identity to apply.
  std::int64_t constant;
  /// Whether the constant may sit on either side. `1/x` is not `x` and `0-x` is not `x`, so
  /// division and subtraction match only the right-hand form.
  bool commutative;
  /// Whether the rewrite is exact for floating point as well as integers.
  bool exact_for_floats;
};

/// Multiplying or dividing by one is exact for every float, including infinities, signed zeros and
/// NaNs.
///
/// Adding or subtracting zero is not, because the match accepts either zero: `-0.0 + 0.0` is
/// `+0.0`, and `x - (-0.0)` is the same sum. The sign of a zero shows once something divides by it,
/// so the additive identities apply to integers only.
constexpr std::array<Identity, 4> kIdentities{{
    {"mul", 1, true, true},
    {"div", 1, false, true},
    {"add", 0, true, false},
    {"sub", 0, false, false},
}};

const Identity* find_identity(const ir::Node* node, bool is_float_result) {
  for (const Identity& identity : kIdentities) {
    if (!is_op(node, identity.op)) {
      continue;
    }
    return is_float_result && !identity.exact_for_floats ? nullptr : &identity;
  }
  return nullptr;
}

/// The operand this node could collapse to, or null when no identity applies.
///
/// The operand has to have the result's type. A broadcast makes them differ:
/// `add(%x: i32[1], %zeros: i32[4])` produces `i32[4]`, and `%x` cannot stand in for it.
ir::Value* identity_operand(const ir::Node* node) {
  if (node->result_count() != 1) {
    return nullptr;
  }
  const ir::TensorType& result = node->result(0)->type();
  const Identity* identity = find_identity(node, ir::is_float(result.dtype()));
  if (identity == nullptr) {
    return nullptr;
  }

  // The constant arrived as an attribute rather than an operand.
  if (node->operand_count() == 1) {
    ir::Value* only = node->operand(0);
    const bool applies = has_scalar_argument(node, identity->constant) && only->type() == result;
    return applies ? only : nullptr;
  }
  if (node->operand_count() != 2) {
    return nullptr;
  }

  ir::Value* lhs = node->operand(0);
  ir::Value* rhs = node->operand(1);
  if (is_constant(rhs, identity->constant) && lhs->type() == result) {
    return lhs;
  }
  if (identity->commutative && is_constant(lhs, identity->constant) && rhs->type() == result) {
    return rhs;
  }
  return nullptr;
}

/// The operand of a unary node that makes it redundant, or null.
///
/// Both cases are exact: `neg(neg(x))` is `x` bit for bit, and `relu` and `abs` are idempotent.
ir::Value* redundant_unary(const ir::Node* node) {
  if (node->operand_count() != 1 || node->result_count() != 1) {
    return nullptr;
  }
  ir::Node* producer = node->operand(0)->producer();
  if (producer == nullptr || producer->result_count() != 1) {
    return nullptr;
  }
  if (node->operand(0)->type() != node->result(0)->type()) {
    return nullptr;
  }

  if (is_op(node, "neg") && is_op(producer, "neg")) {
    ir::Value* inner = producer->operand(0);
    return inner->type() == node->result(0)->type() ? inner : nullptr;
  }
  if ((is_op(node, "relu") && is_op(producer, "relu")) ||
      (is_op(node, "abs") && is_op(producer, "abs"))) {
    return node->operand(0);
  }
  return nullptr;
}

/// Replaces an operation with one of its operands when an identity says they are equal.
///
/// Precondition: none.
/// Postcondition: no node that is still used matches one of the identities above.
///
/// Every identity must be exact, including for signed zeros, infinities and NaNs. The replaced
/// node is left with no users for dead-node-elimination to remove.
class AlgebraicSimplification final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "algebraic-simplify"; }

  [[nodiscard]] std::string_view description() const override {
    return "replace operations whose result is already available";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;

    for (ir::Node* node : graph.nodes()) {
      ir::Value* replacement = identity_operand(node);
      if (replacement == nullptr) {
        replacement = redundant_unary(node);
      }
      if (replacement == nullptr) {
        continue;
      }

      // A change only if it moved a use. The replaced node stays until dead-node-elimination, so
      // running this again finds it again and must not report a change.
      if (is_used(graph, node->result(0))) {
        changed = true;
      }
      TESSIQ_RETURN_IF_ERROR(graph.replace_all_uses(node->result(0), replacement));
    }

    return changed;
  }

 private:
  static bool is_used(const ir::Graph& graph, const ir::Value* value) {
    if (value->has_users()) {
      return true;
    }
    const auto& results = graph.results();
    return std::find(results.begin(), results.end(), value) != results.end();
  }
};

TESSIQ_REGISTER_PASS(AlgebraicSimplification)

} // namespace

} // namespace tessiq::passes
