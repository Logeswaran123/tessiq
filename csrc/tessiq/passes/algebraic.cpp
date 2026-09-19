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
/// Integer and floating attributes both answer, because a graph may spell one as the other and a
/// rewrite that only recognised its own spelling would fire unpredictably. The comparison is exact:
/// 0 and 1 are representable in every dtype the IR carries, so there is no tolerance to choose.
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
/// Imported graphs carry `aten.mul.Tensor`; graphs written by hand for a test carry `mul`. Both
/// are matched without building a string, because this runs once per identity per node.
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

/// One table, so an identity cannot be half-added.
///
/// Multiplying or dividing by one is exact for every value a float can hold, including the
/// infinities, both zeros and every NaN, because the result is the operand's own bit pattern.
///
/// Adding or subtracting zero is not. `(-0.0) + 0.0` is `+0.0`, so the rewrite would change the
/// sign of a zero, and a value that compares equal is not the same value once something divides
/// by it. Integers have one zero and no such trap, so the additive identities are theirs alone.
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
/// Every rewrite here replaces the node's result with one of its own operands, so the two must
/// have the same type. A broadcast makes them differ: `add(%x: i32[1], %zeros: i32[4])` produces
/// `i32[4]`, and `%x` cannot stand in for it because the zeros are what gave it its shape.
ir::Value* identity_operand(const ir::Node* node) {
  if (node->result_count() != 1) {
    return nullptr;
  }
  const ir::TensorType& result = node->result(0)->type();
  const Identity* identity = find_identity(node, ir::is_float(result.dtype()));
  if (identity == nullptr) {
    return nullptr;
  }

  // The one-operand form, where the constant arrived as an attribute rather than an operand.
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
/// Two cases, both exact. Negation is its own inverse, so `neg(neg(x))` is `x` down to the bit
/// pattern. `relu` and `abs` are idempotent, so applying either twice gives what applying it once
/// already gave.
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

/// Replaces operations whose result is already available with the value that holds it.
///
/// Precondition: none.
/// Postcondition: no node matches one of the identities below.
///
/// Legality is per identity and each one is argued where it is applied. The rule running through
/// all of them is that a rewrite must be exact rather than close: an identity that holds for
/// ordinary values and not for a signed zero, an infinity or a NaN is not an identity, and the
/// graph is not the place to decide that those cases do not matter.
///
/// The simplified node is left in place with no users, as cse does, so dce removes it and a
/// partially simplified graph is valid at every step.
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

      // Only a replacement that moved a use counts as a change. The simplified node stays until
      // dce runs, so a second pass finds it again, and reporting a change every time would keep a
      // fixpoint loop spinning on a graph that has stopped moving.
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
