// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

/// Replaces a node with an earlier identical one.
///
/// Precondition: values are in SSA form, so a name identifies one definition.
/// Postcondition: no two nodes that are still used have the same op, operands, attributes and
/// result types.
///
/// Legality: two nodes with the same op, operands and attributes compute the same thing, provided
/// the op is pure. The duplicate is left with no users for dead-node-elimination to remove.
class CommonSubexpressionElimination final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "cse"; }

  [[nodiscard]] std::string_view description() const override {
    return "replace repeated computations with the first one";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;
    std::unordered_map<std::string, ir::Node*> seen;

    // Forwards, so the earliest node is kept and every use still follows its definition.
    for (ir::Node* node : graph.nodes()) {
      const std::string key = signature(node);
      const auto it = seen.find(key);
      if (it == seen.end()) {
        seen.emplace(key, node);
        continue;
      }

      ir::Node* original = it->second;
      for (std::size_t i = 0; i < node->result_count(); ++i) {
        // A change only if it moved a use. The duplicate stays until dead-node-elimination, so
        // running this again finds it again and must not report a change.
        if (is_used(graph, node->result(i))) {
          changed = true;
        }
        TESSIQ_RETURN_IF_ERROR(graph.replace_all_uses(node->result(i), original->result(i)));
      }
    }

    return changed;
  }

 private:
  /// True if anything still reads `value`, counting the graph's own results as readers.
  static bool is_used(const ir::Graph& graph, const ir::Value* value) {
    if (value->has_users()) {
      return true;
    }
    const auto& results = graph.results();
    return std::find(results.begin(), results.end(), value) != results.end();
  }

  /// Everything that decides whether two nodes compute the same thing. Operands are keyed by name,
  /// which SSA makes unique. Result types are included so that nodes typed differently are never
  /// merged.
  static std::string signature(const ir::Node* node) {
    std::string key = node->op();
    key += '(';
    for (const ir::Value* operand : node->operands()) {
      key += operand->name();
      key += ',';
    }
    key += "){";
    for (const auto& [name, value] : node->attributes()) {
      key += name;
      key += '=';
      key += ir::attribute_to_string(value);
      key += ',';
    }
    key += "}:";
    for (const ir::Value* result : node->results()) {
      key += result->type().to_string();
      key += ',';
    }
    return key;
  }
};

TESSIQ_REGISTER_PASS(CommonSubexpressionElimination)

} // namespace

} // namespace tessiq::passes
