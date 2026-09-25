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
/// Precondition: values are in SSA form, so a name identifies one definition for the whole graph.
/// Postcondition: no two remaining nodes have the same op, operands, attributes and result types.
///
/// Legality: two nodes with the same op, the same operands and the same attributes compute the
/// same function of the same inputs, so either result can stand for the other. This depends on ops
/// being pure. The dead duplicate is left for dead-node-elimination rather than erased here, which
/// keeps this pass to one job and means a duplicate whose results are still referenced elsewhere is
/// not lost.
class CommonSubexpressionElimination final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "cse"; }

  [[nodiscard]] std::string_view description() const override {
    return "replace repeated computations with the first one";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;
    std::unordered_map<std::string, ir::Node*> seen;

    // Forwards, so the node kept is the earliest one, which keeps every use dominated by its
    // definition without reordering anything.
    for (ir::Node* node : graph.nodes()) {
      const std::string key = signature(node);
      const auto it = seen.find(key);
      if (it == seen.end()) {
        seen.emplace(key, node);
        continue;
      }

      ir::Node* original = it->second;
      for (std::size_t i = 0; i < node->result_count(); ++i) {
        // Only a replacement that moved a use counts as a change. The duplicate node is left for
        // dead-node-elimination, so a second run sees it again; reporting a change for it every
        // time would keep a fixpoint loop spinning on a graph that has stopped moving.
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

  /// Everything that decides whether two nodes compute the same thing.
  ///
  /// Operand identity comes from the defining value's name, which SSA makes unique. Result types
  /// are included because two nodes can share an op and operands and still differ, once shape
  /// inference has annotated them differently.
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
