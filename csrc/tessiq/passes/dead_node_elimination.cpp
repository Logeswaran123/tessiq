// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

/// Removes nodes whose results nothing reads.
///
/// Precondition: use lists agree with operands, which the verifier checks.
/// Postcondition: every remaining node has at least one result that is used or returned.
///
/// Legality: removing a node that produces nothing anyone reads cannot change what the graph
/// computes. This holds only because the IR has no side-effecting ops. When one arrives it needs
/// an effects marker and this pass has to consult it, or it will delete the store nobody reads
/// the result of.
class DeadNodeElimination final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "dead-node-elimination"; }

  [[nodiscard]] std::string_view description() const override {
    return "remove nodes whose results are never used";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;

    // Backwards, so a chain of dead nodes goes in one sweep: erasing a consumer is what makes its
    // producer dead. Forwards would need iterating to a fixpoint.
    const std::vector<ir::Node*> nodes = graph.nodes();
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      ir::Node* node = *it;
      if (is_live(graph, node)) {
        continue;
      }
      TESSIQ_RETURN_IF_ERROR(graph.erase(node));
      changed = true;
    }

    return changed;
  }

 private:
  /// A node is live if any result is used by another node or returned by the graph.
  static bool is_live(const ir::Graph& graph, const ir::Node* node) {
    for (const ir::Value* result : node->results()) {
      if (result->has_users()) {
        return true;
      }
      const auto& results = graph.results();
      if (std::find(results.begin(), results.end(), result) != results.end()) {
        return true;
      }
    }
    // A node with no results at all is kept: it produces nothing, so liveness says nothing about
    // it, and it is more likely to be an effect this pass must not touch.
    return node->result_count() == 0;
  }
};

TESSIQ_REGISTER_PASS(DeadNodeElimination)

} // namespace

} // namespace tessiq::passes
