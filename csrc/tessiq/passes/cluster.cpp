// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tessiq/passes/op_table.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

constexpr std::string_view kSupported = "tessiq.supported";
constexpr std::string_view kCluster = "tessiq.cluster";

/// Groups neighbouring supported nodes into clusters the backend can take as a unit.
///
/// Precondition: `mark-supported` has run, so every node carries `tessiq.supported`.
/// Postcondition: every supported node carries `tessiq.cluster`, except those in a unit that only
/// moves data (see `drop_units_that_only_move`).
///
/// A cluster runs as one unit, so the graph of clusters and leftover nodes has to stay acyclic.
/// That is why this is not a union-find over edges. Consider:
///
///     %a = supported(...)
///     %b = unsupported(%a)
///     %c = supported(%a)
///     %d = supported(%b, %c)
///
/// Merging along edges puts `a`, `c` and `d` in one cluster. `b` reads from that cluster and `d`
/// reads `b`, so the cluster would have to run both before and after `b`, and no order works.
///
/// So each merge is tried and kept only if the unit graph is still acyclic. That is a traversal
/// per candidate merge, which is cheap at the sizes models produce.
class Cluster final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "cluster"; }

  [[nodiscard]] std::string_view description() const override {
    return "group neighbouring supported nodes into clusters";
  }

  Result<bool> run(ir::Graph& graph) override {
    const std::vector<ir::Node*> nodes = graph.nodes();
    TESSIQ_ASSIGN_OR_RETURN(auto supported, read_support(nodes));

    // Every node starts in its own unit. Unsupported nodes are never merged but still take part
    // in the cycle check.
    std::unordered_map<const ir::Node*, std::size_t> unit;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      unit[nodes[i]] = i;
    }

    // Merging two units can make another merge legal, so repeat until nothing changes.
    while (merge_one_round(nodes, supported, unit)) {
    }

    drop_units_that_only_move(nodes, supported, unit);
    return assign(nodes, unit, supported);
  }

 private:
  using SupportMap = std::unordered_map<const ir::Node*, bool>;

  /// Reads what mark-supported recorded, failing if it has not run.
  static Result<SupportMap> read_support(const std::vector<ir::Node*>& nodes) {
    SupportMap supported;
    for (const ir::Node* node : nodes) {
      const ir::Attribute* attribute = node->attribute(kSupported);
      const bool* value = attribute == nullptr ? nullptr : std::get_if<bool>(attribute);
      if (value == nullptr) {
        return invalid_argument("node '" + node->op() + "' has no " + std::string(kSupported) +
                                " attribute; run mark-supported before cluster");
      }
      supported[node] = *value;
    }
    return supported;
  }

  /// Tries every supported edge once. Returns whether anything merged.
  static bool merge_one_round(const std::vector<ir::Node*>& nodes,
                              const SupportMap& supported,
                              std::unordered_map<const ir::Node*, std::size_t>& unit) {
    bool merged = false;
    for (const ir::Node* consumer : nodes) {
      if (!supported.at(consumer)) {
        continue;
      }
      for (const ir::Value* operand : consumer->operands()) {
        const ir::Node* producer = operand->producer();
        const bool joinable =
            producer != nullptr && supported.at(producer) && unit.at(producer) != unit.at(consumer);
        if (joinable && try_merge(nodes, unit, unit.at(producer), unit.at(consumer))) {
          merged = true;
        }
      }
    }
    return merged;
  }

  /// Merges `from` into `into`, keeping the merge only if the unit graph stays acyclic.
  static bool try_merge(const std::vector<ir::Node*>& nodes,
                        std::unordered_map<const ir::Node*, std::size_t>& unit,
                        std::size_t into,
                        std::size_t from) {
    const std::unordered_map<const ir::Node*, std::size_t> before = unit;
    for (auto& [node, id] : unit) {
      if (id == from) {
        id = into;
      }
    }
    if (has_cycle(nodes, unit)) {
      unit = before;
      return false;
    }
    return true;
  }

  /// True if the graph over units contains a cycle.
  static bool has_cycle(const std::vector<ir::Node*>& nodes,
                        const std::unordered_map<const ir::Node*, std::size_t>& unit) {
    std::unordered_map<std::size_t, std::unordered_set<std::size_t>> edges;
    for (const ir::Node* consumer : nodes) {
      const std::size_t to = unit.at(consumer);
      for (const ir::Value* operand : consumer->operands()) {
        const ir::Node* producer = operand->producer();
        if (producer == nullptr) {
          continue;
        }
        const std::size_t from = unit.at(producer);
        if (from != to) {
          edges[from].insert(to);
        }
      }
    }

    std::unordered_set<std::size_t> visiting;
    std::unordered_set<std::size_t> done;
    for (const auto& [id, targets] : edges) {
      if (done.count(id) == 0 && visit(id, edges, visiting, done)) {
        return true;
      }
    }
    return false;
  }

  /// Depth-first search. True if a back edge is found, which is a cycle.
  static bool visit(std::size_t id,
                    const std::unordered_map<std::size_t, std::unordered_set<std::size_t>>& edges,
                    std::unordered_set<std::size_t>& visiting,
                    std::unordered_set<std::size_t>& done) {
    if (visiting.count(id) != 0) {
      return true;
    }
    if (done.count(id) != 0) {
      return false;
    }

    visiting.insert(id);
    if (const auto it = edges.find(id); it != edges.end()) {
      for (const std::size_t next : it->second) {
        if (visit(next, edges, visiting, done)) {
          return true;
        }
      }
    }
    visiting.erase(id);
    done.insert(id);
    return false;
  }

  /// Leaves a unit on the host when every operation in it only rearranges indices.
  ///
  /// A movement operation is a view on either side, so it is worth a transfer only when something
  /// in the same unit computes. A unit of nothing but views would copy its input to the device,
  /// take a view, and copy it back. Movement operations often sit between operations the backend
  /// cannot run, so claiming them alone turns each one into a pair of transfers.
  static void
  drop_units_that_only_move(const std::vector<ir::Node*>& nodes,
                            SupportMap& supported,
                            const std::unordered_map<const ir::Node*, std::size_t>& unit) {
    std::unordered_map<std::size_t, bool> computes;
    for (const ir::Node* node : nodes) {
      if (!supported.at(node)) {
        continue;
      }
      const OpInfo* info = find_op(node->op());
      const bool moves_only = info != nullptr && info->category == OpCategory::kMovement;
      computes[unit.at(node)] = computes[unit.at(node)] || !moves_only;
    }

    for (const ir::Node* node : nodes) {
      if (supported.at(node) && !computes[unit.at(node)]) {
        supported[node] = false;
      }
    }
  }

  /// Writes `tessiq.cluster`, numbering clusters in the order they first appear so the output is
  /// the same every run.
  static Result<bool> assign(const std::vector<ir::Node*>& nodes,
                             const std::unordered_map<const ir::Node*, std::size_t>& unit,
                             const SupportMap& supported) {
    std::unordered_map<std::size_t, std::int64_t> numbering;
    bool changed = false;

    for (ir::Node* node : nodes) {
      if (!supported.at(node)) {
        continue;
      }
      const std::size_t id = unit.at(node);
      const auto [it, inserted] =
          numbering.emplace(id, static_cast<std::int64_t>(numbering.size()));

      const ir::Attribute* existing = node->attribute(kCluster);
      const std::int64_t* current =
          existing == nullptr ? nullptr : std::get_if<std::int64_t>(existing);
      if (current != nullptr && *current == it->second) {
        continue;
      }
      node->set_attribute(std::string(kCluster), it->second);
      changed = true;
    }

    return changed;
  }
};

TESSIQ_REGISTER_PASS(Cluster)

} // namespace

} // namespace tessiq::passes
