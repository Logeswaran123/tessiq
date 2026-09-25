// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tessiq/codegen/kernel.h"
#include "tessiq/ir/graph.h"
#include "tessiq/passes/op_traits.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

constexpr std::string_view kCluster = "tessiq.cluster";
constexpr std::string_view kFusion = "tessiq.fusion";

/// The cluster recorded on a node, or nothing when it carries none.
const std::int64_t* cluster_of(const ir::Node* node) {
  const ir::Attribute* attribute = node->attribute(kCluster);
  return attribute == nullptr ? nullptr : std::get_if<std::int64_t>(attribute);
}

/// Whether a loop over `node`'s result can read every one of its operands.
///
/// Most operands have the result's shape and are read at the loop index. A smaller one, such as
/// the per-row mean in `x - mean` from softmax and layer norm, is read at an index the emitter
/// derives from the shapes. This asks the emitter through `operand_read` instead of repeating its
/// rules, because a group the emitter then refused would silently get no kernel.
bool every_operand_is_reachable(const ir::Node* node) {
  const ir::TensorType& result = node->result(0)->type();
  for (const ir::Value* operand : node->operands()) {
    if (operand == nullptr || !codegen::operand_read(result, operand->type()).has_value()) {
      return false;
    }
  }
  return true;
}

/// Whether `producer`'s result can stay in a register for `consumer` instead of going through
/// memory. Both must:
///
///  - be elementwise with one result, so they cover the same positions;
///  - have every operand reachable from the loop;
///  - be in the same cluster, since a cluster is one unit of execution;
///  - have the same result type. With broadcasting they would write different numbers of
///    elements, and one loop cannot serve both.
///
/// The value must also have exactly one reader and not be a graph result. Anything else reading it
/// needs it in memory, so it would be written out regardless.
bool can_fuse(const ir::Graph& graph, const ir::Node* producer, const ir::Node* consumer) {
  if (!is_elementwise(producer->op()) || !is_elementwise(consumer->op())) {
    return false;
  }
  if (producer->result_count() != 1 || consumer->result_count() != 1) {
    return false;
  }
  if (!every_operand_is_reachable(producer) || !every_operand_is_reachable(consumer)) {
    return false;
  }

  const std::int64_t* from = cluster_of(producer);
  const std::int64_t* into = cluster_of(consumer);
  if (from == nullptr || into == nullptr || *from != *into) {
    return false;
  }

  const ir::Value* value = producer->result(0);
  if (value->users().size() != 1) {
    return false;
  }
  if (value->type() != consumer->result(0)->type()) {
    return false;
  }

  const auto& results = graph.results();
  return std::find(results.begin(), results.end(), value) == results.end();
}

/// Whether a node left in a group of one should still get a kernel: an elementwise node in a
/// cluster whose operands the loop can reach. Otherwise it runs through the fallback, which copies
/// its operands to the host and its result back.
bool runs_alone_as_a_kernel(const ir::Node* node) {
  if (!is_elementwise(node->op()) || node->result_count() != 1) {
    return false;
  }
  if (cluster_of(node) == nullptr) {
    return false;
  }
  return every_operand_is_reachable(node);
}

/// Groups chains of elementwise operations that can run as one kernel.
///
/// Precondition: `cluster` has run, so the nodes the backend claims carry `tessiq.cluster`.
/// Postcondition: every node that becomes a kernel carries `tessiq.fusion`, including a group of
/// one. A node without it has no kernel and runs through the fallback.
///
/// The conditions on each edge are in `can_fuse`. Unlike clustering, merging cannot create a
/// cycle, so a union-find is enough. Every node in a group but the last has its only reader inside
/// the group, so values leave a group only from its last node, and every node in the group reaches
/// that one. A path from it back into the group would be a cycle in the original graph.
class ElementwiseFusion final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "fuse-elementwise"; }

  [[nodiscard]] std::string_view description() const override {
    return "group chains of elementwise operations into one kernel";
  }

  Result<bool> run(ir::Graph& graph) override {
    const std::vector<ir::Node*> nodes = graph.nodes();

    std::unordered_map<const ir::Node*, std::size_t> position;
    position.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      position.emplace(nodes[i], i);
    }

    // Indexed by position rather than keyed by pointer, so the numbering is the same every run.
    std::vector<std::size_t> group(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      group[i] = i;
    }

    for (std::size_t i = 0; i < nodes.size(); ++i) {
      for (const ir::Value* operand : nodes[i]->operands()) {
        const ir::Node* producer = operand->producer();
        if (producer == nullptr || !can_fuse(graph, producer, nodes[i])) {
          continue;
        }
        const std::size_t into = group[position.at(producer)];
        const std::size_t from = group[i];
        if (into == from) {
          continue;
        }
        std::replace(group.begin(), group.end(), from, into);
      }
    }

    return annotate(nodes, group);
  }

 private:
  /// Writes `tessiq.fusion` on every node in a group of two or more, and on a node alone that
  /// `runs_alone_as_a_kernel`. Groups are numbered from zero in the order they first appear.
  static Result<bool> annotate(const std::vector<ir::Node*>& nodes,
                               const std::vector<std::size_t>& group) {
    std::vector<std::size_t> sizes(nodes.size(), 0);
    for (const std::size_t id : group) {
      sizes[id]++;
    }

    bool changed = false;
    std::unordered_map<std::size_t, std::int64_t> numbering;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (sizes[group[i]] < 2 && !runs_alone_as_a_kernel(nodes[i])) {
        continue;
      }
      const auto [it, inserted] =
          numbering.emplace(group[i], static_cast<std::int64_t>(numbering.size()));

      const ir::Attribute* existing = nodes[i]->attribute(kFusion);
      const auto* held = existing == nullptr ? nullptr : std::get_if<std::int64_t>(existing);
      if (held != nullptr && *held == it->second) {
        continue;
      }
      nodes[i]->set_attribute(std::string(kFusion), it->second);
      changed = true;
    }

    return changed;
  }
};

TESSIQ_REGISTER_PASS(ElementwiseFusion)

} // namespace

} // namespace tessiq::passes
