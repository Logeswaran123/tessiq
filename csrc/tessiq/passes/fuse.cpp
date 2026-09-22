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

/// Whether the value `producer` computes may be consumed in a register by its single reader,
/// instead of being written to memory and read back.
///
/// Every condition here is a reason the value would have to be materialised anyway:
///
///  - Both ends must be elementwise, or they do not agree on which positions they touch.
///  - Both must be in the same cluster, since a cluster is what becomes one unit of execution.
///  - The types must match exactly. Broadcasting means the two operations write different numbers
///    of elements, so one loop cannot serve both.
///  - The producer must have exactly one reader. Two readers means fusing into either one leaves
///    the other needing the value from memory, so it has to be written out regardless.
///  - The producer's result must not be returned by the graph, which is the same condition: a
///    caller reading it needs it in memory.
/// Whether a loop over `node`'s result can reach every one of its operands.
///
/// An operand the same shape as the result is read at the loop's own index, which is what almost
/// all of them are. One with fewer elements is read somewhere else, and the emitter works out
/// where from the shapes, so the question here is whether such an expression exists rather than
/// whether the shapes match. `x - mean` with one mean per row is the common case and arrives from
/// every softmax and layer norm.
///
/// It asks the code generator rather than deciding for itself, because a group this pass allows
/// and the emitter then declines is a group that silently does not become a kernel.
bool every_operand_is_reachable(const ir::Node* node) {
  const ir::TensorType& result = node->result(0)->type();
  for (const ir::Value* operand : node->operands()) {
    if (operand == nullptr || codegen::index_for(result, operand->type()).empty()) {
      return false;
    }
  }
  return true;
}

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

/// Whether a node the group could not grow around is still worth a kernel by itself.
///
/// Anything the emitter can write a loop for, which is an elementwise operation the backend
/// claims and whose operands it can reach. The alternative for such a node is the fallback, which
/// moves its operands to the host and its result back.
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
/// Postcondition: every node that becomes a kernel carries `tessiq.fusion`, including one that
/// fused with nothing. A group of one is still a kernel, because the alternative is not "a kernel
/// per operation" but no kernel: a node without the attribute reaches the fallback.
///
/// Legality is argued per condition in `can_fuse`. The structural part is worth stating separately:
/// unlike clustering, this cannot create a cycle, so it needs no check for one. Fusing happens only
/// along an edge whose producer has exactly one reader, so the only edge leaving the producer goes
/// to the consumer. A cycle through the merged pair would need an edge out of it and back in, and
/// the only way back in is through an operand of the producer, which is defined before it. That is
/// why this pass is a union-find and `cluster` is not.
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

    // Groups are indexed by position rather than keyed by node pointer. Relabelling then walks a
    // vector in a fixed order, so the same graph fuses the same way on every run, which iterating
    // a map of pointers would not guarantee.
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
  /// Writes `tessiq.fusion` on every node in a group of two or more, numbered from zero in the
  /// order the groups first appear.
  static Result<bool> annotate(const std::vector<ir::Node*>& nodes,
                               const std::vector<std::size_t>& group) {
    std::vector<std::size_t> sizes(nodes.size(), 0);
    for (const std::size_t id : group) {
      sizes[id]++;
    }

    bool changed = false;
    std::unordered_map<std::size_t, std::int64_t> numbering;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      // A group of one still gets a kernel. It used to be skipped, on the reasoning that not
      // fusing already means a kernel per operation, and that was wrong: a node with no fusion
      // group has no kernel at all, so it reached the fallback and cost a round trip. A lone
      // relu on the device was a transfer to the host and back.
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
