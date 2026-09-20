// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/ir/recipe.h"

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/memory_plan.h"

namespace tessiq::ir {

namespace {

constexpr std::string_view kCluster = "tessiq.cluster";

/// The cluster recorded on a node, or kHost when it carries none.
Result<int> cluster_of(const Node* node) {
  const Attribute* attribute = node->attribute(kCluster);
  if (attribute == nullptr) {
    return kHost;
  }
  const auto* value = std::get_if<std::int64_t>(attribute);
  if (value == nullptr) {
    return invalid_argument("node '" + node->op() + "' has a " + std::string(kCluster) +
                            " attribute that is not an integer");
  }
  return static_cast<int>(*value);
}

} // namespace

std::string Recipe::to_string() const {
  std::string text = "recipe: " + std::to_string(steps.size()) + " step(s), arena " +
                     std::to_string(arena_bytes) + " bytes, naive " + std::to_string(naive_bytes) +
                     " bytes\n";
  for (const Step& step : steps) {
    text += "  ";
    for (std::size_t i = 0; i < step.results.size(); ++i) {
      text += (i == 0 ? "%" : ", %") + step.results[i];
    }
    text += " = " + step.op;
    text += step.cluster == kHost ? " [host]" : " [cluster " + std::to_string(step.cluster) + "]";
    text += "\n";
  }
  for (const Buffer& buffer : buffers) {
    text += "  buffer %" + buffer.name + " @ " + std::to_string(buffer.offset) + " + " +
            std::to_string(buffer.size) + "\n";
  }
  return text;
}

Result<Recipe> build_recipe(const Graph& graph, std::size_t alignment) {
  TESSIQ_ASSIGN_OR_RETURN(const MemoryPlan plan, plan_memory(graph, alignment));

  Recipe recipe;
  recipe.arena_bytes = plan.arena_bytes;
  recipe.naive_bytes = plan.naive_bytes;

  recipe.steps.reserve(graph.node_count());
  for (const Node* node : graph.nodes()) {
    TESSIQ_ASSIGN_OR_RETURN(const int cluster, cluster_of(node));

    Step step;
    step.op = node->op();
    step.cluster = cluster;
    step.results.reserve(node->result_count());
    for (const Value* result : node->results()) {
      step.results.push_back(result->name());
    }
    recipe.steps.push_back(std::move(step));
  }

  recipe.buffers.reserve(plan.allocations.size());
  for (const Allocation& allocation : plan.allocations) {
    recipe.buffers.push_back({allocation.value->name(), allocation.offset, allocation.size});
  }

  return recipe;
}

} // namespace tessiq::ir
