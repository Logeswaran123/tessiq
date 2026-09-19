// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/ir/liveness.h"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "tessiq/ir/graph.h"

namespace tessiq::ir {

namespace {

using Positions = std::unordered_map<const Node*, std::size_t>;

/// The position of the last node reading `value`, or `defined_at` when nothing does.
Result<std::size_t>
last_read(const Value* value, std::size_t defined_at, const Positions& positions) {
  std::size_t last = defined_at;
  for (const Node* user : value->users()) {
    const auto it = positions.find(user);
    if (it == positions.end()) {
      return internal_error("value %" + value->name() + " is used by '" + user->op() +
                            "', which is not in this graph");
    }
    last = std::max(last, it->second);
  }
  return last;
}

} // namespace

Result<std::vector<LiveRange>> compute_liveness(const Graph& graph) {
  const std::vector<Node*> nodes = graph.nodes();

  Positions positions;
  positions.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    positions.emplace(nodes[i], i);
  }

  const auto& results = graph.results();
  const auto returns = [&results](const Value* value) {
    return std::find(results.begin(), results.end(), value) != results.end();
  };

  std::vector<LiveRange> ranges;
  ranges.reserve(graph.params().size() + nodes.size());

  for (const Value* param : graph.params()) {
    TESSIQ_ASSIGN_OR_RETURN(const std::size_t last, last_read(param, 0, positions));
    ranges.push_back({param, 0, returns(param) ? nodes.size() : last, true, returns(param)});
  }

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    for (const Value* result : nodes[i]->results()) {
      TESSIQ_ASSIGN_OR_RETURN(const std::size_t last, last_read(result, i, positions));
      ranges.push_back({result, i, returns(result) ? nodes.size() : last, false, returns(result)});
    }
  }

  return ranges;
}

} // namespace tessiq::ir
