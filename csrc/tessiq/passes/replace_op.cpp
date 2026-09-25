// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/replace_op.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::passes {

ReplaceOp::ReplaceOp(std::string name,
                     std::string from,
                     std::string to,
                     std::vector<ir::DType> dtypes)
    : name_(std::move(name)),
      description_(from + " becomes " + to + " where " + to + " has the types"),
      from_(std::move(from)), to_(std::move(to)), dtypes_(std::move(dtypes)) {}

bool ReplaceOp::has_the_types(const ir::Node& node) const {
  // The core builds as C++17, so no std::ranges here.
  const auto declared = [this](const ir::Value* value) {
    return value == nullptr ||
           std::find(dtypes_.begin(), dtypes_.end(), value->type().dtype()) != dtypes_.end();
  };
  const std::vector<ir::Value*>& operands = node.operands();
  const std::vector<ir::Value*> results = node.results();
  return std::all_of(operands.begin(), operands.end(), declared) &&
         std::all_of(results.begin(), results.end(), declared);
}

Result<bool> ReplaceOp::run(ir::Graph& graph) {
  bool changed = false;
  for (ir::Node* node : graph.nodes()) {
    if (node->op() != from_ || node->result_count() != 1 || !has_the_types(*node)) {
      continue;
    }
    node->set_op(to_);
    changed = true;
  }
  return changed;
}

} // namespace tessiq::passes
