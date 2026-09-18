// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/op_support.h"

#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tessiq::passes {

namespace {

/// Elementwise operations, which is what fusion will target first.
///
/// The ATen spellings are what the importer produces, so this list is comparable against a real
/// imported graph without translation.
const std::vector<std::string>& builtin_ops() {
  static const std::vector<std::string> ops{
      "aten.abs.default",
      "aten.add.Tensor",
      "aten.div.Tensor",
      "aten.mul.Tensor",
      "aten.neg.default",
      "aten.relu.default",
      "aten.sigmoid.default",
      "aten.sub.Tensor",
      "aten.tanh.default",
  };
  return ops;
}

} // namespace

OpSupport::OpSupport(std::vector<std::string> ops) {
  for (auto& op : ops) {
    ops_.insert(std::move(op));
  }
}

bool OpSupport::supports(std::string_view op) const { return ops_.find(op) != ops_.end(); }

void OpSupport::add(std::string op) { ops_.insert(std::move(op)); }

void OpSupport::remove(std::string_view op) {
  if (const auto it = ops_.find(op); it != ops_.end()) {
    ops_.erase(it);
  }
}

std::vector<std::string> OpSupport::names() const { return {ops_.begin(), ops_.end()}; }

OpSupport& default_op_support() {
  static OpSupport support(builtin_ops());
  return support;
}

} // namespace tessiq::passes
