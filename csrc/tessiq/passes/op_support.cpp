// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/op_support.h"

#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessiq/passes/op_table.h"

namespace tessiq::passes {

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
  // Seeded from the generated table rather than a list here, so an operation gains a kernel in
  // one place and the coverage document cannot disagree with what the backend claims.
  static OpSupport support(ops_with_kernels());
  return support;
}

} // namespace tessiq::passes
