// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/op_table.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace tessiq::passes {

namespace {

/// The rows come from ops.toml. Sorted by name there, so a lookup could binary search; it walks
/// instead, because the table is small enough that the search would be the more complicated half
/// of this file.
constexpr std::array<OpInfo, 26> kOps{{
#include "tessiq/passes/op_table.inc"
}};

std::vector<std::string> names_where(bool (*keep)(const OpInfo&)) {
  std::vector<std::string> names;
  for (const OpInfo& op : kOps) {
    if (keep(op)) {
      names.emplace_back(op.name);
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

} // namespace

const OpInfo* find_op(std::string_view name) {
  const auto* const it =
      std::find_if(kOps.begin(), kOps.end(), [name](const OpInfo& op) { return op.name == name; });
  return it == kOps.end() ? nullptr : it;
}

std::vector<std::string> op_names() {
  return names_where([](const OpInfo&) { return true; });
}

std::vector<std::string> ops_with_kernels() {
  return names_where([](const OpInfo& op) { return op.has_kernel; });
}

std::size_t op_count() { return kOps.size(); }

} // namespace tessiq::passes
