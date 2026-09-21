// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/op_traits.h"

#include <string>
#include <string_view>
#include <vector>

#include "tessiq/passes/op_table.h"

namespace tessiq::passes {

bool is_elementwise(std::string_view op) {
  const OpInfo* info = find_op(op);
  return info != nullptr && info->category == OpCategory::kElementwise;
}

int elementwise_arity(std::string_view op) { return is_elementwise(op) ? find_op(op)->arity : 0; }

std::string_view elementwise_expression(std::string_view op) {
  return is_elementwise(op) ? find_op(op)->expression : std::string_view{};
}

std::vector<std::string> elementwise_ops() {
  std::vector<std::string> names;
  for (const std::string& name : op_names()) {
    if (is_elementwise(name)) {
      names.push_back(name);
    }
  }
  return names;
}

} // namespace tessiq::passes
