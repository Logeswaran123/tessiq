// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/op_traits.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace tessiq::passes {

namespace {

/// One table, so an operation cannot be half-added.
///
/// Both spellings are here because both occur. A graph imported from torch carries the ATen name;
/// a graph written by hand for a pass test carries the short one.
constexpr std::array<std::string_view, 24> kElementwise{{
    "abs",
    "add",
    "div",
    "mul",
    "neg",
    "relu",
    "sigmoid",
    "sqrt",
    "sub",
    "tanh",
    "aten.abs.default",
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.div.Scalar",
    "aten.div.Tensor",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.neg.default",
    "aten.relu.default",
    "aten.sigmoid.default",
    "aten.sqrt.default",
    "aten.sub.Scalar",
    "aten.sub.Tensor",
    "aten.tanh.default",
}};

} // namespace

bool is_elementwise(std::string_view op) {
  return std::find(kElementwise.begin(), kElementwise.end(), op) != kElementwise.end();
}

std::vector<std::string> elementwise_ops() {
  std::vector<std::string> names;
  names.reserve(kElementwise.size());
  for (const std::string_view op : kElementwise) {
    names.emplace_back(op);
  }
  std::sort(names.begin(), names.end());
  return names;
}

} // namespace tessiq::passes
