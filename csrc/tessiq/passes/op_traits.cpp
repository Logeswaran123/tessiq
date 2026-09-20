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

struct Trait {
  std::string_view op;
  /// Arguments the operation takes, counting a constant that arrived as an attribute.
  int arity;
  /// The scalar expression, with `$0` and `$1` for the arguments.
  std::string_view expression;
};

/// One table, so an operation cannot be half-added: whether it is elementwise, how many arguments
/// it takes, and what it computes all come from the same row.
///
/// Both spellings are here because both occur. A graph imported from torch carries the ATen name;
/// a graph written by hand for a pass test carries the short one. A test pairs them so a missing
/// one is a failure rather than a chain that quietly stops fusing.
///
/// The expressions use `scalar_t` rather than a concrete type, and call the C++ overloads of the
/// maths functions rather than the float-suffixed ones, so the same text compiles for every
/// element type the IR carries.
constexpr std::array<Trait, 24> kTraits{{
    {"abs", 1, "fabs($0)"},
    {"add", 2, "($0 + $1)"},
    {"div", 2, "($0 / $1)"},
    {"mul", 2, "($0 * $1)"},
    {"neg", 1, "(-$0)"},
    {"relu", 1, "($0 > scalar_t(0) ? $0 : scalar_t(0))"},
    {"sigmoid", 1, "(scalar_t(1) / (scalar_t(1) + exp(-$0)))"},
    {"sqrt", 1, "sqrt($0)"},
    {"sub", 2, "($0 - $1)"},
    {"tanh", 1, "tanh($0)"},
    {"aten.abs.default", 1, "fabs($0)"},
    {"aten.add.Scalar", 2, "($0 + $1)"},
    {"aten.add.Tensor", 2, "($0 + $1)"},
    {"aten.div.Scalar", 2, "($0 / $1)"},
    {"aten.div.Tensor", 2, "($0 / $1)"},
    {"aten.mul.Scalar", 2, "($0 * $1)"},
    {"aten.mul.Tensor", 2, "($0 * $1)"},
    {"aten.neg.default", 1, "(-$0)"},
    {"aten.relu.default", 1, "($0 > scalar_t(0) ? $0 : scalar_t(0))"},
    {"aten.sigmoid.default", 1, "(scalar_t(1) / (scalar_t(1) + exp(-$0)))"},
    {"aten.sqrt.default", 1, "sqrt($0)"},
    {"aten.sub.Scalar", 2, "($0 - $1)"},
    {"aten.sub.Tensor", 2, "($0 - $1)"},
    {"aten.tanh.default", 1, "tanh($0)"},
}};

const Trait* find(std::string_view op) {
  const auto* const it =
      std::find_if(kTraits.begin(), kTraits.end(), [op](const Trait& t) { return t.op == op; });
  return it == kTraits.end() ? nullptr : it;
}

} // namespace

bool is_elementwise(std::string_view op) { return find(op) != nullptr; }

int elementwise_arity(std::string_view op) {
  const Trait* trait = find(op);
  return trait == nullptr ? 0 : trait->arity;
}

std::string_view elementwise_expression(std::string_view op) {
  const Trait* trait = find(op);
  return trait == nullptr ? std::string_view{} : trait->expression;
}

std::vector<std::string> elementwise_ops() {
  std::vector<std::string> names;
  names.reserve(kTraits.size());
  for (const Trait& trait : kTraits) {
    names.emplace_back(trait.op);
  }
  std::sort(names.begin(), names.end());
  return names;
}

} // namespace tessiq::passes
