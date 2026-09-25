// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"

namespace tessiq::codegen {

/// A tensor a call reads, named by the IR value that holds it.
struct CallOperand {
  std::string name;
};

/// One argument of a call to a registered kernel.
using CallArgument = std::variant<CallOperand, std::int64_t, double, bool>;

/// A node that calls a kernel somebody registered, described so the kernel can be launched.
///
/// The compiler builds nothing for it; the kernel came with its registration. This records which
/// values the call reads and which scalars it was given, in schema order, which is the order the
/// kernel takes them in.
struct CustomCall {
  /// The operation, as a graph spells it.
  std::string op;

  /// The value it writes.
  std::string output;

  /// Every argument in the order the schema declares them, tensors and scalars interleaved.
  std::vector<CallArgument> arguments;

  /// Whether some extent it reads or writes is not known until the call.
  bool has_dynamic_extent = false;

  /// Values of the original graph a rewrite folded into the call, which it computes on its way.
  std::vector<std::string> absorbed;

  /// The tensors it reads, in the order they appear among the arguments.
  [[nodiscard]] std::vector<std::string> inputs() const;
};

/// Every node the backend claims whose operation is a registered kernel, in graph order.
///
/// A node with a scalar the launch has no type for, such as a list or a string, is left out. It
/// still runs on the device, through the operation's own device implementation.
[[nodiscard]] std::vector<CustomCall> custom_calls(const ir::Graph& graph);

} // namespace tessiq::codegen
