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
/// The compiler compiles nothing for it. The kernel's source and how to lay its launch out were
/// registered with it, so all the compiler adds is which values the call reads and which scalars
/// it was given, in the order the schema declares them. That is the order the registered code
/// expects its arguments in.
struct CustomCall {
  /// The operation, as a graph spells it.
  std::string op;

  /// The value it writes.
  std::string output;

  /// Every argument in the order the schema declares them, tensors and scalars interleaved.
  std::vector<CallArgument> arguments;

  /// Whether some extent it reads or writes is not known until the call.
  bool has_dynamic_extent = false;

  /// The tensors it reads, in the order they appear among the arguments.
  [[nodiscard]] std::vector<std::string> inputs() const;
};

/// Every node the backend claims whose operation is a registered kernel, in graph order.
///
/// A node whose arguments cannot be put back in order is left out, which is not a failure: it is
/// still placed on the device, and it runs there one call at a time through the operation's own
/// device implementation. That covers a scalar that is a list or a string, which the launch has no
/// type for.
[[nodiscard]] std::vector<CustomCall> custom_calls(const ir::Graph& graph);

} // namespace tessiq::codegen
