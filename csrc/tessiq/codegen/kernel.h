// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// One argument to an operation: either a value the graph computes, or a constant.
///
/// The importer turns a non-tensor argument into an attribute keyed by its position, so
/// `aten.mul.Tensor(%x) {arg1 = 2.0}` has one operand and one constant, and the generated code
/// needs both in the right order.
struct Argument {
  /// An IR value name when `is_value`, otherwise the constant as it should appear in source.
  std::string text;
  bool is_value = true;
};

/// One operation inside a fused kernel.
///
/// Results are IR value names. The IR is in static single assignment form, so a name identifies
/// one definition for the whole graph, which means generated code can use them as variable names
/// directly instead of inventing a numbering. Code somebody can read against the IR it came from
/// is worth more than code that is one character shorter.
struct Step {
  std::string op;
  std::vector<Argument> args;
  std::string result;
};

/// A fusion group, resolved into everything a code generator needs.
///
/// Every value in `body` other than the inputs and the output stays in a register. That is the
/// whole point of the group: the intermediates are never written anywhere.
struct FusedKernel {
  /// Identifier for the generated function. Valid in C, and unique within a graph.
  std::string name;

  /// The element type every value in the group shares. Fusion refuses a type change, so there is
  /// exactly one.
  ir::DType dtype = ir::DType::kF32;

  /// Elements each input and the output hold. Fusion refuses a shape change, so there is one
  /// count rather than one per buffer.
  std::int64_t elements = 0;

  /// Values read from memory, in the order the kernel takes them as parameters.
  std::vector<std::string> inputs;

  /// The value written to memory.
  std::string output;

  std::vector<Step> body;
};

/// Pulls fusion group `group` out of `graph`.
///
/// Precondition: `fuse-elementwise` has run, so the group's nodes carry `tessiq.fusion`.
///
/// Fails when the group is not a single unit a loop can express: mixed dtypes, mixed element
/// counts, or more than one value escaping. Those are conditions `fuse-elementwise` already
/// refuses, so a failure here means the two disagree rather than that the graph is unusual.
Result<FusedKernel> extract_kernel(const ir::Graph& graph, std::int64_t group);

/// Fusion group numbers in a graph, ascending.
[[nodiscard]] std::vector<std::int64_t> fusion_groups(const ir::Graph& graph);

} // namespace tessiq::codegen
