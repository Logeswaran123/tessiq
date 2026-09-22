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
/// One value a fused kernel reads, and how it indexes it.
///
/// `index` is an expression in `i`, the thread's output position. It is `i` for an operand the
/// same shape as the result, which is almost all of them. An operand with fewer elements, such as
/// the per-row mean a layer norm subtracts, reads a different position, and the expression says
/// which. The shapes are known when the kernel is written, so the arithmetic is emitted rather
/// than worked out per element.
struct KernelInput {
  std::string name;
  std::string index = "i";

  /// The layout the index expression was written for. The launch checks it, because an index
  /// worked out for one layout and applied to another reads the right buffer in the wrong order.
  ir::TensorType type;
};

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
  std::vector<KernelInput> inputs;

  /// The inputs' names, in order, for callers that only bind buffers.
  [[nodiscard]] std::vector<std::string> input_names() const;

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

/// How a loop over `output` reads `operand`, as an expression in `i`.
///
/// Covers both reasons an operand is not read at `i`. It may have fewer elements, which is a
/// broadcast, and it may hold the ones it has in a different order, which is a stride. Both are
/// the same arithmetic: a step per axis, zero where the operand broadcasts and its own where it
/// does not.
///
/// Empty when there is none, which is how fusion asks whether an operand can be reached at all.
/// The two have to agree: a group fusion allows and the emitter then declines is a group that
/// silently does not become a kernel.
[[nodiscard]] std::string index_for(const ir::TensorType& output, const ir::TensorType& operand);

/// Fusion group numbers in a graph, ascending.
[[nodiscard]] std::vector<std::int64_t> fusion_groups(const ir::Graph& graph);

} // namespace tessiq::codegen
