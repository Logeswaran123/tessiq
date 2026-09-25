// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
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
/// How a loop over the result reads one operand.
///
/// Three of the four forms need one number, and that number can arrive at the launch, which is
/// what lets a kernel be written for a shape nobody knows yet. The fourth needs every extent and
/// so is only ever available for a shape that is known.
struct OperandRead {
  enum class Kind {
    /// Position `i`: the operand has the result's shape, so one index is the same element in
    /// both. Almost every operand.
    kSameShape,
    /// Position `i / d`: runs of `d` output elements share one operand element. The per-row mean
    /// a layer norm subtracts and the per-row maximum a softmax does.
    kRepeated,
    /// Position `i % d`: the operand is `d` elements read over and over. A bias row.
    kCycled,
    /// The position taken apart into coordinates and stepped through the operand's own layout.
    /// What a permuted operand needs, and the one form that cannot be written without numbers.
    kStrided,
  };

  Kind kind = Kind::kSameShape;

  /// For `kRepeated`, how many output elements share one operand element. For `kCycled`, the
  /// operand's own element count. `ir::kDynamic` when it is a product of extents that are not
  /// known, in which case the launch works it out from the sizes of the buffers it is handed.
  /// One for the other two kinds, which need no number.
  std::int64_t divisor = 1;

  /// Whether the number is one the kernel source can hold.
  [[nodiscard]] bool is_known() const { return divisor != ir::kDynamic; }
};

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

  /// How the operand is reached, which is what a kernel without known extents uses. `index` is
  /// the same answer with the numbers folded in, and is empty when there were none to fold.
  OperandRead read;

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
  ///
  /// `ir::kDynamic` when the graph was traced for more than one shape and the count is not known
  /// until the call. The generated code is the same either way, because the count reaches the
  /// kernel as an argument rather than as a constant; only the launch differs.
  std::int64_t elements = 0;

  /// Whether the element count arrives with the tensors rather than being known here.
  [[nodiscard]] bool has_dynamic_extent() const { return elements == ir::kDynamic; }

  /// Values read from memory, in the order the kernel takes them as parameters.
  std::vector<KernelInput> inputs;

  /// The inputs' names, in order, for callers that only bind buffers.
  [[nodiscard]] std::vector<std::string> input_names() const;

  /// The value written to memory.
  std::string output;

  std::vector<Step> body;

  /// Values of the original graph that a rewrite folded into one of the steps. The kernel computes
  /// them on its way as much as it computes the steps, so the executor has to skip them as well.
  std::vector<std::string> absorbed;
};

/// The value the executor holds for `value`, which is the tensor a kernel is handed for it.
///
/// `value` itself, unless it is a copy `layout-assign` made so that a kernel could read a strided
/// value in order. The executor makes that copy on its own, by packing every input of a kernel that
/// packs its inputs, so the copy never exists as a tensor of its own and its name is not one the
/// executor knows. What the executor holds is the value the copy was made from.
///
/// A kernel reads such an input under the source's name and with the copy's layout, which is the
/// packed layout the executor hands over.
[[nodiscard]] const ir::Value* held_value(const ir::Value* value);

/// The values of the original graph a rewrite folded into `node`, from `tessiq.absorbed`.
///
/// Empty for a node no rewrite made. The executor walks the original graph, so a kernel standing in
/// for such a node has to list these among what it computes, or they would run a second time.
[[nodiscard]] std::vector<std::string> absorbed_by(const ir::Node& node);

/// Pulls fusion group `group` out of `graph`.
///
/// Precondition: `fuse-elementwise` has run, so the group's nodes carry `tessiq.fusion`.
///
/// Fails when the group is not a single unit a loop can express: mixed dtypes, mixed element
/// counts, or more than one value escaping. Those are conditions `fuse-elementwise` already
/// refuses, so a failure here means the two disagree rather than that the graph is unusual.
Result<FusedKernel> extract_kernel(const ir::Graph& graph, std::int64_t group);

/// How a loop over `output` reads `operand`, or nothing when no kernel here can reach it.
///
/// Covers both reasons an operand is not read at `i`. It may have fewer elements, which is a
/// broadcast, and it may hold the ones it has in a different order, which is a stride.
///
/// Asked by three places: fusion, to decide whether a group can form; `mark-supported`, to decide
/// whether the node belongs on the device at all; and the emitter, to write the index. The three
/// disagreeing is how an operation ends up on the device with no kernel to run it, which costs a
/// transfer each way and buys nothing, so they ask one function rather than three.
[[nodiscard]] std::optional<OperandRead> operand_read(const ir::TensorType& output,
                                                      const ir::TensorType& operand);

/// How a loop over `output` reads `operand`, as an expression in `i`.
///
/// Empty when there is no such expression, which is either because the operand cannot be reached
/// at all or because reaching it needs a number that is not known until the call. `operand_read`
/// tells the two apart.
[[nodiscard]] std::string index_for(const ir::TensorType& output, const ir::TensorType& operand);

/// Fusion group numbers in a graph, ascending.
[[nodiscard]] std::vector<std::int64_t> fusion_groups(const ir::Graph& graph);

} // namespace tessiq::codegen
