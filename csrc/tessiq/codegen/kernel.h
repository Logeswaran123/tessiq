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
/// `aten.mul.Tensor(%x) {arg1 = 2.0}` has one operand and one constant, in that order.
struct Argument {
  /// An IR value name when `is_value`, otherwise the constant as it should appear in source.
  std::string text;
  bool is_value = true;
};

/// One operation inside a fused kernel.
///
/// Results are IR value names, which SSA makes unique, so the generated code uses them as
/// variable names and reads against the IR it came from.
struct Step {
  std::string op;
  std::vector<Argument> args;
  std::string result;
};

/// How a loop over the result reads one operand.
///
/// Three of the four forms need at most one number, which can be passed at launch, so they work
/// for shapes not known at compile time. `kStrided` needs every extent in the source.
struct OperandRead {
  enum class Kind {
    /// Position `i`: the operand has the result's shape. Almost every operand.
    kSameShape,
    /// Position `i / d`: runs of `d` output elements share one operand element, such as the
    /// per-row mean in layer norm or the per-row maximum in softmax.
    kRepeated,
    /// Position `i % d`: the operand is `d` elements read over and over. A bias row.
    kCycled,
    /// The position split into coordinates and mapped through the operand's layout: a strided
    /// operand, or a broadcast that is neither a run nor a cycle, such as `[4,1,8]` over `[4,6,8]`.
    kStrided,
  };

  Kind kind = Kind::kSameShape;

  /// For `kRepeated`, how many output elements share one operand element. For `kCycled`, the
  /// operand's element count. `ir::kDynamic` when it depends on unknown extents, in which case the
  /// launch derives it from the buffer sizes. 1 for the other kinds.
  std::int64_t divisor = 1;

  /// Whether the number is one the kernel source can hold.
  [[nodiscard]] bool is_known() const { return divisor != ir::kDynamic; }
};

/// One value a fused kernel reads, and how it indexes it.
///
/// `index` is an expression in `i`, the thread's output position: `i` for an operand with the
/// result's shape, something else for a broadcast one.
struct KernelInput {
  std::string name;
  std::string index = "i";

  /// How the operand is read. `index` is the same with the divisor folded in, and is empty when
  /// the divisor is not known until the launch.
  OperandRead read;

  /// The operand's type as the kernel reads it. After layout-assign this is always packed.
  ir::TensorType type;
};

/// A fusion group, resolved into everything a code generator needs. Values in `body` other than
/// the inputs and the output stay in registers.
struct FusedKernel {
  /// Identifier for the generated function. Valid in C, and unique within a graph.
  std::string name;

  /// The element type every value in the group shares; fusion refuses a type change.
  ir::DType dtype = ir::DType::kF32;

  /// Elements in the output, one thread each. An input has as many, or fewer when it broadcasts.
  ///
  /// `ir::kDynamic` when the graph was traced for more than one shape. The count is a kernel
  /// argument either way, so only the launch differs.
  std::int64_t elements = 0;

  /// Whether the element count arrives with the tensors rather than being known here.
  [[nodiscard]] bool has_dynamic_extent() const { return elements == ir::kDynamic; }

  /// Values read from memory, in the order the kernel takes them as parameters.
  std::vector<KernelInput> inputs;

  /// The inputs' names, in order, for callers that only bind buffers.
  [[nodiscard]] std::vector<std::string> input_names() const;

  /// The value written to memory.
  std::string output;

  /// Scalars the kernel takes by value, of its element type, after the count and any divisors. A
  /// step reads one through an argument that is not a value, whose text is the scalar's name, so
  /// one kernel serves every value the scalar takes.
  std::vector<std::string> scalars;

  std::vector<Step> body;

  /// Values of the original graph that a rewrite folded into one of the steps. The executor skips
  /// them along with the steps.
  std::vector<std::string> absorbed;
};

/// The value the executor holds for `value`, which is the tensor a kernel is handed for it.
///
/// `value` itself, unless it is a `tessiq.contiguous` copy from layout-assign. The executor packs
/// kernel inputs itself, so the copy is never a tensor of its own and the executor holds the value
/// it was made from. A kernel reads that value by the source's name, with the copy's packed layout.
[[nodiscard]] const ir::Value* held_value(const ir::Value* value);

/// The values of the original graph a rewrite folded into `node`, from `tessiq.absorbed`.
///
/// Empty for a node no rewrite made. The executor walks the original graph, so a kernel for such a
/// node lists these among what it computes, or they would run twice.
[[nodiscard]] std::vector<std::string> absorbed_by(const ir::Node& node);

/// Pulls fusion group `group` out of `graph`.
///
/// Precondition: `fuse-elementwise` has run, so the group's nodes carry `tessiq.fusion`.
///
/// Fails when the group is not one loop: mixed dtypes, mixed element counts, or more than one
/// value escaping. `fuse-elementwise` already refuses those, so a failure means the two disagree.
Result<FusedKernel> extract_kernel(const ir::Graph& graph, std::int64_t group);

/// How a loop over `output` reads `operand`, or nothing when no kernel here can reach it.
///
/// Covers both reasons an operand is not read at `i`: it has fewer elements (a broadcast), or
/// holds them in a different order (a stride).
///
/// Fusion, `mark-supported` and the emitter all ask this, so they cannot disagree about which
/// nodes get a kernel.
[[nodiscard]] std::optional<OperandRead> operand_read(const ir::TensorType& output,
                                                      const ir::TensorType& operand);

/// How a loop over `output` reads `operand`, as an expression in `i`.
///
/// Empty when the operand cannot be reached or the expression needs a number not known until the
/// call; `operand_read` tells the two apart.
[[nodiscard]] std::string index_for(const ir::TensorType& output, const ir::TensorType& operand);

/// Fusion group numbers in a graph, ascending.
[[nodiscard]] std::vector<std::int64_t> fusion_groups(const ir::Graph& graph);

} // namespace tessiq::codegen
