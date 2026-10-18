// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "tessiq/codegen/kernel.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_support.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/op_traits.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

/// Whether anything a node reads or writes has an extent that is not known until the graph runs.
bool any_extent_is_unknown(const ir::Node& node) {
  for (std::size_t i = 0; i < node.result_count(); ++i) {
    if (!node.result(i)->type().shape().is_static()) {
      return true;
    }
  }
  for (std::size_t i = 0; i < node.operand_count(); ++i) {
    const ir::Value* operand = node.operand(i);
    if (operand != nullptr && !operand->type().shape().is_static()) {
      return true;
    }
  }
  return false;
}

/// Whether every value the node reads and writes has an element type `info` accepts.
///
/// Most operations accept every type and leave the question to the emitter. The ones that name
/// types are those backed by cuBLAS or cuDNN, which are float32 only here, and custom operations,
/// whose kernels were written for particular types.
bool accepts_the_types(const OpInfo& info, const ir::Node& node) {
  for (std::size_t i = 0; i < node.result_count(); ++i) {
    if (!info.accepts(node.result(i)->type().dtype())) {
      return false;
    }
  }
  for (std::size_t i = 0; i < node.operand_count(); ++i) {
    const ir::Value* operand = node.operand(i);
    if (operand != nullptr && !info.accepts(operand->type().dtype())) {
      return false;
    }
  }
  return true;
}

/// Whether everything the node reads and writes has one element type.
///
/// Generated loops and reductions load, compute and store in a single type, so a boolean operand
/// of a float multiply, or an int32 sum that PyTorch widens to int64, would be read or written
/// wrongly.
bool has_one_element_type(const ir::Node& node) {
  const ir::DType dtype = node.result(0)->type().dtype();
  for (std::size_t i = 1; i < node.result_count(); ++i) {
    if (node.result(i)->type().dtype() != dtype) {
      return false;
    }
  }
  for (std::size_t i = 0; i < node.operand_count(); ++i) {
    const ir::Value* operand = node.operand(i);
    if (operand != nullptr && operand->type().dtype() != dtype) {
      return false;
    }
  }
  return true;
}

/// Whether an elementwise or matmul node has an argument its kernel does not read.
///
/// Those kernels read positional arguments, which the importer keys `arg0`, `arg1` and so on up to
/// the operation's arity. A keyword argument keeps its own name, such as `alpha` in
/// `torch.add(x, y, alpha=2)`, `approximate` in GELU's or `beta` in addmm's, and the kernel would
/// compute as if it had been left out.
bool has_an_argument_it_does_not_read(const OpInfo& info, const ir::Node& node) {
  for (const auto& [name, value] : node.attributes()) {
    const std::string_view key(name);
    if (key.substr(0, 7) == "tessiq.") {
      continue;
    }
    if (key.substr(0, 3) != "arg") {
      return true;
    }
    int position = 0;
    for (const char c : key.substr(3)) {
      if (c < '0' || c > '9') {
        return true;
      }
      position = (position * 10) + (c - '0');
    }
    if (key.size() == 3 || position >= info.arity) {
      return true;
    }
  }
  return false;
}

/// Whether a kernel can be built for this node, given its element types and shapes.
///
/// `OpSupport` answers for the operation and this answers for the node. It says no when a value
/// has a type the operation does not accept, when an elementwise or matmul node has an argument its
/// kernel does not read, when the node is a convolution cuDNN has no plan for, or when an extent
/// is unknown until the call and the kernel would need it compiled in.
/// Matmuls, reductions and packs take their extents as arguments, and so does a loop that reads
/// its operands at the output position or in whole runs or cycles of it. A loop that steps
/// through a permuted operand axis by axis does not.
///
/// A claimed node with no kernel runs through the host fallback, which costs a transfer each way
/// on top of the computation, so it is better left unclaimed.
bool has_a_kernel_for_these_types(const ir::Node& node) {
  const OpInfo* info = find_op(node.op());
  if (info != nullptr && !accepts_the_types(*info, node)) {
    return false;
  }
  if (info != nullptr &&
      (info->category == OpCategory::kElementwise || info->category == OpCategory::kReduction) &&
      !has_one_element_type(node)) {
    return false;
  }
  if (info != nullptr &&
      (info->category == OpCategory::kElementwise || info->category == OpCategory::kMatmul) &&
      has_an_argument_it_does_not_read(*info, node)) {
    return false;
  }
  if (info != nullptr && info->category == OpCategory::kCustom) {
    // A custom kernel reads its extents from the tensors it is given.
    return true;
  }
  if (info != nullptr && info->category == OpCategory::kConvolution) {
    // Checked before the extents, because a convolution can lack a kernel even when they are all
    // known: plans are built only for 2-D convolutions, which have 4-D tensors. The descriptors
    // are built at compile time, so an unknown extent rules one out too.
    return node.result_count() == 1 && node.result(0)->type().shape().rank() == 4 &&
           !any_extent_is_unknown(node);
  }
  if (!any_extent_is_unknown(node)) {
    return true;
  }
  if (info == nullptr) {
    return false;
  }
  if (info->category == OpCategory::kMovement) {
    // Movement operations have no kernel; they become views wherever they run. `cluster` leaves
    // a unit made only of them on the host.
    return true;
  }
  if (info->category == OpCategory::kMatmul || info->category == OpCategory::kReduction ||
      info->category == OpCategory::kCopy) {
    // These take their extents as arguments: BLAS at the call, the reduction kernel as three
    // numbers, the pack kernel as an extent and a stride per axis. Whether they handle the layout
    // or the axes is decided by `extract_gemm`, `extract_reduction` and the pack registration, as
    // it is for static shapes.
    return true;
  }
  if (info->category != OpCategory::kElementwise || node.result_count() != 1) {
    return false;
  }

  const ir::TensorType& result = node.result(0)->type();
  for (std::size_t i = 0; i < node.operand_count(); ++i) {
    const ir::Value* operand = node.operand(i);
    if (operand == nullptr || !codegen::operand_read(result, operand->type()).has_value()) {
      return false;
    }
  }
  return result.is_contiguous();
}

/// Records on each node whether the backend can run it.
///
/// Precondition: none.
/// Postcondition: every node carries `tessiq.supported`.
///
/// Only annotates. Keeping it apart from clustering puts the answer in the printed IR, so a
/// clustering test states its input directly.
class MarkSupported final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "mark-supported"; }

  [[nodiscard]] std::string_view description() const override {
    return "record on each node whether the backend can run it";
  }

  Result<bool> run(ir::Graph& graph) override {
    const OpSupport& support = default_op_support();
    bool changed = false;

    for (ir::Node* node : graph.nodes()) {
      const bool supported = support.supports(node->op()) && has_a_kernel_for_these_types(*node);
      const ir::Attribute* existing = node->attribute(kAttribute);
      if (existing != nullptr && std::get_if<bool>(existing) != nullptr &&
          *std::get_if<bool>(existing) == supported) {
        continue;
      }
      node->set_attribute(std::string(kAttribute), supported);
      changed = true;
    }

    return changed;
  }

 private:
  /// Dotted, so it cannot collide with the `arg0`, `arg1` attributes the importer writes for
  /// positional scalars.
  static constexpr std::string_view kAttribute = "tessiq.supported";
};

TESSIQ_REGISTER_PASS(MarkSupported)

} // namespace

} // namespace tessiq::passes
