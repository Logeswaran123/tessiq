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

/// Whether there can be a kernel for this node given the types it actually has.
///
/// Support is a property of the operation for every shape the compiler knows, and this is the one
/// case where it is not. When an extent is not known until the call, what can still run is
/// anything whose numbers can arrive with the tensors: a matrix multiply, a reduction, and a loop
/// reading operands at the output position or in whole runs or cycles of it. What cannot is a
/// loop that steps through a permuted operand axis by axis, because that needs every extent
/// written into the source and there are none to write.
///
/// Claiming those anyway is worse than leaving them alone. A node on the device with no kernel
/// reaches the host fallback, which is the same computation plus a transfer each way, so the
/// graph gets slower the more of it the compiler claims. Measured on the encoder block: claiming
/// everything put 69 operations on that path and took 12.2 ms against 0.7 ms for eager.
bool has_a_kernel_for_these_types(const ir::Node& node) {
  if (!any_extent_is_unknown(node)) {
    return true;
  }
  const OpInfo* info = find_op(node.op());
  if (info == nullptr) {
    return false;
  }
  if (info->category == OpCategory::kMovement) {
    // No kernel either way. A movement operation is interpreted wherever it is placed, and
    // `cluster` already keeps one that moves nothing onto the host.
    return true;
  }
  if (info->category == OpCategory::kMatmul || info->category == OpCategory::kReduction ||
      info->category == OpCategory::kCopy) {
    // All three take their extents as arguments: BLAS at the call, the reduction kernel as three
    // numbers beside its buffers, and the pack kernel as one extent and one step per axis. So a
    // shape that changes does not stop any of them. Whether the layout or the axes are ones they
    // can handle is a separate question, and one that `extract_gemm`, `extract_reduction` and the
    // pack registration already answer, exactly as they do for a shape that is known. This asks
    // only what the extents decide.
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
/// This changes nothing about what the graph computes; it annotates. Splitting it from clustering
/// means the answer is visible in the printed IR, so a clustering test can be read without also
/// having to work out which operations the support set happened to contain.
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
