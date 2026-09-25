// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <string_view>
#include <vector>

#include "tessiq/codegen/kernel.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_support.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

constexpr std::string_view kSupported = "tessiq.supported";

/// The operation an inserted copy carries. Not an ATen name, so it is not mistaken for part of the
/// model.
constexpr std::string_view kMaterialise = "tessiq.contiguous";

bool claimed(const ir::Node* node) {
  const ir::Attribute* attribute = node->attribute(kSupported);
  const bool* value = attribute == nullptr ? nullptr : std::get_if<bool>(attribute);
  return value != nullptr && *value;
}

/// Whether `node` is a claimed elementwise operation or reduction, whose generated kernel reads
/// its operands in packed order.
///
/// A claimed movement operation is not: it returns a view of its input's storage, so packing the
/// input first would add a copy and give the same result.
bool reads_through_a_kernel(const ir::Node* node) {
  if (!claimed(node)) {
    return false;
  }
  const OpInfo* info = find_op(node->op());
  return info != nullptr &&
         (info->category == OpCategory::kElementwise || info->category == OpCategory::kReduction);
}

/// Inserts a `tessiq.contiguous` copy before each strided operand of a generated kernel.
///
/// Precondition: `mark-supported` has run, so every node says whether the backend claims it.
/// Postcondition: every operand a generated kernel reads is contiguous.
///
/// Generated kernels index their buffers as packed row-major, so a strided input has to be packed
/// first. The executor does that itself, calling `.contiguous()` on the inputs of any kernel whose
/// `packs_inputs()` is true. This pass records each copy as a node so it shows in the IR. The
/// emitter reads through it to the original value (see `codegen::held_value`).
class LayoutAssign final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "layout-assign"; }

  [[nodiscard]] std::string_view description() const override {
    return "copy the strided values that claimed operations read";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;

    // A copy of the node list, so the copies inserted below are not visited.
    const std::vector<ir::Node*> original = graph.nodes();

    for (ir::Node* node : original) {
      if (!reads_through_a_kernel(node)) {
        continue;
      }
      for (std::size_t i = 0; i < node->operand_count(); ++i) {
        ir::Value* operand = node->operand(i);
        if (operand == nullptr || operand->type().is_contiguous()) {
          continue;
        }

        ir::Node* copy = graph.insert_node(std::string(kMaterialise), node);
        copy->append_operand(operand);
        // Not in the op table, so marked here. The executor makes the copy; there is no kernel.
        copy->set_attribute(std::string(kSupported), false);

        ir::Value* packed = copy->append_result(graph.unique_name(operand->name() + ".contiguous"),
                                                operand->type().contiguous());
        node->set_operand(i, packed);
        changed = true;
      }
    }

    return changed;
  }
};

TESSIQ_REGISTER_PASS(LayoutAssign)

} // namespace

} // namespace tessiq::passes
