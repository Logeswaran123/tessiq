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

/// The operation an inserted materialisation carries.
///
/// Not an ATen name, because it is not an operation the model asked for. It is the copy that has
/// to happen for a kernel to be able to read a value at all, and naming it after something in
/// ATen would make it look like part of the model.
constexpr std::string_view kMaterialise = "tessiq.contiguous";

bool claimed(const ir::Node* node) {
  const ir::Attribute* attribute = node->attribute(kSupported);
  const bool* value = attribute == nullptr ? nullptr : std::get_if<bool>(attribute);
  return value != nullptr && *value;
}

/// Whether `node` reads its operands through a generated kernel, which walks memory in order.
///
/// Not the same question as whether the backend claims it. A movement operation is claimed and
/// reads nothing in order: it produces a different view of its input's storage, and a permute
/// given a packed copy would produce a permuted view of the copy, which is the same answer after
/// a transfer nobody needed. Packing before one would undo exactly what makes it free.
bool reads_through_a_kernel(const ir::Node* node) {
  if (!claimed(node)) {
    return false;
  }
  const OpInfo* info = find_op(node->op());
  return info != nullptr &&
         (info->category == OpCategory::kElementwise || info->category == OpCategory::kReduction);
}

/// Copies the strided values a generated kernel cannot step through.
///
/// Precondition: `mark-supported` has run, so every node says whether the backend claims it.
/// Postcondition: every value a kernel reads is one the emitter has an index expression for.
///
/// Every kernel this backend generates reads `p_in[i]` and writes `p_out[i]`, one index across
/// every buffer. That is only the same elements when the buffers are laid out the same way, so a
/// strided input has to be packed first. The runtime already does this: the executor calls
/// `.contiguous()` on every kernel input before launching one. What it does not do is say so, so
/// the copies were invisible to everything that reasons about the graph, including the memory
/// plan and anyone reading the IR to find out why a chain did not fuse.
///
/// Teaching the emitter to step through a layout is what made most of these stop. What is left is
/// the layouts no index expression describes, such as one with a negative stride, and there the
/// copy is the only way the value can be read at all.
class LayoutAssign final : public Pass {
 public:
  [[nodiscard]] std::string_view name() const override { return "layout-assign"; }

  [[nodiscard]] std::string_view description() const override {
    return "copy the strided values that claimed operations read";
  }

  Result<bool> run(ir::Graph& graph) override {
    bool changed = false;

    // `nodes()` returns a snapshot by value, which is what this needs: inserting into the graph
    // while walking it would otherwise visit what was just inserted, and a materialisation is
    // never itself a claimed node.
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
        // Marked here rather than left for another mark-supported run, because the operation is
        // not in the table and never will be: it is a copy the runtime performs, not a kernel.
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
