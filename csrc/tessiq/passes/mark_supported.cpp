// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <string_view>

#include "tessiq/passes/op_support.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {

namespace {

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
      const bool supported = support.supports(node->op());
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
