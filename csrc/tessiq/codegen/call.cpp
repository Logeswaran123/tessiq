// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/call.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/passes/op_registry.h"
#include "tessiq/passes/op_table.h"

namespace tessiq::codegen {

namespace {

/// The scalar an attribute holds, or nothing for one a kernel parameter has no type for.
std::optional<CallArgument> scalar_of(const ir::Attribute& attribute) {
  if (const auto* integer = std::get_if<std::int64_t>(&attribute)) {
    return CallArgument{*integer};
  }
  if (const auto* real = std::get_if<double>(&attribute)) {
    return CallArgument{*real};
  }
  if (const auto* flag = std::get_if<bool>(&attribute)) {
    return CallArgument{*flag};
  }
  return std::nullopt;
}

/// The attribute holding the argument at `position`, or null when that position is a tensor.
///
/// The importer keeps a positional scalar as `argN` and a keyword one under its own name, so a
/// position is looked up both ways. The schema's parameter names are what connect the two.
const ir::Attribute* argument_at(const ir::Node& node,
                                 std::size_t position,
                                 const std::vector<std::string>& parameters) {
  if (const ir::Attribute* positional = node.attribute("arg" + std::to_string(position))) {
    return positional;
  }
  return position < parameters.size() ? node.attribute(parameters[position]) : nullptr;
}

bool claimed(const ir::Node& node) {
  const ir::Attribute* supported = node.attribute("tessiq.supported");
  const auto* flag = supported == nullptr ? nullptr : std::get_if<bool>(supported);
  return flag != nullptr && *flag;
}

bool has_unknown_extent(const ir::Node& node) {
  for (std::size_t i = 0; i < node.operand_count(); ++i) {
    if (node.operand(i) != nullptr && !node.operand(i)->type().shape().is_static()) {
      return true;
    }
  }
  return node.result_count() == 1 && !node.result(0)->type().shape().is_static();
}

/// The call `node` makes, or nothing when its arguments cannot be put back in order.
std::optional<CustomCall> call_of(const ir::Node& node,
                                  const std::vector<std::string>& parameters) {
  CustomCall call;
  call.op = node.op();
  call.output = node.result(0)->name();
  call.has_dynamic_extent = has_unknown_extent(node);

  std::size_t operand = 0;
  for (std::size_t position = 0;; ++position) {
    if (const ir::Attribute* attribute = argument_at(node, position, parameters)) {
      const auto scalar = scalar_of(*attribute);
      if (!scalar.has_value()) {
        return std::nullopt;
      }
      call.arguments.push_back(*scalar);
    } else if (operand < node.operand_count()) {
      call.arguments.emplace_back(CallOperand{node.operand(operand++)->name()});
    } else {
      break;
    }
  }
  // A scalar past the last position reached would mean a position was skipped, which only a
  // keyword argument the schema does not declare can do. The kernel would receive it nowhere.
  std::size_t scalars = 0;
  for (const CallArgument& argument : call.arguments) {
    scalars += std::holds_alternative<CallOperand>(argument) ? 0U : 1U;
  }
  std::size_t attributes = 0;
  for (const auto& [name, value] : node.attributes()) {
    const bool annotation = std::string_view(name).substr(0, 7) == "tessiq.";
    attributes += annotation ? 0U : 1U;
  }
  if (operand != node.operand_count() || scalars != attributes) {
    return std::nullopt;
  }
  return call;
}

} // namespace

std::vector<std::string> CustomCall::inputs() const {
  std::vector<std::string> names;
  for (const CallArgument& argument : arguments) {
    if (const auto* operand = std::get_if<CallOperand>(&argument)) {
      names.push_back(operand->name);
    }
  }
  return names;
}

std::vector<CustomCall> custom_calls(const ir::Graph& graph) {
  std::vector<CustomCall> calls;
  for (const ir::Node* node : graph.nodes()) {
    const passes::OpInfo* info = passes::find_op(node->op());
    if (info == nullptr || info->category != passes::OpCategory::kCustom || !claimed(*node) ||
        node->result_count() != 1) {
      continue;
    }
    const passes::OpDefinition* definition = passes::find_registered_definition(node->op());
    const std::vector<std::string> none;
    auto call = call_of(*node, definition == nullptr ? none : definition->parameters);
    if (call.has_value()) {
      calls.push_back(std::move(*call));
    }
  }
  return calls;
}

} // namespace tessiq::codegen
