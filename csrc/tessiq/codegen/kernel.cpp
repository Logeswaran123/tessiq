// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/kernel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_traits.h"

namespace tessiq::codegen {

namespace {

constexpr std::string_view kFusion = "tessiq.fusion";

/// The fusion group a node belongs to, or nothing.
const std::int64_t* group_of(const ir::Node* node) {
  const ir::Attribute* attribute = node->attribute(kFusion);
  return attribute == nullptr ? nullptr : std::get_if<std::int64_t>(attribute);
}

} // namespace

namespace {

/// The operand's extents lined up against the output's, or nothing when they cannot broadcast.
///
/// Broadcasting aligns shapes to the right, so a missing axis reads as 1. An axis that is neither
/// 1 nor the output's own extent is not a broadcast of it.
struct Alignment {
  /// The operand's extents lined up against the output's, 1 where it has no axis.
  std::vector<std::int64_t> extents;
  /// What to step by on each axis of the output: the operand's own stride, or zero where it
  /// broadcasts and the same element is read for the whole axis.
  std::vector<std::int64_t> steps;
};

std::optional<Alignment> align(const ir::TensorType& output, const ir::TensorType& operand) {
  const std::vector<std::int64_t>& out = output.shape().dims();
  const std::vector<std::int64_t>& in = operand.shape().dims();
  if (in.size() > out.size()) {
    return std::nullopt;
  }

  // Contiguous is recorded as no layout, so the strides it implies are computed here.
  const ir::Strides strides =
      operand.is_contiguous() ? ir::contiguous_strides(operand.shape()) : operand.strides();
  if (strides.size() != in.size()) {
    return std::nullopt;
  }

  Alignment aligned;
  aligned.extents.assign(out.size(), 1);
  aligned.steps.assign(out.size(), 0);

  const std::size_t offset = out.size() - in.size();
  for (std::size_t j = 0; j < in.size(); ++j) {
    aligned.extents[offset + j] = in[j];
    aligned.steps[offset + j] = strides[j];
  }
  for (std::size_t j = 0; j < out.size(); ++j) {
    if (aligned.extents[j] != out[j] && aligned.extents[j] != 1) {
      return std::nullopt;
    }
    if (aligned.extents[j] == 1) {
      // One element read for the whole axis, whatever stride it nominally has.
      aligned.steps[j] = 0;
    }
    if (aligned.steps[j] < 0) {
      // A negative stride is a reversed view. Nothing here produces one and the arithmetic below
      // would need an origin to count back from, so it is refused rather than guessed at.
      return std::nullopt;
    }
  }
  return aligned;
}

/// Whether the steps are the packed row-major ones for `extents`, which is what `i` means.
bool steps_are_packed(const std::vector<std::int64_t>& extents,
                      const std::vector<std::int64_t>& steps) {
  std::int64_t expected = 1;
  for (std::size_t j = extents.size(); j > 0; --j) {
    const std::size_t axis = j - 1;
    if (extents[axis] != 1 && steps[axis] != expected) {
      return false;
    }
    expected *= extents[axis];
  }
  return true;
}

/// `i / n` when the operand is the output packed, with a run of trailing axes read once each.
///
/// The per-row value a softmax and a layer norm subtract, and the cheapest case there is.
std::optional<std::string> prefix_index(const std::vector<std::int64_t>& out,
                                        const Alignment& aligned) {
  std::size_t kept = out.size();
  while (kept > 0 && aligned.steps[kept - 1] == 0) {
    --kept;
  }
  for (std::size_t j = 0; j < kept; ++j) {
    if (aligned.extents[j] != out[j]) {
      return std::nullopt;
    }
  }
  if (!steps_are_packed(aligned.extents, aligned.steps)) {
    return std::nullopt;
  }

  std::int64_t collapsed = 1;
  for (std::size_t j = kept; j < out.size(); ++j) {
    collapsed *= out[j];
  }
  return collapsed == 1 ? std::string("i") : "i / " + std::to_string(collapsed);
}

/// `i % n` when the operand is packed and matches only the output's trailing axes, a bias row.
std::optional<std::string> suffix_index(const std::vector<std::int64_t>& out,
                                        const Alignment& aligned) {
  std::size_t leading = 0;
  while (leading < out.size() && aligned.steps[leading] == 0) {
    ++leading;
  }
  for (std::size_t j = leading; j < out.size(); ++j) {
    if (aligned.extents[j] != out[j]) {
      return std::nullopt;
    }
  }
  if (!steps_are_packed(aligned.extents, aligned.steps)) {
    return std::nullopt;
  }

  std::int64_t total = 1;
  for (std::size_t j = leading; j < out.size(); ++j) {
    total *= out[j];
  }
  return "i % " + std::to_string(total);
}

/// The output position taken apart into coordinates and stepped through the operand.
///
/// What a permuted or otherwise strided operand needs, and what materialising one used to be for.
std::string general_index(const std::vector<std::int64_t>& out, const Alignment& aligned) {
  std::string text;
  for (std::size_t j = out.size(); j > 0; --j) {
    const std::size_t axis = j - 1;
    if (aligned.steps[axis] == 0) {
      continue;
    }

    std::int64_t divisor = 1;
    for (std::size_t d = axis + 1; d < out.size(); ++d) {
      divisor *= out[d];
    }

    std::string term = "(";
    if (divisor == 1) {
      term += "i";
    } else {
      term += "(i / ";
      term += std::to_string(divisor);
      term += ")";
    }
    term += " % ";
    term += std::to_string(out[axis]);
    term += ")";
    if (aligned.steps[axis] != 1) {
      term += " * ";
      term += std::to_string(aligned.steps[axis]);
    }

    if (text.empty()) {
      text = std::move(term);
    } else {
      term += " + ";
      term += text;
      text = std::move(term);
    }
  }
  return text.empty() ? "0" : text;
}

} // namespace

std::string index_for(const ir::TensorType& output, const ir::TensorType& operand) {
  if (output.shape().dims() == operand.shape().dims() && operand.is_contiguous()) {
    return "i";
  }
  const std::optional<Alignment> aligned = align(output, operand);
  if (!aligned.has_value()) {
    return {};
  }
  if (const std::optional<std::string> prefix = prefix_index(output.shape().dims(), *aligned)) {
    return *prefix;
  }
  if (const std::optional<std::string> suffix = suffix_index(output.shape().dims(), *aligned)) {
    return *suffix;
  }
  return general_index(output.shape().dims(), *aligned);
}

std::vector<std::string> FusedKernel::input_names() const {
  std::vector<std::string> names;
  names.reserve(inputs.size());
  for (const KernelInput& input : inputs) {
    names.push_back(input.name);
  }
  return names;
}

std::vector<std::int64_t> fusion_groups(const ir::Graph& graph) {
  std::set<std::int64_t> seen;
  for (const ir::Node* node : graph.nodes()) {
    if (const std::int64_t* group = group_of(node); group != nullptr) {
      seen.insert(*group);
    }
  }
  return {seen.begin(), seen.end()};
}

namespace {

/// The nodes carrying `tessiq.fusion = group`, in graph order.
std::vector<const ir::Node*> members_of(const ir::Graph& graph, std::int64_t group) {
  std::vector<const ir::Node*> members;
  for (const ir::Node* node : graph.nodes()) {
    if (const std::int64_t* id = group_of(node); id != nullptr && *id == group) {
      members.push_back(node);
    }
  }
  return members;
}

/// Turns one node into a step, recording any operand the group does not compute as an input.
Result<Step> build_step(const ir::Node* node,
                        const ir::TensorType& output,
                        const std::unordered_set<const ir::Node*>& inside,
                        std::unordered_set<std::string>& seen,
                        std::vector<KernelInput>& inputs) {
  const std::string_view expression = passes::elementwise_expression(node->op());
  if (expression.empty()) {
    return unsupported("no expression for '" + node->op() + "'");
  }

  Step step;
  step.op = node->op();
  step.result = node->result(0)->name();

  // Arguments are rebuilt position by position rather than operands first. The importer keys a
  // non-tensor argument by the position it occupied, so a constant can come before a tensor:
  // `1 - sigmoid(x)` in a backward graph is `sub` with the constant at position 0. Appending the
  // constants after the operands would compute `sigmoid(x) - 1`, which is the same shape, the
  // same type, and the wrong number.
  const auto arity = static_cast<std::size_t>(passes::elementwise_arity(node->op()));
  std::size_t next_operand = 0;
  for (std::size_t position = 0; position < arity; ++position) {
    if (const ir::Attribute* attribute = node->attribute("arg" + std::to_string(position));
        attribute != nullptr) {
      step.args.push_back({ir::attribute_to_string(*attribute), false});
      continue;
    }
    if (next_operand >= node->operand_count()) {
      return invalid_argument("'" + node->op() + "' takes " + std::to_string(arity) +
                              " arguments but has only " + std::to_string(node->operand_count()) +
                              " operands and no arg" + std::to_string(position));
    }
    const ir::Value* operand = node->operand(next_operand++);
    step.args.push_back({operand->name(), true});
    if (inside.count(operand->producer()) == 0 && seen.insert(operand->name()).second) {
      const std::string index = index_for(output, operand->type());
      if (index.empty()) {
        return unsupported("'" + node->op() + "' reads " + operand->type().to_string() +
                           " into a loop over " + output.to_string() +
                           ", which no index expression describes");
      }
      inputs.push_back({operand->name(), index, operand->type()});
    }
  }

  return step;
}

/// The values the group computes that something outside it reads.
std::vector<std::string> escaping_values(const ir::Graph& graph,
                                         const std::vector<const ir::Node*>& members,
                                         const std::unordered_set<const ir::Node*>& inside) {
  const auto& results = graph.results();
  std::vector<std::string> escaping;
  for (const ir::Node* node : members) {
    const ir::Value* value = node->result(0);
    const bool returned = std::find(results.begin(), results.end(), value) != results.end();
    const bool read_outside =
        std::any_of(value->users().begin(), value->users().end(), [&inside](const ir::Node* user) {
          return inside.count(user) == 0;
        });
    if (returned || read_outside) {
      escaping.push_back(value->name());
    }
  }
  return escaping;
}

/// Checks every member produces one value of the same type, and returns that type.
Result<ir::TensorType> common_type(const std::vector<const ir::Node*>& members,
                                   std::int64_t group) {
  const ir::TensorType& first = members.front()->result(0)->type();
  if (!first.shape().is_static()) {
    return invalid_argument("fusion group " + std::to_string(group) + " has shape " +
                            first.to_string() + ", which has no element count");
  }
  for (const ir::Node* node : members) {
    if (node->result_count() != 1) {
      return invalid_argument("fusion group " + std::to_string(group) + " contains '" + node->op() +
                              "', which has " + std::to_string(node->result_count()) + " results");
    }
    if (node->result(0)->type() != first) {
      return invalid_argument("fusion group " + std::to_string(group) + " mixes " +
                              first.to_string() + " and " + node->result(0)->type().to_string() +
                              "; one loop cannot write both");
    }
  }
  return first;
}

} // namespace

Result<FusedKernel> extract_kernel(const ir::Graph& graph, std::int64_t group) {
  const std::vector<const ir::Node*> members = members_of(graph, group);
  if (members.empty()) {
    return not_found("no fusion group " + std::to_string(group) + " in graph '" + graph.name() +
                     "'");
  }
  TESSIQ_ASSIGN_OR_RETURN(const ir::TensorType type, common_type(members, group));

  const ir::TensorType& result_type = type;
  const std::unordered_set<const ir::Node*> inside(members.begin(), members.end());

  FusedKernel kernel;
  kernel.name = "tessiq_fused_" + std::to_string(group);
  kernel.dtype = type.dtype();
  kernel.elements = type.shape().element_count();

  std::unordered_set<std::string> seen;
  for (const ir::Node* node : members) {
    TESSIQ_ASSIGN_OR_RETURN(Step step, build_step(node, result_type, inside, seen, kernel.inputs));
    kernel.body.push_back(std::move(step));
  }

  // Exactly one value leaves the group. Every other member has a single reader inside it, which
  // is the condition that let it fuse in the first place.
  const std::vector<std::string> escaping = escaping_values(graph, members, inside);
  if (escaping.size() > 1) {
    return invalid_argument("fusion group " + std::to_string(group) + " has " +
                            std::to_string(escaping.size()) +
                            " values leaving it, which one kernel cannot write");
  }
  // A group whose result nothing reads is dead and dce removes it. Writing the last value keeps
  // the kernel well formed if one is met before that happens.
  kernel.output = escaping.empty() ? members.back()->result(0)->name() : escaping.front();

  return kernel;
}

} // namespace tessiq::codegen
