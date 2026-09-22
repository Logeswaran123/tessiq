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
std::optional<std::vector<std::int64_t>> aligned_extents(const ir::Shape& output,
                                                         const ir::Shape& operand) {
  const std::vector<std::int64_t>& out = output.dims();
  const std::vector<std::int64_t>& in = operand.dims();
  if (in.size() > out.size()) {
    return std::nullopt;
  }

  std::vector<std::int64_t> padded(out.size(), 1);
  const std::size_t offset = out.size() - in.size();
  for (std::size_t j = 0; j < in.size(); ++j) {
    padded[offset + j] = in[j];
  }
  for (std::size_t j = 0; j < out.size(); ++j) {
    if (padded[j] != out[j] && padded[j] != 1) {
      return std::nullopt;
    }
  }
  return padded;
}

/// Elements in the output from axis `from` onwards.
std::int64_t extent_from(const std::vector<std::int64_t>& out, std::size_t from) {
  std::int64_t total = 1;
  for (std::size_t j = from; j < out.size(); ++j) {
    total *= out[j];
  }
  return total;
}

/// `i / n` when the operand is the output with a run of trailing axes collapsed to 1.
///
/// The per-row value a softmax and a layer norm subtract, and the cheapest case there is.
std::optional<std::string> prefix_index(const std::vector<std::int64_t>& out,
                                        const std::vector<std::int64_t>& padded) {
  std::size_t kept = out.size();
  while (kept > 0 && padded[kept - 1] == 1) {
    --kept;
  }
  for (std::size_t j = 0; j < kept; ++j) {
    if (padded[j] != out[j]) {
      return std::nullopt;
    }
  }
  const std::int64_t collapsed = extent_from(out, kept);
  return collapsed == 1 ? std::string("i") : "i / " + std::to_string(collapsed);
}

/// `i % n` when the operand matches only the output's trailing axes, which is a bias row.
std::optional<std::string> suffix_index(const std::vector<std::int64_t>& out,
                                        const std::vector<std::int64_t>& padded) {
  std::size_t leading = 0;
  while (leading < out.size() && padded[leading] == 1) {
    ++leading;
  }
  for (std::size_t j = leading; j < out.size(); ++j) {
    if (padded[j] != out[j]) {
      return std::nullopt;
    }
  }
  return "i % " + std::to_string(extent_from(out, leading));
}

/// The output position taken apart into coordinates and put back together against the operand.
std::string general_index(const std::vector<std::int64_t>& out,
                          const std::vector<std::int64_t>& padded) {
  std::string text;
  std::int64_t stride = 1;
  for (std::size_t j = out.size(); j > 0; --j) {
    const std::size_t axis = j - 1;
    const std::int64_t step = stride;
    stride *= padded[axis];
    if (padded[axis] == 1) {
      continue;
    }

    const std::int64_t divisor = extent_from(out, axis + 1);
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
    if (step != 1) {
      term += " * ";
      term += std::to_string(step);
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

std::string index_for(const ir::Shape& output, const ir::Shape& operand) {
  if (output.dims() == operand.dims()) {
    return "i";
  }
  const std::optional<std::vector<std::int64_t>> padded = aligned_extents(output, operand);
  if (!padded.has_value()) {
    return {};
  }
  if (const std::optional<std::string> prefix = prefix_index(output.dims(), *padded)) {
    return *prefix;
  }
  if (const std::optional<std::string> suffix = suffix_index(output.dims(), *padded)) {
    return *suffix;
  }
  return general_index(output.dims(), *padded);
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
                        const ir::Shape& output,
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
      const std::string index = index_for(output, operand->type().shape());
      if (index.empty()) {
        return unsupported("'" + node->op() + "' reads " + operand->type().to_string() +
                           " into a loop over " + output.to_string() +
                           ", which no index expression describes");
      }
      inputs.push_back({operand->name(), index});
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

  const ir::Shape& result_shape = type.shape();
  const std::unordered_set<const ir::Node*> inside(members.begin(), members.end());

  FusedKernel kernel;
  kernel.name = "tessiq_fused_" + std::to_string(group);
  kernel.dtype = type.dtype();
  kernel.elements = type.shape().element_count();

  std::unordered_set<std::string> seen;
  for (const ir::Node* node : members) {
    TESSIQ_ASSIGN_OR_RETURN(Step step, build_step(node, result_shape, inside, seen, kernel.inputs));
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
