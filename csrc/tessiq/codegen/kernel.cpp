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

/// Where a rewrite records the values it absorbed.
constexpr std::string_view kAbsorbed = "tessiq.absorbed";

/// The operation `layout-assign` gives the copies it inserts.
constexpr std::string_view kMaterialise = "tessiq.contiguous";

/// The fusion group a node belongs to, or nothing.
const std::int64_t* group_of(const ir::Node* node) {
  const ir::Attribute* attribute = node->attribute(kFusion);
  return attribute == nullptr ? nullptr : std::get_if<std::int64_t>(attribute);
}

} // namespace

namespace {

/// An operand's extents and steps, lined up against the output's axes.
struct Alignment {
  /// The operand's extents lined up against the output's, 1 where it has no axis.
  std::vector<std::int64_t> extents;
  /// What to step by on each axis of the output: the operand's own stride, or zero where it
  /// broadcasts and the same element is read for the whole axis.
  std::vector<std::int64_t> steps;
};

/// Lines `operand` up against `output`, or nothing when it does not broadcast to it. Shapes align
/// to the right, so a missing axis reads as 1; an axis that is neither 1 nor the output's extent
/// does not broadcast.
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
///
/// Stops at the first unknown extent and answers yes for what it checked. That is safe because
/// an operand with an unknown extent only reaches this when its type is contiguous.
bool steps_are_packed(const std::vector<std::int64_t>& extents,
                      const std::vector<std::int64_t>& steps) {
  std::int64_t expected = 1;
  for (std::size_t j = extents.size(); j > 0; --j) {
    const std::size_t axis = j - 1;
    if (extents[axis] == ir::kDynamic || steps[axis] == ir::kDynamic) {
      return true;
    }
    if (extents[axis] != 1 && steps[axis] != expected) {
      return false;
    }
    expected *= extents[axis];
  }
  return true;
}

/// The product of `dims[from, to)`, or `ir::kDynamic` when one of them is not known.
std::int64_t span(const std::vector<std::int64_t>& dims, std::size_t from, std::size_t to) {
  std::int64_t total = 1;
  for (std::size_t j = from; j < to; ++j) {
    if (dims[j] == ir::kDynamic) {
      return ir::kDynamic;
    }
    total *= dims[j];
  }
  return total;
}

/// How many output elements share one operand element, when the operand is the output packed with
/// a run of trailing axes read once each.
///
/// The per-row value a softmax or a layer norm subtracts.
std::optional<std::int64_t> repeated_divisor(const std::vector<std::int64_t>& out,
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
  return span(out, kept, out.size());
}

/// How many elements the operand holds, when it is packed and matches only the output's trailing
/// axes. A bias row, read over and over down the result.
std::optional<std::int64_t> cycled_divisor(const std::vector<std::int64_t>& out,
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
  return span(out, leading, out.size());
}

/// The output position taken apart into coordinates and stepped through the operand.
///
/// For an operand that is neither read at `i` nor a run or a cycle.
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

std::optional<OperandRead> operand_read(const ir::TensorType& output,
                                        const ir::TensorType& operand) {
  if (output.shape().dims() == operand.shape().dims() && operand.is_contiguous()) {
    // Identical dim lists, so one index walks both. Two unknown extents both print as `?`, so the
    // launch checks the sizes it is handed (see `FusedKernel::launch` in torch/fused.cpp).
    return OperandRead{};
  }
  const std::optional<Alignment> aligned = align(output, operand);
  if (!aligned.has_value()) {
    return std::nullopt;
  }

  const auto broadcast = [](OperandRead::Kind kind, std::int64_t divisor) {
    OperandRead read;
    read.kind = kind;
    read.divisor = divisor;
    return read;
  };

  const std::vector<std::int64_t>& out = output.shape().dims();
  if (const std::optional<std::int64_t> repeated = repeated_divisor(out, *aligned)) {
    // A divisor of one is the same as reading at `i`.
    return *repeated == 1 ? OperandRead{} : broadcast(OperandRead::Kind::kRepeated, *repeated);
  }
  if (const std::optional<std::int64_t> cycled = cycled_divisor(out, *aligned)) {
    return broadcast(OperandRead::Kind::kCycled, *cycled);
  }
  if (!output.shape().is_static() || !operand.shape().is_static()) {
    // The strided form needs every extent in the source.
    return std::nullopt;
  }
  OperandRead strided;
  strided.kind = OperandRead::Kind::kStrided;
  return strided;
}

std::string index_for(const ir::TensorType& output, const ir::TensorType& operand) {
  const std::optional<OperandRead> read = operand_read(output, operand);
  if (!read.has_value() || !read->is_known()) {
    return {};
  }
  switch (read->kind) {
  case OperandRead::Kind::kSameShape:
    return "i";
  case OperandRead::Kind::kRepeated:
    return "i / " + std::to_string(read->divisor);
  case OperandRead::Kind::kCycled:
    return "i % " + std::to_string(read->divisor);
  case OperandRead::Kind::kStrided:
    break;
  }
  const std::optional<Alignment> aligned = align(output, operand);
  return aligned.has_value() ? general_index(output.shape().dims(), *aligned) : std::string{};
}

const ir::Value* held_value(const ir::Value* value) {
  // A loop, in case another pass ever copies a copy.
  while (value != nullptr && value->producer() != nullptr &&
         value->producer()->op() == kMaterialise && value->producer()->operand_count() == 1) {
    value = value->producer()->operand(0);
  }
  return value;
}

std::vector<std::string> absorbed_by(const ir::Node& node) {
  const ir::Attribute* attribute = node.attribute(kAbsorbed);
  const auto* names = attribute == nullptr ? nullptr : std::get_if<std::string>(attribute);
  std::vector<std::string> absorbed;
  if (names == nullptr) {
    return absorbed;
  }
  std::size_t start = 0;
  while (start <= names->size()) {
    const std::size_t comma = std::min(names->find(',', start), names->size());
    if (comma > start) {
      absorbed.push_back(names->substr(start, comma - start));
    }
    start = comma + 1;
  }
  return absorbed;
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

  // Arguments are rebuilt by position, not operands first. The importer keys a non-tensor
  // argument by its position, so a constant can come before a tensor: `1 - sigmoid(x)` in a
  // backward graph is `sub` with the constant at position 0, and appending constants after the
  // operands would compute `sigmoid(x) - 1`.
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
    const std::string& name = held_value(operand)->name();
    step.args.push_back({name, true});
    if (inside.count(operand->producer()) == 0 && seen.insert(name).second) {
      const std::optional<OperandRead> read = operand_read(output, operand->type());
      if (!read.has_value()) {
        return unsupported("'" + node->op() + "' reads " + operand->type().to_string() +
                           " into a loop over " + output.to_string() +
                           ", which no index expression describes");
      }
      // The index is empty when the divisor is not known here; the emitter then makes it a
      // parameter and the launch supplies it.
      inputs.push_back({name, index_for(output, operand->type()), *read, operand->type()});
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
  // `kDynamic` when the shape is not known here; the launch then reads the count off the output.
  kernel.elements = type.shape().is_static() ? type.shape().element_count() : ir::kDynamic;

  std::unordered_set<std::string> seen;
  for (const ir::Node* node : members) {
    TESSIQ_ASSIGN_OR_RETURN(Step step, build_step(node, result_type, inside, seen, kernel.inputs));
    kernel.body.push_back(std::move(step));
    const std::vector<std::string> absorbed = absorbed_by(*node);
    kernel.absorbed.insert(kernel.absorbed.end(), absorbed.begin(), absorbed.end());
  }

  // Exactly one value leaves the group; every other member has its only reader inside it.
  const std::vector<std::string> escaping = escaping_values(graph, members, inside);
  if (escaping.size() > 1) {
    return invalid_argument("fusion group " + std::to_string(group) + " has " +
                            std::to_string(escaping.size()) +
                            " values leaving it, which one kernel cannot write");
  }
  // A group whose result nothing reads is dead; writing its last value keeps the kernel well
  // formed until dead-node-elimination removes it.
  kernel.output = escaping.empty() ? members.back()->result(0)->name() : escaping.front();

  return kernel;
}

} // namespace tessiq::codegen
