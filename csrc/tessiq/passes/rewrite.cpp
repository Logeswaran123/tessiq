// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/rewrite.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::passes {

namespace {

/// Where a replacement records the values it absorbed.
constexpr std::string_view kAbsorbed = "tessiq.absorbed";

/// The prefix of the attributes passes add, which a pattern neither writes nor compares.
constexpr std::string_view kAnnotation = "tessiq.";

bool is_annotation(std::string_view name) {
  return name.substr(0, kAnnotation.size()) == kAnnotation;
}

/// The name an attribute captures, or empty when it is an ordinary value.
std::string_view capture_of(const ir::Attribute& value) {
  const auto* text = std::get_if<std::string>(&value);
  if (text == nullptr || text->size() < 2 || text->front() != '$') {
    return {};
  }
  return std::string_view(*text).substr(1);
}

std::optional<double> number_of(const ir::Attribute& value) {
  if (const auto* integer = std::get_if<std::int64_t>(&value)) {
    return static_cast<double>(*integer);
  }
  if (const auto* real = std::get_if<double>(&value)) {
    return *real;
  }
  return std::nullopt;
}

/// Whether two attribute values are the same, an integer and a double of the same number included.
bool same_value(const ir::Attribute& first, const ir::Attribute& second) {
  const std::optional<double> left = number_of(first);
  const std::optional<double> right = number_of(second);
  if (left.has_value() && right.has_value()) {
    // Exact on purpose: a constant in a pattern is the one the model writes, not one close to it.
    return !(*left < *right) && !(*right < *left);
  }
  return first == second;
}

/// Whether `node`'s two operands can be read in either order.
///
/// `add` only without `alpha`, which scales the second operand and not the first.
bool commutes(const ir::Node& node) {
  return node.operand_count() == 2 && node.attribute("alpha") == nullptr &&
         (node.op() == "aten.add.Tensor" || node.op() == "aten.mul.Tensor");
}

/// What a match has bound so far.
struct Match {
  /// Each pattern input, and the value of the graph in its place.
  std::map<const ir::Value*, ir::Value*> inputs;

  /// Each pattern node, and the node of the graph it matched.
  std::map<const ir::Node*, ir::Node*> nodes;

  /// Every node of the graph the match covers.
  std::set<const ir::Node*> taken;

  /// Each captured scalar, by name.
  std::map<std::string, ir::Attribute, std::less<>> captures;
};

bool match_node(const ir::Node* pattern, ir::Node* node, Match& match);

bool match_value(const ir::Value* pattern, ir::Value* value, Match& match) {
  if (pattern->producer() == nullptr) {
    const auto [bound, fresh] = match.inputs.emplace(pattern, value);
    return fresh || bound->second == value;
  }
  ir::Node* producer = value->producer();
  return producer != nullptr && pattern->result_index() == value->result_index() &&
         match_node(pattern->producer(), producer, match);
}

bool attributes_match(const ir::Node& pattern, const ir::Node& node, Match& match) {
  std::size_t compared = 0;
  for (const auto& [name, value] : node.attributes()) {
    if (is_annotation(name)) {
      continue;
    }
    const ir::Attribute* wanted = pattern.attribute(name);
    if (wanted == nullptr) {
      return false;
    }
    ++compared;
    const std::string_view captured = capture_of(*wanted);
    if (captured.empty()) {
      if (!same_value(*wanted, value)) {
        return false;
      }
      continue;
    }
    const auto [bound, fresh] = match.captures.emplace(std::string(captured), value);
    if (!fresh && !same_value(bound->second, value)) {
      return false;
    }
  }
  // Every attribute the pattern names has to be there: one it names that the node lacks would
  // mean the node was called with a default the pattern did not ask for.
  return compared == pattern.attributes().size();
}

bool operands_match(const ir::Node& pattern, ir::Node& node, Match& match, bool swapped) {
  const std::size_t count = pattern.operand_count();
  for (std::size_t i = 0; i < count; ++i) {
    const ir::Value* wanted = pattern.operand(i);
    ir::Value* operand = node.operand(swapped ? count - 1 - i : i);
    if (wanted == nullptr || operand == nullptr || !match_value(wanted, operand, match)) {
      return false;
    }
  }
  return true;
}

/// Matches `pattern` against `node` and, through their operands, everything before them.
///
/// Leaves `match` partly filled in when it fails. Every caller either gives up on the match or
/// restores the copy it took, which is what trying a commutative node's other order does.
bool match_node(const ir::Node* pattern, ir::Node* node, Match& match) {
  if (const auto found = match.nodes.find(pattern); found != match.nodes.end()) {
    return found->second == node;
  }
  if (match.taken.count(node) != 0 || node->op() != pattern->op() ||
      node->result_count() != pattern->result_count() ||
      node->operand_count() != pattern->operand_count() ||
      !attributes_match(*pattern, *node, match)) {
    return false;
  }
  match.nodes.emplace(pattern, node);
  match.taken.insert(node);

  const Match before = match;
  if (operands_match(*pattern, *node, match, false)) {
    return true;
  }
  if (!commutes(*pattern)) {
    return false;
  }
  match = before;
  return operands_match(*pattern, *node, match, true);
}

bool declares(const std::vector<ir::DType>& dtypes, const ir::Value* value) {
  return dtypes.empty() ||
         std::find(dtypes.begin(), dtypes.end(), value->type().dtype()) != dtypes.end();
}

/// Whether replacing `match` keeps the graph computing what it did.
bool is_legal(const ir::Graph& graph,
              const Match& match,
              const ir::Node* root,
              const std::vector<ir::DType>& dtypes) {
  const std::vector<ir::Value*>& results = graph.results();
  for (const auto& [pattern, node] : match.nodes) {
    if (node == root) {
      continue;
    }
    // The value stops existing, so nothing outside the match may read it, and it may not be
    // returned. A training graph returns each value its backward graph needs, which is what keeps
    // a pattern from swallowing one.
    const ir::Value* value = node->result(0);
    if (std::find(results.begin(), results.end(), value) != results.end()) {
      return false;
    }
    for (const ir::Node* user : value->users()) {
      if (match.taken.count(user) == 0) {
        return false;
      }
    }
  }
  for (const auto& [pattern, value] : match.inputs) {
    if (value->producer() != nullptr && match.taken.count(value->producer()) != 0) {
      return false;
    }
    if (!declares(dtypes, value)) {
      return false;
    }
  }
  return declares(dtypes, root->result(0));
}

/// Every node of a pattern that leads to `root`, which has to be all of them.
Result<std::set<const ir::Node*>> leading_to(const ir::Node* root) {
  std::set<const ir::Node*> reached;
  std::vector<const ir::Node*> pending{root};
  while (!pending.empty()) {
    const ir::Node* node = pending.back();
    pending.pop_back();
    if (!reached.insert(node).second) {
      continue;
    }
    for (const ir::Value* operand : node->operands()) {
      if (operand == nullptr) {
        return invalid_argument("'" + node->op() + "' in a pattern has an operand missing");
      }
      if (operand->producer() != nullptr) {
        pending.push_back(operand->producer());
      }
    }
  }
  return reached;
}

/// The names `node` captures that `captures` does not hold yet, added to it.
void add_captures(const ir::Node& node, std::vector<std::string>& captures) {
  for (const auto& [name, value] : node.attributes()) {
    const std::string_view captured = capture_of(value);
    if (!captured.empty() &&
        std::find(captures.begin(), captures.end(), captured) == captures.end()) {
      captures.emplace_back(captured);
    }
  }
}

/// The first match of the pattern rooted at `pattern_root` that can be replaced, and its root.
std::optional<std::pair<Match, ir::Node*>>
first_match(ir::Graph& graph, const ir::Node* pattern_root, const std::vector<ir::DType>& dtypes) {
  for (ir::Node* root : graph.nodes()) {
    if (root->op() != pattern_root->op()) {
      continue;
    }
    Match match;
    if (match_node(pattern_root, root, match) && is_legal(graph, match, root, dtypes)) {
      return std::make_pair(std::move(match), root);
    }
  }
  return std::nullopt;
}

/// Gives `call` the match's tensors as operands and its scalars as `arg<position>` attributes.
Result<void> fill_arguments(ir::Node& call,
                            const Match& match,
                            const ir::Graph& pattern,
                            const std::vector<RewriteArgument>& arguments) {
  for (std::size_t position = 0; position < arguments.size(); ++position) {
    const RewriteArgument& argument = arguments[position];
    if (!argument.tensor) {
      const auto captured = match.captures.find(argument.name);
      if (captured == match.captures.end()) {
        return internal_error("the pattern captures no " + argument.name);
      }
      call.set_attribute("arg" + std::to_string(position), captured->second);
      continue;
    }
    const std::vector<ir::Value*> inputs = pattern.params();
    const auto input = std::find_if(inputs.begin(), inputs.end(), [&argument](const ir::Value* v) {
      return v->name() == argument.name;
    });
    const auto bound = input == inputs.end() ? match.inputs.end() : match.inputs.find(*input);
    if (bound == match.inputs.end()) {
      return internal_error("the pattern has no input " + argument.name);
    }
    call.append_operand(bound->second);
  }
  return ok();
}

/// Replaces the matched nodes with `call`, which takes over the root's name and every use of it.
Result<void> take_over(ir::Graph& graph, ir::Node* call, ir::Node* root, const Match& match) {
  std::vector<ir::Node*> absorbed;
  std::string names;
  for (ir::Node* node : graph.nodes()) {
    if (node != root && match.taken.count(node) != 0) {
      absorbed.push_back(node);
      names += names.empty() ? "" : ",";
      names += node->result(0)->name();
    }
  }
  call->set_attribute(std::string(kAbsorbed), names);

  ir::Value* old = root->result(0);
  const std::string kept = old->name();
  ir::Value* result = call->append_result(graph.unique_name(kept + "_rewritten"), old->type());
  TESSIQ_RETURN_IF_ERROR(graph.replace_all_uses(old, result));
  TESSIQ_RETURN_IF_ERROR(graph.erase(root));
  // Last first, so each node is erased once nothing reads it.
  for (auto it = absorbed.rbegin(); it != absorbed.rend(); ++it) {
    TESSIQ_RETURN_IF_ERROR(graph.erase(*it));
  }
  result->set_name(kept);
  return ok();
}

} // namespace

Result<PatternShape> check_pattern(const ir::Graph& pattern) {
  if (pattern.results().size() != 1) {
    return invalid_argument("a pattern returns one value, and this one returns " +
                            std::to_string(pattern.results().size()));
  }
  const ir::Value* root = pattern.results().front();
  if (root->producer() == nullptr) {
    return invalid_argument("a pattern has to compute the value it returns, and %" + root->name() +
                            " is one of its inputs");
  }
  // Every node has to lead to the result. One that does not would be matched against nothing.
  TESSIQ_ASSIGN_OR_RETURN(const std::set<const ir::Node*> reached, leading_to(root->producer()));

  PatternShape shape;
  for (const ir::Node* node : pattern.nodes()) {
    if (node->result_count() != 1) {
      return invalid_argument("'" + node->op() + "' in a pattern has " +
                              std::to_string(node->result_count()) +
                              " results, and a pattern node has one");
    }
    if (reached.count(node) == 0) {
      return invalid_argument("%" + node->result(0)->name() +
                              " in a pattern does not lead to the value it returns");
    }
    add_captures(*node, shape.captures);
  }
  for (const ir::Value* input : pattern.params()) {
    if (!input->has_users()) {
      return invalid_argument("input %" + input->name() + " of a pattern is never read");
    }
    shape.inputs.push_back(input->name());
  }
  return shape;
}

Rewrite::Rewrite(std::string name,
                 std::shared_ptr<const ir::Graph> pattern,
                 std::string op,
                 std::vector<RewriteArgument> arguments,
                 std::vector<ir::DType> dtypes)
    : name_(std::move(name)), description_("each match of a pattern becomes one " + op),
      pattern_(std::move(pattern)), op_(std::move(op)), arguments_(std::move(arguments)),
      dtypes_(std::move(dtypes)) {}

Result<bool> Rewrite::run(ir::Graph& graph) {
  const ir::Node* pattern_root = pattern_->results().front()->producer();
  bool changed = false;
  // A replacement erases nodes, so each search starts over rather than carrying on over a list of
  // nodes some of which no longer exist.
  while (auto found = first_match(graph, pattern_root, dtypes_)) {
    auto& [match, root] = *found;
    ir::Node* call = graph.insert_node(op_, root);
    const Result<void> filled = fill_arguments(*call, match, *pattern_, arguments_);
    if (!filled.has_value()) {
      return filled.error().with_context(name_);
    }
    TESSIQ_RETURN_IF_ERROR(take_over(graph, call, root, match));
    changed = true;
  }
  return changed;
}

} // namespace tessiq::passes
