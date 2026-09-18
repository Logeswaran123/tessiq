// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/ir/graph.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "tessiq/support/macros.h"

namespace tessiq::ir {

namespace {

/// Prints a double so that parsing it back gives the same value, and so that a whole number still
/// looks like a float. Without the trailing ".0" a printed 1.0 reads back as an integer attribute.
std::string format_double(double value) {
  std::ostringstream out;
  out.precision(17);
  out << value;
  std::string text = out.str();
  if (text.find_first_of(".eE") == std::string::npos) {
    text += ".0";
  }
  return text;
}

} // namespace

std::string attribute_to_string(const Attribute& value) {
  return std::visit(
      [](const auto& held) -> std::string {
        using T = std::decay_t<decltype(held)>;
        if constexpr (std::is_same_v<T, std::int64_t>) {
          return std::to_string(held);
        } else if constexpr (std::is_same_v<T, double>) {
          return format_double(held);
        } else if constexpr (std::is_same_v<T, bool>) {
          return held ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::string>) {
          return "\"" + held + "\"";
        } else {
          std::string text = "[";
          for (std::size_t i = 0; i < held.size(); ++i) {
            text += (i != 0 ? "," : "") + std::to_string(held[i]);
          }
          return text + "]";
        }
      },
      value);
}

// --- Value --------------------------------------------------------------------------------------

void Value::add_user(Node* user) { users_.push_back(user); }

void Value::remove_user(Node* user) {
  // One entry per operand slot, so a node using a value twice appears twice. Erase one occurrence,
  // not all of them, or dropping one operand would unregister the other.
  const auto it = std::find(users_.begin(), users_.end(), user);
  if (it != users_.end()) {
    users_.erase(it);
  }
}

// --- Node ---------------------------------------------------------------------------------------

std::vector<Value*> Node::results() const {
  std::vector<Value*> out;
  out.reserve(results_.size());
  for (const auto& result : results_) {
    out.push_back(result.get());
  }
  return out;
}

const Attribute* Node::attribute(std::string_view name) const {
  const auto it = attributes_.find(name);
  return it == attributes_.end() ? nullptr : &it->second;
}

void Node::set_attribute(std::string name, Attribute value) {
  attributes_.insert_or_assign(std::move(name), std::move(value));
}

void Node::append_operand(Value* value) {
  TESSIQ_CHECK(value != nullptr, "append_operand with a null value");
  operands_.push_back(value);
  value->add_user(this);
}

void Node::set_operand(std::size_t i, Value* value) {
  TESSIQ_CHECK(value != nullptr, "set_operand with a null value");
  TESSIQ_CHECK(i < operands_.size(), "set_operand index out of range");
  operands_[i]->remove_user(this);
  operands_[i] = value;
  value->add_user(this);
}

Value* Node::append_result(std::string name, TensorType type) {
  results_.push_back(
      std::make_unique<Value>(std::move(name), std::move(type), this, results_.size()));
  return results_.back().get();
}

void Node::drop_operands() {
  for (Value* operand : operands_) {
    operand->remove_user(this);
  }
  operands_.clear();
}

// --- Graph --------------------------------------------------------------------------------------

std::vector<Value*> Graph::params() const {
  std::vector<Value*> out;
  out.reserve(params_.size());
  for (const auto& param : params_) {
    out.push_back(param.get());
  }
  return out;
}

std::vector<Node*> Graph::nodes() const {
  std::vector<Node*> out;
  out.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    out.push_back(node.get());
  }
  return out;
}

Value* Graph::add_param(std::string name, TensorType type) {
  params_.push_back(std::make_unique<Value>(std::move(name), std::move(type), nullptr, 0));
  return params_.back().get();
}

Node* Graph::add_node(std::string op) {
  nodes_.push_back(std::make_unique<Node>(this, std::move(op)));
  return nodes_.back().get();
}

Node* Graph::insert_node(std::string op, Node* before) {
  if (before == nullptr) {
    return add_node(std::move(op));
  }
  const auto position = std::find_if(
      nodes_.begin(), nodes_.end(), [before](const auto& held) { return held.get() == before; });
  TESSIQ_CHECK(position != nodes_.end(), "insert_node before a node from another graph");
  const auto inserted = nodes_.insert(position, std::make_unique<Node>(this, std::move(op)));
  return inserted->get();
}

void Graph::set_results(std::vector<Value*> results) { results_ = std::move(results); }

Result<void> Graph::erase(Node* node) {
  TESSIQ_CHECK(node != nullptr, "erase with a null node");

  for (const Value* result : node->results()) {
    if (result->has_users()) {
      return invalid_argument("cannot erase '" + node->op() + "': its result %" + result->name() +
                              " is still used by " + std::to_string(result->users().size()) +
                              " node(s)");
    }
    if (std::find(results_.begin(), results_.end(), result) != results_.end()) {
      return invalid_argument("cannot erase '" + node->op() + "': its result %" + result->name() +
                              " is a graph result");
    }
  }

  node->drop_operands();
  const auto it = std::find_if(
      nodes_.begin(), nodes_.end(), [node](const auto& held) { return held.get() == node; });
  if (it == nodes_.end()) {
    return invalid_argument("cannot erase a node that belongs to another graph");
  }
  nodes_.erase(it);
  return ok();
}

Result<void> Graph::replace_all_uses(Value* from, Value* to) {
  const bool both_present = from != nullptr && to != nullptr;
  TESSIQ_CHECK(both_present, "replace_all_uses with a null value");
  if (from == to) {
    return ok();
  }
  if (from->type() != to->type()) {
    return invalid_argument("cannot replace %" + from->name() + " of type " +
                            from->type().to_string() + " with %" + to->name() + " of type " +
                            to->type().to_string());
  }

  // Copied, because rewriting an operand mutates the list being walked.
  const std::vector<Node*> users = from->users();
  for (Node* user : users) {
    for (std::size_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) == from) {
        user->set_operand(i, to);
      }
    }
  }

  std::replace(results_.begin(), results_.end(), from, to);
  return ok();
}

std::string Graph::unique_name(std::string_view hint) const {
  std::string candidate(hint);
  int suffix = 0;
  while (name_taken(candidate)) {
    candidate = std::string(hint) + "_" + std::to_string(suffix++);
  }
  return candidate;
}

bool Graph::name_taken(std::string_view name) const {
  for (const auto& param : params_) {
    if (param->name() == name) {
      return true;
    }
  }
  for (const auto& node : nodes_) {
    for (const Value* result : node->results()) {
      if (result->name() == name) {
        return true;
      }
    }
  }
  return false;
}

// --- verify -------------------------------------------------------------------------------------

namespace {

/// Every operand is defined earlier, and every use list agrees with the operands that exist.
Result<void> verify_operands(const Node* node, const std::unordered_set<const Value*>& defined) {
  for (std::size_t i = 0; i < node->operand_count(); ++i) {
    const Value* operand = node->operand(i);
    if (operand == nullptr) {
      return internal_error("node '" + node->op() + "' has a null operand at " + std::to_string(i));
    }
    if (defined.count(operand) == 0) {
      return internal_error("node '" + node->op() + "' uses %" + operand->name() +
                            " before it is defined");
    }
    const auto& users = operand->users();
    if (std::find(users.begin(), users.end(), node) == users.end()) {
      return internal_error("use list for %" + operand->name() + " is missing '" + node->op() +
                            "'; a pass rewrote an operand without going through set_operand");
    }
  }
  return ok();
}

/// Results point back at their producer, and no name is used twice in the graph.
Result<void> verify_results(const Node* node,
                            std::unordered_set<const Value*>& defined,
                            std::unordered_set<std::string>& names) {
  for (const Value* result : node->results()) {
    if (result->producer() != node) {
      return internal_error("%" + result->name() + " does not point back at its producer");
    }
    if (!names.insert(result->name()).second) {
      return internal_error("duplicate value name %" + result->name());
    }
    defined.insert(result);
  }
  return ok();
}

} // namespace

Result<void> verify(const Graph& graph) {
  std::unordered_set<const Value*> defined;
  std::unordered_set<std::string> names;

  for (const Value* param : graph.params()) {
    if (!names.insert(param->name()).second) {
      return internal_error("duplicate value name %" + param->name());
    }
    defined.insert(param);
  }

  for (const Node* node : graph.nodes()) {
    if (node->graph() != &graph) {
      return internal_error("node '" + node->op() + "' belongs to another graph");
    }
    TESSIQ_RETURN_IF_ERROR(verify_operands(node, defined));
    TESSIQ_RETURN_IF_ERROR(verify_results(node, defined, names));
  }

  for (const Value* result : graph.results()) {
    if (defined.count(result) == 0) {
      return internal_error("graph result %" + result->name() + " is not defined in the graph");
    }
  }

  return ok();
}

} // namespace tessiq::ir
