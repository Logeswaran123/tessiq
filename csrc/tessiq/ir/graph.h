// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::ir {

class Graph;
class Node;

/// A constant carried on a node: `%c = const() {value = 1.0} : f32[]`.
///
/// Small on purpose. Anything that needs a richer attribute is usually asking for an operand.
///
/// `std::monostate` is Python's None, which real graphs carry: a convolution with no bias passes
/// None where the bias goes, and dropping it would renumber every argument after it. It is first
/// so that a default-constructed attribute is the absent one.
using Attribute = std::
    variant<std::monostate, std::int64_t, double, bool, std::string, std::vector<std::int64_t>>;

/// Renders an attribute as the textual IR spells it.
std::string attribute_to_string(const Attribute& value);

/// Attributes in name order, so printing a graph is deterministic and golden tests are stable.
using AttributeMap = std::map<std::string, Attribute, std::less<>>;

/// An SSA value: a graph input, or one result of one node.
///
/// Values are owned by the graph or the node that produces them and are referred to by pointer
/// everywhere else. A value outlives every use of it, because removing a node with live uses is
/// rejected by the graph rather than left to the caller.
class Value {
 public:
  Value(std::string name, TensorType type, Node* producer, std::size_t result_index)
      : name_(std::move(name)), type_(std::move(type)), producer_(producer),
        result_index_(result_index) {}

  [[nodiscard]] const std::string& name() const { return name_; }

  void set_name(std::string name) { name_ = std::move(name); }

  [[nodiscard]] const TensorType& type() const { return type_; }

  void set_type(TensorType type) { type_ = std::move(type); }

  /// The node producing this value, or null for a graph input.
  [[nodiscard]] Node* producer() const { return producer_; }

  /// Which of the producer's results this is.
  [[nodiscard]] std::size_t result_index() const { return result_index_; }

  /// Nodes consuming this value. Maintained by the graph, so it is never stale.
  [[nodiscard]] const std::vector<Node*>& users() const { return users_; }

  [[nodiscard]] bool has_users() const { return !users_.empty(); }

 private:
  friend class Node;
  friend class Graph;

  void add_user(Node* user);
  void remove_user(Node* user);

  std::string name_;
  TensorType type_;
  Node* producer_;
  std::size_t result_index_;
  std::vector<Node*> users_;
};

/// One operation: an op name, ordered operands, ordered results, and attributes.
class Node {
 public:
  Node(Graph* graph, std::string op) : graph_(graph), op_(std::move(op)) {}

  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  Node(Node&&) = delete;
  Node& operator=(Node&&) = delete;

  ~Node() = default;

  [[nodiscard]] const std::string& op() const { return op_; }

  /// Makes this node a different operation over the same operands, attributes and results.
  ///
  /// For a rewrite that swaps one operation for another that computes the same thing. Everything
  /// that refers to the node or to its results stays valid, which replacing it with a new node
  /// would not give: the results would have to be renamed back to what the executor knows them by.
  void set_op(std::string op) { op_ = std::move(op); }

  [[nodiscard]] Graph* graph() const { return graph_; }

  [[nodiscard]] const std::vector<Value*>& operands() const { return operands_; }

  [[nodiscard]] Value* operand(std::size_t i) const { return operands_.at(i); }

  [[nodiscard]] std::size_t operand_count() const { return operands_.size(); }

  [[nodiscard]] std::vector<Value*> results() const;

  [[nodiscard]] Value* result(std::size_t i) const { return results_.at(i).get(); }

  [[nodiscard]] std::size_t result_count() const { return results_.size(); }

  [[nodiscard]] const AttributeMap& attributes() const { return attributes_; }

  /// The attribute named `name`, or null if the node has none.
  [[nodiscard]] const Attribute* attribute(std::string_view name) const;

  void set_attribute(std::string name, Attribute value);

  /// Appends an operand, recording this node as one of its users.
  void append_operand(Value* value);

  /// Points operand `i` at `value`, updating both values' user lists.
  void set_operand(std::size_t i, Value* value);

  /// Appends a result. The node owns it.
  Value* append_result(std::string name, TensorType type);

  /// Drops every operand, so the node holds no references. Used before erasing it.
  void drop_operands();

 private:
  friend class Graph;

  Graph* graph_;
  std::string op_;
  std::vector<Value*> operands_;
  std::vector<std::unique_ptr<Value>> results_;
  AttributeMap attributes_;
};

/// A function: named inputs, a list of nodes in order, and results.
///
/// Nodes are held in execution order. Passes that reorder are responsible for keeping every
/// operand defined before its use, which the verifier checks.
class Graph {
 public:
  explicit Graph(std::string name) : name_(std::move(name)) {}

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(Graph&&) = delete;

  ~Graph() = default;

  [[nodiscard]] const std::string& name() const { return name_; }

  [[nodiscard]] std::vector<Value*> params() const;

  [[nodiscard]] const std::vector<Value*>& results() const { return results_; }

  [[nodiscard]] std::vector<Node*> nodes() const;

  [[nodiscard]] std::size_t node_count() const { return nodes_.size(); }

  /// Adds a graph input.
  Value* add_param(std::string name, TensorType type);

  /// Appends a node. Operands and results are added to it afterwards.
  Node* add_node(std::string op);

  /// Inserts a node immediately before `before`, or appends when `before` is null.
  ///
  /// A pass that creates a replacement has to put it where its uses can see it. Appending would
  /// leave the definition after the use, which the verifier rejects.
  Node* insert_node(std::string op, Node* before);

  void set_results(std::vector<Value*> results);

  /// Removes `node`, which must have no users of any of its results.
  ///
  /// Returns kInvalidArgument rather than asserting: a pass asking to erase a live node has made
  /// an ordinary mistake about the graph, and the message names what still uses it.
  Result<void> erase(Node* node);

  /// Replaces every use of `from` with `to`, including graph results.
  Result<void> replace_all_uses(Value* from, Value* to);

  /// A name no value in this graph currently uses, derived from `hint`.
  [[nodiscard]] std::string unique_name(std::string_view hint) const;

 private:
  [[nodiscard]] bool name_taken(std::string_view name) const;

  std::string name_;
  std::vector<std::unique_ptr<Value>> params_;
  std::vector<std::unique_ptr<Node>> nodes_;
  std::vector<Value*> results_;
};

/// Checks the structural invariants a pass is allowed to assume.
///
/// Every operand defined before its use, use lists agreeing with actual operands, results
/// belonging to nodes in the graph, no duplicate names. Run after every pass in debug builds:
/// a pass that corrupts the graph should be caught in its own test, not three passes later.
Result<void> verify(const Graph& graph);

} // namespace tessiq::ir
