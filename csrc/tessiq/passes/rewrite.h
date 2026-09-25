// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/pass.h"
#include "tessiq/support/result.h"

namespace tessiq::passes {

/// What a pattern binds, found by `check_pattern`.
struct PatternShape {
  /// The pattern's inputs, which bind to whatever tensor sits in their place, by name.
  std::vector<std::string> inputs;

  /// The scalars it captures, written `"$name"` as an attribute's value, by name.
  std::vector<std::string> captures;
};

/// Checks that `pattern` is something a `Rewrite` can match, and returns what it binds.
///
/// A pattern is a graph in the textual IR that returns one value computed by a node, and every
/// node leads to that value. An attribute whose value is the string `"$name"` captures the matched
/// node's value. Types are not compared, so they can be written as anything, such as `f32[?]`.
[[nodiscard]] Result<PatternShape> check_pattern(const ir::Graph& pattern);

/// One argument of the operation a match becomes.
struct RewriteArgument {
  /// The pattern input or the captured scalar that supplies it.
  std::string name;

  /// A tensor becomes an operand, a scalar the attribute `arg<position>`, as the importer spells a
  /// positional scalar.
  bool tensor = true;
};

/// Replaces every match of a pattern with one call to an operation.
///
/// Each custom operation with a pattern gets one, `rewrite-<name>`.
///
/// A match binds each pattern input to one value of the graph, the same value every time the input
/// appears, and each pattern node to one node with the same operation and the same attributes. An
/// integer and a floating-point attribute match when they are the same number, since `x ** 2` and
/// `x ** 2.0` are the same power. The operands of `add` and `mul` match in either order.
///
/// Precondition: the pattern passed `check_pattern`, and `arguments` names each of its inputs and
/// captures once, in the order `op` takes them.
/// Postcondition: every match is one node of `op`, whose result has the name and the type of the
/// value the pattern returned. The node records the names of the values it absorbed as
/// `tessiq.absorbed`, so a kernel for it can say which nodes of the original graph it computes.
///
/// Legality: the pattern's author declares it computes what `op` computes. Every value inside a
/// match other than the last stops existing, so none may be read outside the match or returned.
/// That also leaves training graphs alone, since AOTAutograd returns each value the backward graph
/// needs. A match whose tensors have a type `op` does not declare is left alone.
class Rewrite final : public Pass {
 public:
  Rewrite(std::string name,
          std::shared_ptr<const ir::Graph> pattern,
          std::string op,
          std::vector<RewriteArgument> arguments,
          std::vector<ir::DType> dtypes);

  [[nodiscard]] std::string_view name() const override { return name_; }

  [[nodiscard]] std::string_view description() const override { return description_; }

  Result<bool> run(ir::Graph& graph) override;

 private:
  std::string name_;
  std::string description_;
  std::shared_ptr<const ir::Graph> pattern_;
  std::string op_;
  std::vector<RewriteArgument> arguments_;
  std::vector<ir::DType> dtypes_;
};

} // namespace tessiq::passes
