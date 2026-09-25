// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {

/// Makes every node of operation `from` a node of operation `to`, where `to` has the types for it.
///
/// A custom operation that replaces a torch one gets one of these, ahead of `mark-supported`, so
/// the rest of the pipeline sees the custom operation.
///
/// Precondition: `to` takes the same operands and attributes as `from`, in the same order, with one
/// result. The custom operation's registration checks the two schemas.
/// Postcondition: no node of `from` is left whose operands and result all have a type in `dtypes`.
/// Every other node, and every value name, is as it was.
///
/// Legality: the author of `to` declared that it computes what `from` does. The pass only changes
/// which operation a node is, so every operand, attribute and use stays where it was. A node with a
/// type `to` was not written for is left alone, and runs however it would have without the pass.
class ReplaceOp final : public Pass {
 public:
  ReplaceOp(std::string name, std::string from, std::string to, std::vector<ir::DType> dtypes);

  [[nodiscard]] std::string_view name() const override { return name_; }

  [[nodiscard]] std::string_view description() const override { return description_; }

  Result<bool> run(ir::Graph& graph) override;

 private:
  /// Whether every tensor `node` reads or writes has a type in `dtypes_`.
  [[nodiscard]] bool has_the_types(const ir::Node& node) const;

  std::string name_;
  std::string description_;
  std::string from_;
  std::string to_;
  std::vector<ir::DType> dtypes_;
};

} // namespace tessiq::passes
