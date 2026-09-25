// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::passes {

/// Broadcasts two shapes, right-aligned, as every array library does it.
///
/// Dimensions are compared from the last one backwards. Two dimensions are compatible when they
/// are equal or one of them is 1, and the result takes the larger. A rank difference is treated as
/// leading dimensions of 1 on the shorter shape.
///
/// A dynamic dimension makes the result dynamic, even against a 1, since it cannot be proven to be
/// 1. Claiming a size that turns out wrong would put a false type on the value.
Result<ir::Shape> broadcast_shapes(const ir::Shape& a, const ir::Shape& b);

/// Whether a rule is known for `op`.
///
/// Most ops have none, which is fine: the importer records every node's type from the tracing
/// metadata.
[[nodiscard]] bool has_shape_rule(std::string_view op);

/// The axes a reduction reduces, normalised against `rank` and sorted ascending.
///
/// Public because the shape rule and the reduction kernel both use it and must agree.
///
/// An absent list and an empty one both mean every axis: torch spells `sum(x)` as the dim_IntList
/// form with an empty list, while the `.default` spellings carry no list at all. A negative index
/// counts from the end, and a bare integer is accepted for IR written by hand.
Result<std::vector<std::size_t>>
reduced_axes(std::string_view op, const ir::AttributeMap& attributes, std::size_t rank);

/// The result types `op` produces from these operand types.
///
/// Precondition: `has_shape_rule(op)`. An error means the operands are inconsistent with the op,
/// such as shapes that cannot broadcast.
///
/// Operand types are the tensor operands only; a scalar argument is an attribute, so a binary op
/// with one constant arrives with one operand. Attributes are required because some rules need
/// them: a reduction's shape depends on its axes and whether it keeps them.
Result<std::vector<ir::TensorType>> infer_result_types(std::string_view op,
                                                       const std::vector<ir::TensorType>& operands,
                                                       const ir::AttributeMap& attributes);

} // namespace tessiq::passes
