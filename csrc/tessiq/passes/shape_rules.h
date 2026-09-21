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
/// A dynamic dimension makes the result dynamic, even against a 1, because a dimension that is not
/// known cannot be proven to be 1. That is the conservative direction: claiming a size that turns
/// out to be wrong puts a false type on a value every later pass trusts.
Result<ir::Shape> broadcast_shapes(const ir::Shape& a, const ir::Shape& b);

/// Whether a rule is known for `op`.
///
/// Most ops have none, and that is not a problem to report. The importer records a type for every
/// node from the tracing metadata, so a node this table has never heard of still carries a usable
/// type and its caller should leave it alone.
[[nodiscard]] bool has_shape_rule(std::string_view op);

/// The axes a reduction reduces, normalised against `rank` and sorted ascending.
///
/// Public because the shape rule and the kernel both need it and must not disagree. A kernel that
/// reduced different axes than the rule predicted would write the right number of elements in the
/// wrong places, which is the kind of wrong answer that looks plausible.
///
/// An absent list and an empty one both mean every axis: torch spells `sum(x)` as the dim_IntList
/// form with an empty list, while the `.default` spellings carry no list at all. A negative index
/// counts from the end, and a bare integer is accepted for IR written by hand.
Result<std::vector<std::size_t>>
reduced_axes(std::string_view op, const ir::AttributeMap& attributes, std::size_t rank);

/// The result types `op` produces from these operand types.
///
/// Precondition: `has_shape_rule(op)`. Asking about an op with no rule is a programmer error
/// rather than a runtime condition, which is why the two questions are separate calls.
///
/// An error means the operands are genuinely inconsistent with the op, such as shapes that cannot
/// broadcast.
///
/// Operand types are the tensor operands only. A scalar argument is an attribute rather than an
/// operand, so a binary op called with one tensor and one constant arrives here with one operand.
///
/// Attributes are the node's, because some rules need them. A reduction's result shape depends on
/// which dimensions it reduces and whether it keeps them, and both are attributes rather than
/// operands. They are a required argument rather than a defaulted one so that a caller cannot
/// quietly leave them out and get the answer for a reduction over every dimension.
Result<std::vector<ir::TensorType>> infer_result_types(std::string_view op,
                                                       const std::vector<ir::TensorType>& operands,
                                                       const ir::AttributeMap& attributes);

} // namespace tessiq::passes
