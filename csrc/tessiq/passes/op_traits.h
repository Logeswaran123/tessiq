// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <vector>

namespace tessiq::passes {

/// Whether `op` computes each output element from the operands at that same position.
///
/// This is what makes fusion possible: two elementwise operations in a row read and write the
/// same positions, so the second can consume the first's value in a register instead of reading
/// it back from memory. An operation that reads a position it is not writing, such as a matrix
/// multiply or a reduction, cannot be fused this way.
///
/// Distinct from `OpSupport::supports`, which asks whether the backend can run an operation at
/// all. The two tables happen to hold the same names today because everything implemented so far
/// is elementwise. They answer different questions and a matrix multiply belongs in one and not
/// the other.
[[nodiscard]] bool is_elementwise(std::string_view op);

/// Arguments `op` takes, counting constants that arrived as attributes. Zero when `op` is not
/// elementwise.
[[nodiscard]] int elementwise_arity(std::string_view op);

/// The scalar expression `op` computes, with `$0` and `$1` standing for its arguments.
///
/// Written in the subset of C++ that a device compiler and a host compiler both accept, with
/// `scalar_t` for the element type, so one spelling serves every target. Empty when `op` is not
/// elementwise.
[[nodiscard]] std::string_view elementwise_expression(std::string_view op);

/// Elementwise operation names, sorted, for diagnostics.
[[nodiscard]] std::vector<std::string> elementwise_ops();

} // namespace tessiq::passes
