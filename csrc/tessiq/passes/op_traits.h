// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <vector>

namespace tessiq::passes {

/// Whether `op` computes each output element from the operands at that same position.
///
/// Two such operations in a row can share a loop, the second reading the first's value from a
/// register. Whether the backend can run an operation at all is `OpSupport::supports`.
[[nodiscard]] bool is_elementwise(std::string_view op);

/// Arguments `op` takes, counting constants that arrived as attributes. Zero when `op` is not
/// elementwise.
[[nodiscard]] int elementwise_arity(std::string_view op);

/// The scalar expression `op` computes, with `$0` and `$1` standing for its arguments.
///
/// C++ that device and host compilers both accept, with `scalar_t` for the element type. Empty
/// when `op` is not elementwise.
[[nodiscard]] std::string_view elementwise_expression(std::string_view op);

/// Elementwise operation names, sorted, for diagnostics.
[[nodiscard]] std::vector<std::string> elementwise_ops();

} // namespace tessiq::passes
