// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/result.h"

namespace tessiq::passes {

/// An operation added while the program runs, rather than generated from ops.toml. The custom
/// operation layer registers one per custom operation when tessiq is imported. `find_op` then
/// finds it like any generated row.
struct OpDefinition {
  /// The spelling a graph carries, as an FX node's target prints it: `tessiq.fast_gelu.default`.
  /// Letters, digits, underscores and dots only, which is what the textual IR can read.
  std::string name;

  /// The operation's name without its overload, such as `tessiq::fast_gelu`.
  std::string canonical;

  /// Elementwise, an expression fused into loops, or custom, a kernel of its own.
  OpCategory category = OpCategory::kElementwise;

  /// Tensor operands it takes. At most ten for an elementwise operation, because the expression
  /// names them `$0` to `$9`.
  int arity = 0;

  /// Every parameter's name in schema order, tensors and scalars alike. Only a custom kernel's
  /// call uses them: the importer keeps a keyword argument under its name, and the name finds its
  /// position again.
  std::vector<std::string> parameters;

  /// The scalar computation in ops.toml's form: `$0` for the first operand, constants written
  /// `scalar_t(1.702)`, and `exp` rather than `expf` so one expression serves every element type.
  /// Empty for a custom kernel.
  std::string expression;

  /// The element types it has a kernel for.
  std::vector<ir::DType> dtypes;
};

/// Whether two definitions describe the same operation, field by field.
[[nodiscard]] bool same_definition(const OpDefinition& a, const OpDefinition& b);

/// Adds `definition`, or returns the row already registered under the same name when it is the
/// same definition, so that registering at import time is safe to repeat.
///
/// Refused when the name shadows a generated operation or is already registered differently, has
/// a character the textual IR cannot read, or when the expression reads an operand the operation
/// does not have, skips one it does, or writes a floating-point constant bare. A bare `1.702` is a
/// double and turns a float32 kernel into double arithmetic.
///
/// The returned pointer is valid for the life of the program.
[[nodiscard]] Result<const OpInfo*> register_op(const OpDefinition& definition);

/// The registered row for `name`, or null. `find_op` asks this after the generated table.
[[nodiscard]] const OpInfo* find_registered_op(std::string_view name);

/// Everything registered under `name`, or null. Valid for the life of the program.
[[nodiscard]] const OpDefinition* find_registered_definition(std::string_view name);

/// Every registered spelling, sorted.
[[nodiscard]] std::vector<std::string> registered_op_names();

} // namespace tessiq::passes
