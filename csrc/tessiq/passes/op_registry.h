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

/// An operation added while the program runs, rather than generated from ops.toml.
///
/// This is how a custom operation reaches the compiler without anyone editing tessiq's C++ or
/// rebuilding it. Once registered it is found by `find_op` like any generated row, so marking,
/// clustering, fusion and the emitter all treat it the way they treat `add`.
struct OpDefinition {
  /// The spelling a graph carries, which is what an FX node's target prints as, such as
  /// `demo.fast_gelu.default`. Letters, digits, underscores and dots only, because the textual IR
  /// reads an operation's name up to its opening parenthesis and a colon would end it early.
  std::string name;

  /// What the operation is called apart from its overload, such as `demo::fast_gelu`.
  std::string canonical;

  /// Only elementwise so far: an expression the emitter drops into a fused loop.
  OpCategory category = OpCategory::kElementwise;

  /// Tensor operands it takes, between one and ten, because the expression names them `$0` to
  /// `$9`.
  int arity = 0;

  /// The scalar computation in the same form ops.toml uses: `$0` for the first operand, a constant
  /// written `scalar_t(1.702)`, and maths functions spelled `exp` rather than `expf` so that one
  /// expression serves every element type.
  std::string expression;

  /// The element types the expression was written for. Float32 and float64.
  std::vector<ir::DType> dtypes;
};

/// Whether two definitions describe the same operation, field by field.
[[nodiscard]] bool same_definition(const OpDefinition& a, const OpDefinition& b);

/// Adds `definition`, or returns the row already registered under the same name when it is the
/// same definition, so that registering at import time is safe to repeat.
///
/// Refused, with a message saying what to change, when the name shadows a generated operation or
/// is already registered differently, when the name has a character the textual IR cannot read,
/// when the expression reads an operand the operation does not have or never reads one it does,
/// and when the expression writes a floating-point constant bare. `1.702` is a double in C++, and
/// one double anywhere in a float32 expression turns the whole of it into double arithmetic. The
/// answer is still right, and on a consumer card the kernel runs several times slower with nothing
/// to say why, which is why this is refused here rather than found in a benchmark.
///
/// The returned pointer is valid for the life of the program.
[[nodiscard]] Result<const OpInfo*> register_op(const OpDefinition& definition);

/// The registered row for `name`, or null. `find_op` asks this after the generated table.
[[nodiscard]] const OpInfo* find_registered_op(std::string_view name);

/// Every registered spelling, sorted.
[[nodiscard]] std::vector<std::string> registered_op_names();

} // namespace tessiq::passes
