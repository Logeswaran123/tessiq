// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace tessiq::passes {

/// What a pass may do with an operation.
enum class OpCategory {
  /// Computes output position i from the inputs at position i, so it can share a loop with its
  /// neighbours.
  kElementwise,
  /// Reads many positions to write one, along dimensions the node names rather than the table.
  kReduction,
  /// Reads positions it is not writing. Runs on its own however many kernels it has.
  kMatmul,
};

/// Everything known about one spelling of one operation.
///
/// One row per spelling rather than per operation, because a lookup is by the name a graph
/// carries and an imported graph carries the ATen spelling while a hand-written one carries the
/// short name.
struct OpInfo {
  /// The spelling a graph carries, which is what a lookup matches.
  std::string_view name;

  /// The operation this spelling is one of, as ops.toml names it. `aten.sum.dim_IntList` and
  /// `aten.sum.default` are both `sum`, so code that has to act on which operation it is can
  /// switch on this instead of matching every spelling of it.
  std::string_view canonical;

  OpCategory category;

  /// Arguments it takes, counting constants that arrived as attributes.
  int arity;

  /// The scalar computation, with `$0` and `$1` for the arguments. Empty unless elementwise.
  std::string_view expression;

  /// Whether the backend can run it. An operation without a kernel falls back to the host, which
  /// is correct and costs two transfers.
  bool has_kernel;
};

/// The row for `name`, or null when the table has never heard of it.
///
/// The table is generated from `ops.toml` by `tools/gen_ops.py`, so an operation cannot be marked
/// fusable without saying what it computes, or given a kernel in one place and not another. Two
/// hand-written tables holding overlapping facts is how an operation gets half-added, and that
/// happened here before this one existed.
[[nodiscard]] const OpInfo* find_op(std::string_view name);

/// Every spelling in the table, sorted.
[[nodiscard]] std::vector<std::string> op_names();

/// Spellings whose operation has a kernel, sorted.
[[nodiscard]] std::vector<std::string> ops_with_kernels();

/// How many rows the table holds, which is one per aten spelling rather than one per operation.
[[nodiscard]] std::size_t spelling_count();

} // namespace tessiq::passes
