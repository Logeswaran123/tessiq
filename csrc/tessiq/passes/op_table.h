// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/types.h"

namespace tessiq::passes {

/// The bit for `dtype` in `OpInfo::dtypes`.
[[nodiscard]] constexpr std::uint32_t dtype_bit(ir::DType dtype) {
  return std::uint32_t{1} << static_cast<unsigned>(dtype);
}

/// Every element type. An operation from ops.toml accepts them all unless it names its own there.
/// For a generated loop the emitter decides per type when it compiles; a matmul or convolution
/// calls a library that takes only some types, so those name theirs.
inline constexpr std::uint32_t kEveryDType = ~std::uint32_t{0};

/// What a pass may do with an operation.
enum class OpCategory {
  /// Computes output position i from the inputs at position i, so it can share a loop with its
  /// neighbours.
  kElementwise,
  /// Reads many positions to write one, along dimensions the node names rather than the table.
  kReduction,
  /// Rearranges indices and computes nothing. Runs as metadata on the device: the result shares
  /// its input's storage, so there is no kernel and nothing to fuse.
  kMovement,
  /// Reads positions it is not writing. Runs on its own however many kernels it has.
  kMatmul,
  /// Reads its input's layout and writes the same values packed. Unlike a movement operation it
  /// does not share storage and costs a pass over the data, which matters for placement: a unit of
  /// nothing but views is not worth moving to the device, and a unit holding a copy is.
  kCopy,
  /// Reads a window of its input for each output element. A cuDNN call, which picks an algorithm
  /// for the shapes and the card.
  kConvolution,
  /// A custom operation with a kernel of its own. Runs alone, like a matmul. The compiler knows its
  /// operands and types but not what it computes, so it neither fuses it nor derives its shape.
  kCustom,
};

/// Everything known about one spelling of one operation.
///
/// One row per spelling: an imported graph carries the ATen spelling and a hand-written one the
/// short name, and a lookup is by whichever the graph carries.
struct OpInfo {
  /// The spelling a graph carries, which is what a lookup matches.
  std::string_view name;

  /// The operation this spelling is one of, as ops.toml names it: `aten.sum.dim_IntList` and
  /// `aten.sum.default` are both `sum`.
  std::string_view canonical;

  OpCategory category;

  /// Arguments it takes, counting constants that arrived as attributes.
  int arity;

  /// The scalar computation, with `$0` and `$1` for the arguments. Empty unless elementwise.
  std::string_view expression;

  /// Whether the backend can run it. One without a kernel runs on the host.
  bool has_kernel;

  /// The element types it has a kernel for, one bit per `ir::DType`.
  std::uint32_t dtypes = kEveryDType;

  [[nodiscard]] constexpr bool accepts(ir::DType dtype) const {
    return (dtypes & dtype_bit(dtype)) != 0;
  }
};

/// The row for `name`, or null. Looks in the table generated from `ops.toml` by
/// `tools/gen_ops.py`, then among operations added with `register_op` (op_registry.h).
[[nodiscard]] const OpInfo* find_op(std::string_view name);

/// The generated row for `name`, or null. Registered operations are not looked at.
[[nodiscard]] const OpInfo* find_builtin_op(std::string_view name);

/// Every spelling in the generated table, sorted. Registered operations are not in it.
[[nodiscard]] std::vector<std::string> op_names();

/// Spellings whose operation has a kernel, sorted.
[[nodiscard]] std::vector<std::string> ops_with_kernels();

/// How many rows the table holds, which is one per aten spelling rather than one per operation.
[[nodiscard]] std::size_t spelling_count();

} // namespace tessiq::passes
