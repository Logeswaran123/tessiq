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

/// Every element type. An operation generated from ops.toml accepts them all here, because whether
/// a kernel exists for a type is the emitter's question and it answers it when it compiles.
inline constexpr std::uint32_t kEveryDType = ~std::uint32_t{0};

/// What a pass may do with an operation.
enum class OpCategory {
  /// Computes output position i from the inputs at position i, so it can share a loop with its
  /// neighbours.
  kElementwise,
  /// Reads many positions to write one, along dimensions the node names rather than the table.
  kReduction,
  /// Rearranges indices and computes nothing. The device runs these as metadata, so there is no
  /// kernel and nothing to fuse: the result shares its input's storage.
  kMovement,
  /// Reads positions it is not writing. Runs on its own however many kernels it has.
  kMatmul,
  /// Reads its input's layout and writes the same values packed. Computes nothing, which makes it
  /// look like a movement operation, but a movement operation shares its input's storage and this
  /// one does not: it costs a pass over the data. That distinction decides placement, because a
  /// unit holding nothing but views is not worth putting on a device and a unit holding a copy is.
  kCopy,
  /// Reads a window of its input for each output element. A library call rather than generated
  /// code: cuDNN chooses among several algorithms by the shapes and the card, which is the reason
  /// to call it rather than write one.
  kConvolution,
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

  /// The element types it has a kernel for, one bit per `ir::DType`. Every type for an operation
  /// from ops.toml. A registered operation names its own, because its expression was written for
  /// them and nobody has checked it against any other.
  std::uint32_t dtypes = kEveryDType;

  [[nodiscard]] constexpr bool accepts(ir::DType dtype) const {
    return (dtypes & dtype_bit(dtype)) != 0;
  }
};

/// The row for `name`, or null when nothing has heard of it.
///
/// The table is generated from `ops.toml` by `tools/gen_ops.py`, so an operation cannot be marked
/// fusable without saying what it computes, or given a kernel in one place and not another. Two
/// hand-written tables holding overlapping facts is how an operation gets half-added, and that
/// happened here before this one existed. An operation added while the program runs, through
/// `register_op` in op_registry.h, is found here too, after the table, so every pass sees it.
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
