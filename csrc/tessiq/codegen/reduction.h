// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// What a reduction combines its elements with.
enum class ReduceOp {
  kSum,
  /// A sum divided by how many elements went into it, which is the only one that needs the count
  /// after the combining is done.
  kMean,
  kMax,
  kMin,
};

/// How the threads are laid over the problem.
///
/// The two cases have opposite access patterns and the right answer is whichever one reads
/// consecutive addresses from consecutive threads.
enum class ReductionShape {
  /// One block per output, its threads walking the reduced axis together and combining through
  /// shared memory. Right when the reduced axis is last, because then the elements one block
  /// reads sit next to each other.
  kBlockPerOutput,
  /// One thread per output, looping over the reduced axis by itself. Right when kept axes follow
  /// the reduced one, because then consecutive threads read consecutive addresses and a block
  /// per output would have every thread striding past the others.
  kThreadPerOutput,
};

/// One reduction, described as the three extents a kernel needs.
///
/// A reduction over a contiguous range of axes is the same problem at every rank: an input viewed
/// as `[outer, reduce, inner]`, producing `[outer, inner]`. Reducing the last axis of `[4,8]` is
/// outer 4, reduce 8, inner 1. Reducing the first is outer 1, reduce 4, inner 8. Reducing
/// everything is outer 1, inner 1. So the kernel takes three numbers and does not care about rank,
/// and the index arithmetic is the same in every case.
///
/// Axes that are not a contiguous range, such as the first and last of a rank-3 tensor, cannot be
/// written this way. Those have no kernel and run on the host.
struct ReductionKernel {
  /// The generated function's name, which is also the cubin cache's key.
  std::string name;

  ReduceOp op = ReduceOp::kSum;

  ir::DType dtype = ir::DType::kF32;

  /// Elements before the reduced axes, which is how many independent slices there are.
  std::int64_t outer = 0;
  /// Elements along the reduced axes, which is how many go into each output element.
  std::int64_t reduce = 0;
  /// Elements after the reduced axes, and the stride between consecutive reduced elements.
  std::int64_t inner = 0;

  /// Which kernel shape reads this layout without fighting the memory system.
  ReductionShape shape = ReductionShape::kBlockPerOutput;

  /// The IR names of the value read and the value written.
  std::string input;
  std::string output;

  /// How many elements the kernel writes.
  [[nodiscard]] std::int64_t outputs() const { return outer * inner; }

  /// Blocks to launch, which follows from the shape.
  [[nodiscard]] std::int64_t blocks() const;
};

/// Describes `node` as a reduction kernel, or nothing if it cannot be one.
///
/// Nothing is not a failure. An operation that is not a reduction, or one whose axes are not a
/// contiguous range, simply has no kernel and runs on the host. An error means the node claims to
/// be a reduction and contradicts itself, such as naming an axis its operand does not have.
Result<std::optional<ReductionKernel>> extract_reduction(const ir::Node& node);

/// The nodes a reduction kernel could be built from, in graph order.
[[nodiscard]] std::vector<const ir::Node*> reduction_nodes(const ir::Graph& graph);

} // namespace tessiq::codegen
