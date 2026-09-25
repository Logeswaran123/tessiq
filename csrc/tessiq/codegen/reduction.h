// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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
  /// A sum divided by the number of elements reduced.
  kMean,
  kMax,
  kMin,
};

/// How the threads are laid over the problem, chosen so consecutive threads read consecutive
/// addresses.
enum class ReductionShape {
  /// One block per output, its threads walking the reduced axis together and combining through
  /// shared memory. For a reduced axis that is last, so one block reads adjacent elements.
  kBlockPerOutput,
  /// One thread per output, looping over the reduced axis alone. For kept axes after the reduced
  /// one, where adjacent outputs read adjacent addresses.
  kThreadPerOutput,
};

/// The three extents one reduction launch walks: the description's own when the shape was known,
/// otherwise derived from the tensor at launch.
struct ReductionExtents {
  std::int64_t outer = 0;
  std::int64_t reduce = 0;
  std::int64_t inner = 0;

  [[nodiscard]] std::int64_t outputs() const { return outer * inner; }

  [[nodiscard]] std::int64_t elements() const { return outer * reduce * inner; }

  /// Whether the three describe a reduction that can actually be walked.
  [[nodiscard]] bool are_usable() const { return outer > 0 && reduce > 0 && inner > 0; }
};

/// Splits `sizes` into the three extents, folding the axes in `[first, last]`.
///
/// An extent that is not known leaves its product `ir::kDynamic`, so a caller can tell the
/// difference between "one, because there are no axes there" and "not known yet".
[[nodiscard]] ReductionExtents
split_axes(const std::vector<std::int64_t>& sizes, std::size_t first, std::size_t last);

/// Blocks to launch, which follows from the kernel shape and the extents it is given.
[[nodiscard]] std::int64_t blocks_for(ReductionShape shape, const ReductionExtents& extents);

/// One reduction, described as the three extents a kernel needs.
///
/// A reduction over a contiguous range of axes is the same problem at every rank: the input viewed
/// as `[outer, reduce, inner]`, producing `[outer, inner]`. The last axis of `[4,8]` is outer 4,
/// reduce 8, inner 1; the first is outer 1, reduce 4, inner 8. Axes that are not a contiguous
/// range, such as the first and last of a rank-3 tensor, have no kernel and run on the host.
struct ReductionKernel {
  /// The generated function's name.
  std::string name;

  ReduceOp op = ReduceOp::kSum;

  ir::DType dtype = ir::DType::kF32;

  /// Elements before the reduced axes, which is how many independent slices there are.
  std::int64_t outer = 0;
  /// Elements along the reduced axes, which is how many go into each output element.
  std::int64_t reduce = 0;
  /// Elements after the reduced axes, and the stride between consecutive reduced elements.
  std::int64_t inner = 0;

  /// The kernel shape that reads this layout with coalesced loads.
  ReductionShape shape = ReductionShape::kBlockPerOutput;

  /// The reduced axes as an inclusive range, and the input's rank. A launch with unknown extents
  /// derives them from these and the tensor's sizes.
  std::size_t first_axis = 0;
  std::size_t last_axis = 0;
  std::size_t rank = 0;

  /// Whether the reduced axes stay in the result with size 1 (`keepdim=True`).
  bool keeps_dims = false;

  /// The IR names of the value read and the value written.
  std::string input;
  std::string output;

  /// What the description itself knows, with `ir::kDynamic` where it knows nothing.
  [[nodiscard]] ReductionExtents extents() const {
    ReductionExtents known;
    known.outer = outer;
    known.reduce = reduce;
    known.inner = inner;
    return known;
  }

  /// Whether any extent the launch needs arrives with the tensor rather than being known here.
  [[nodiscard]] bool has_dynamic_extent() const { return !extents().are_usable(); }

  /// How many elements the kernel writes. Meaningless unless the extents are known.
  [[nodiscard]] std::int64_t outputs() const { return outer * inner; }

  /// Blocks to launch, which follows from the shape. Meaningless unless the extents are known.
  [[nodiscard]] std::int64_t blocks() const { return blocks_for(shape, extents()); }
};

/// Describes `node` as a reduction kernel, or nothing if it cannot be one.
///
/// Nothing when the node is not a reduction or its axes are not a contiguous range; it then runs
/// on the host. An error means the node is malformed, such as naming an axis its operand lacks.
Result<std::optional<ReductionKernel>> extract_reduction(const ir::Node& node);

/// The nodes a reduction kernel could be built from, in graph order.
[[nodiscard]] std::vector<const ir::Node*> reduction_nodes(const ir::Graph& graph);

} // namespace tessiq::codegen
