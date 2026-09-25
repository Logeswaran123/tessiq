// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Describing a reduction as [outer, reduce, inner], which decides whether it gets a kernel. Needs
// no GPU.

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/reduction.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/text.h"

namespace tessiq::codegen {
namespace {

/// A graph with one sum: `shape` in, `result` out, reducing `axes`.
std::string sum_over(std::string_view shape, std::string_view result, std::string_view axes) {
  return "graph @g(%a: f32" + std::string(shape) + ") -> f32" + std::string(result) + " {\n" +
         "  %s = aten.sum.dim_IntList(%a) {arg1 = " + std::string(axes) + "} : f32" +
         std::string(result) + "\n  return %s\n}\n";
}

/// The one reduction `text` describes, as a value.
///
/// Fails the test and returns a default-constructed kernel when there is none, so callers can read
/// fields directly.
ReductionKernel extract(std::string_view text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  if (!graph.has_value()) {
    return {};
  }
  const auto nodes = reduction_nodes(**graph);
  EXPECT_EQ(nodes.size(), 1U) << "expected exactly one reduction in the graph";
  if (nodes.size() != 1) {
    return {};
  }
  auto kernel = extract_reduction(*nodes.front());
  EXPECT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  if (!kernel.has_value() || !kernel->has_value()) {
    EXPECT_TRUE(kernel.has_value() && kernel->has_value()) << "no kernel shape for this graph";
    return {};
  }
  return **std::move(kernel);
}

/// Whether `text`'s reduction has a kernel shape at all, for the cases where it should not.
bool has_kernel_shape(std::string_view text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value());
  if (!graph.has_value()) {
    return false;
  }
  const auto nodes = reduction_nodes(**graph);
  if (nodes.size() != 1) {
    return false;
  }
  auto kernel = extract_reduction(*nodes.front());
  EXPECT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  return kernel.has_value() && kernel->has_value();
}

/// The CUDA source for `text`'s reduction.
std::string source_for(std::string_view text) {
  const auto emitted = cuda_emitter().emit(extract(text));
  EXPECT_TRUE(emitted.has_value()) << (emitted.has_value() ? "" : emitted.error().to_string());
  return emitted.has_value() ? *emitted : std::string{};
}

TEST(ReductionTest, TheLastAxisIsContiguous) {
  const ReductionKernel parsed = extract(sum_over("[4,8]", "[4]", "[1]"));
  EXPECT_EQ(parsed.outer, 4);
  EXPECT_EQ(parsed.reduce, 8);
  // Consecutive reduced elements sit next to each other, so the stride between them is one.
  EXPECT_EQ(parsed.inner, 1);
  EXPECT_EQ(parsed.outputs(), 4);
}

TEST(ReductionTest, AnInputLayoutAssignCopiedIsAskedForUnderItsSourcesName) {
  // The executor packs a reduction's input itself, so the kernel names the value layout-assign's
  // copy was made from and reads it packed.
  const ReductionKernel parsed = extract(R"(graph @g(%a: f32[4,8:1,4]) -> f32[4] {
  %a.contiguous = tessiq.contiguous(%a) {tessiq.supported = false} : f32[4,8]
  %s = aten.sum.dim_IntList(%a.contiguous) {arg1 = [1]} : f32[4]
  return %s
}
)");
  EXPECT_EQ(parsed.input, "a");
  EXPECT_EQ(parsed.outer, 4);
  EXPECT_EQ(parsed.reduce, 8);
  EXPECT_EQ(parsed.inner, 1);
}

TEST(ReductionTest, TheFirstAxisIsStrided) {
  const ReductionKernel parsed = extract(sum_over("[4,8]", "[8]", "[0]"));
  EXPECT_EQ(parsed.outer, 1);
  EXPECT_EQ(parsed.reduce, 4);
  // Eight kept elements sit between two reduced ones.
  EXPECT_EQ(parsed.inner, 8);
  EXPECT_EQ(parsed.outputs(), 8);
}

TEST(ReductionTest, AMiddleAxisSetsBothExtents) {
  const ReductionKernel parsed = extract(sum_over("[4,8,3]", "[4,3]", "[1]"));
  EXPECT_EQ(parsed.outer, 4);
  EXPECT_EQ(parsed.reduce, 8);
  EXPECT_EQ(parsed.inner, 3);
  EXPECT_EQ(parsed.outputs(), 12);
}

TEST(ReductionTest, EveryAxisCollapsesToOneOutput) {
  const ReductionKernel parsed = extract(sum_over("[4,8,3]", "[]", "[]"));
  EXPECT_EQ(parsed.outer, 1);
  EXPECT_EQ(parsed.reduce, 96);
  EXPECT_EQ(parsed.inner, 1);
}

TEST(ReductionTest, ARangeOfAxesGoesAtOnce) {
  // Two adjacent axes fold into one reduced extent.
  const ReductionKernel parsed = extract(sum_over("[4,8,3]", "[4]", "[1, 2]"));
  EXPECT_EQ(parsed.outer, 4);
  EXPECT_EQ(parsed.reduce, 24);
  EXPECT_EQ(parsed.inner, 1);
}

TEST(ReductionTest, AxesThatAreNotAdjacentHaveNoKernel) {
  // Reducing the first and last axis leaves the kept axis in the middle, which [outer, reduce,
  // inner] cannot describe. It runs on the host instead.
  EXPECT_FALSE(has_kernel_shape(sum_over("[4,8,3]", "[8]", "[0, 2]")));
}

TEST(ReductionTest, AnUnknownExtentLeavesItsProductUnknown) {
  // The extents are kernel arguments, so an unknown shape still gets a kernel. The description
  // carries the reduced axes; the numbers come from the tensor.
  const ReductionKernel kernel = extract("graph @g(%a: f32[4,?]) -> f32[4] {\n"
                                         "  %s = aten.sum.dim_IntList(%a) {arg1 = [1]} : f32[4]\n"
                                         "  return %s\n}\n");
  EXPECT_TRUE(kernel.has_dynamic_extent());
  EXPECT_EQ(kernel.outer, 4);
  EXPECT_EQ(kernel.reduce, ir::kDynamic);
  EXPECT_EQ(kernel.inner, 1);
  EXPECT_EQ(kernel.first_axis, 1U);
  EXPECT_EQ(kernel.last_axis, 1U);
  EXPECT_EQ(kernel.rank, 2U);
  // Still block-per-output, since the reduced axis is last.
  EXPECT_EQ(kernel.shape, ReductionShape::kBlockPerOutput);
}

TEST(ReductionTest, TheExtentsAreWorkedOutFromTheSizesTheLaunchIsGiven) {
  // The split the launch does once it has a tensor.
  const ReductionExtents extents = split_axes({8, 4, 64, 16}, 2, 2);
  EXPECT_EQ(extents.outer, 32);
  EXPECT_EQ(extents.reduce, 64);
  EXPECT_EQ(extents.inner, 16);
  EXPECT_EQ(extents.elements(), 8 * 4 * 64 * 16);
  EXPECT_EQ(extents.outputs(), 32 * 16);
  EXPECT_TRUE(extents.are_usable());
}

TEST(ReductionTest, AnUnknownSizeLeavesTheSplitUnusable) {
  const ReductionExtents extents = split_axes({ir::kDynamic, 8}, 1, 1);
  EXPECT_EQ(extents.outer, ir::kDynamic);
  EXPECT_FALSE(extents.are_usable()) << "a launch cannot be given a number nobody knows";
}

TEST(ReductionTest, ARankZeroInputHasNoKernel) {
  EXPECT_FALSE(has_kernel_shape(sum_over("[]", "[]", "[]")));
}

TEST(ReductionTest, ANegativeAxisCountsFromTheEnd) {
  const ReductionKernel kernel = extract("graph @g(%a: f32[4,8,3]) -> f32[4,8] {\n"
                                         "  %m = aten.mean.dim(%a) {arg1 = [-1]} : f32[4,8]\n"
                                         "  return %m\n}\n");
  EXPECT_EQ(kernel.outer, 32);
  EXPECT_EQ(kernel.reduce, 3);
  EXPECT_EQ(kernel.inner, 1);
  EXPECT_EQ(kernel.op, ReduceOp::kMean);
}

TEST(ReductionTest, AnElementwiseOperationIsNotOne) {
  auto graph = ir::parse(
      "graph @g(%a: f32[4]) -> f32[4] {\n  %r = aten.relu.default(%a) : f32[4]\n  return %r\n}\n");
  ASSERT_TRUE(graph.has_value());
  EXPECT_TRUE(reduction_nodes(**graph).empty());
}

TEST(ReductionTest, TheShapeFollowsWhetherAKeptAxisComesAfter) {
  // Decided here rather than in the emitter, because the launch has to agree with it.
  EXPECT_EQ(extract(sum_over("[4,8]", "[4]", "[1]")).shape, ReductionShape::kBlockPerOutput);
  EXPECT_EQ(extract(sum_over("[4,8]", "[8]", "[0]")).shape, ReductionShape::kThreadPerOutput);
  EXPECT_EQ(extract(sum_over("[4,8,3]", "[4,3]", "[1]")).shape, ReductionShape::kThreadPerOutput);
  EXPECT_EQ(extract(sum_over("[4,8,3]", "[]", "[]")).shape, ReductionShape::kBlockPerOutput);
}

TEST(ReductionTest, TheBlockCountFollowsTheShape) {
  // A block per output launches one block each. A thread per output fits kBlockSize outputs in a
  // block and rounds up.
  const ReductionKernel per_block = extract(sum_over("[4,8]", "[4]", "[1]"));
  EXPECT_EQ(per_block.blocks(), 4);

  const ReductionKernel per_thread = extract(sum_over("[600,8]", "[8]", "[0]"));
  EXPECT_EQ(per_thread.outputs(), 8);
  EXPECT_EQ(per_thread.blocks(), 1);

  const ReductionKernel wide = extract(sum_over("[4,1000]", "[1000]", "[0]"));
  EXPECT_EQ(wide.outputs(), 1000);
  EXPECT_EQ(wide.blocks(), 4) << "1000 outputs over 256 threads rounds up to four blocks";
}

// --- the emitted source ---------------------------------------------------------------------

TEST(ReductionEmitterTest, ASumStartsFromZeroAndAMaxFromTheData) {
  // A sum starts at zero. A max starts from an element, since there is no per-type lowest value
  // without math.h.
  const std::string sum = source_for(sum_over("[4,8]", "[4]", "[1]"));
  EXPECT_NE(sum.find("scalar_t acc = (scalar_t)0;"), std::string::npos) << sum;

  const std::string max = source_for("graph @g(%a: f32[4,8]) -> f32[4] {\n"
                                     "  %m = aten.amax.default(%a) {arg1 = [1]} : f32[4]\n"
                                     "  return %m\n}\n");
  EXPECT_NE(max.find("scalar_t acc = base[0];"), std::string::npos) << max;
  EXPECT_NE(max.find("isnan(a)"), std::string::npos)
      << "max must test for a NaN, since torch propagates one and a plain compare does not";
}

TEST(ReductionEmitterTest, AMeanDividesByWhatItSummed) {
  const std::string source = source_for("graph @g(%a: f32[4,8]) -> f32[4] {\n"
                                        "  %m = aten.mean.dim(%a) {arg1 = [1]} : f32[4]\n"
                                        "  return %m\n}\n");
  EXPECT_NE(source.find("scratch[0] / (scalar_t)reduce"), std::string::npos) << source;
}

TEST(ReductionEmitterTest, TheSourceNeedsNoIncludes) {
  // NVRTC does not include math.h, so INFINITY and HUGE_VALF are undefined there.
  const std::string source = source_for(sum_over("[4,8]", "[4]", "[1]"));
  EXPECT_EQ(source.find("#include"), std::string::npos) << source;
  EXPECT_EQ(source.find("INFINITY"), std::string::npos) << source;
}

TEST(ReductionEmitterTest, AThreadPerOutputKernelHasNoSharedMemory) {
  // Each thread finishes its own output, so there is no shared buffer and no __syncthreads.
  const std::string source = source_for(sum_over("[4,8]", "[8]", "[0]"));
  EXPECT_EQ(source.find("__shared__"), std::string::npos) << source;
  EXPECT_EQ(source.find("__syncthreads"), std::string::npos) << source;
  EXPECT_NE(source.find("blockIdx.x * (long long)blockDim.x + threadIdx.x"), std::string::npos)
      << source;
}

TEST(ReductionEmitterTest, TheSharedBufferMatchesTheBlockSize) {
  // The block has kBlockSize threads, each writing its own slot.
  const std::string source = source_for(sum_over("[4,8]", "[4]", "[1]"));
  EXPECT_NE(source.find("__shared__ scalar_t scratch[" + std::to_string(kBlockSize) + "]"),
            std::string::npos)
      << source;
}

} // namespace
} // namespace tessiq::codegen
