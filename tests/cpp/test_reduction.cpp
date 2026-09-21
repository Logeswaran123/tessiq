// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Describing a reduction as [outer, reduce, inner], which is the decision that says whether it
// gets a kernel at all. Needs no GPU: it is arithmetic on shapes.

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
/// Returns a default-constructed kernel when there is none, having already failed the test that
/// there is. Returning a value rather than an optional keeps every assertion below reading the
/// fields directly, which is the thing each test is about.
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
  // Two adjacent axes are one reduced extent, which is why the shape is three numbers and not one
  // per axis.
  const ReductionKernel parsed = extract(sum_over("[4,8,3]", "[4]", "[1, 2]"));
  EXPECT_EQ(parsed.outer, 4);
  EXPECT_EQ(parsed.reduce, 24);
  EXPECT_EQ(parsed.inner, 1);
}

TEST(ReductionTest, AxesThatAreNotAdjacentHaveNoKernel) {
  // Reducing the first and last axis leaves the kept axis in the middle, which [outer, reduce,
  // inner] cannot describe. Not an error: it runs on the host and the answer is still right.
  EXPECT_FALSE(has_kernel_shape(sum_over("[4,8,3]", "[8]", "[0, 2]")));
}

TEST(ReductionTest, AnUnknownExtentHasNoKernel) {
  // The extents are baked into the launch, so a shape that is not known yet cannot be launched.
  EXPECT_FALSE(has_kernel_shape(sum_over("[4,?]", "[4]", "[1]")));
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
  // Which kernel is written is decided here rather than in the emitter, because the launch has to
  // agree with it: one shape wants a block per output and the other a thread per output.
  EXPECT_EQ(extract(sum_over("[4,8]", "[4]", "[1]")).shape, ReductionShape::kBlockPerOutput);
  EXPECT_EQ(extract(sum_over("[4,8]", "[8]", "[0]")).shape, ReductionShape::kThreadPerOutput);
  EXPECT_EQ(extract(sum_over("[4,8,3]", "[4,3]", "[1]")).shape, ReductionShape::kThreadPerOutput);
  EXPECT_EQ(extract(sum_over("[4,8,3]", "[]", "[]")).shape, ReductionShape::kBlockPerOutput);
}

TEST(ReductionTest, TheBlockCountFollowsTheShape) {
  // A block per output launches one block each. A thread per output packs kBlockSize of them
  // into a block, so it launches far fewer and has to round up.
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
  // A sum's identity is a constant. A max has none that works for every type without math.h, so
  // it starts from an element instead, which is idempotent under max and propagates a NaN.
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
  // It goes to NVRTC, which does not include the CUDA headers the way nvcc does. INFINITY and
  // HUGE_VALF are both undefined there, and finding that out at run time is a compile error
  // inside a model.
  const std::string source = source_for(sum_over("[4,8]", "[4]", "[1]"));
  EXPECT_EQ(source.find("#include"), std::string::npos) << source;
  EXPECT_EQ(source.find("INFINITY"), std::string::npos) << source;
}

TEST(ReductionEmitterTest, AThreadPerOutputKernelHasNoSharedMemory) {
  // Each thread finishes its own output, so there is nothing to combine between threads and no
  // __syncthreads at all. A shared buffer here would be a sign the wrong template was chosen.
  const std::string source = source_for(sum_over("[4,8]", "[8]", "[0]"));
  EXPECT_EQ(source.find("__shared__"), std::string::npos) << source;
  EXPECT_EQ(source.find("__syncthreads"), std::string::npos) << source;
  EXPECT_NE(source.find("blockIdx.x * (long long)blockDim.x + threadIdx.x"), std::string::npos)
      << source;
}

TEST(ReductionEmitterTest, TheSharedBufferMatchesTheBlockSize) {
  // The block is launched with kBlockSize threads and each one writes its own slot, so a buffer
  // declared at a different size would be written past.
  const std::string source = source_for(sum_over("[4,8]", "[4]", "[1]"));
  EXPECT_NE(source.find("__shared__ scalar_t scratch[" + std::to_string(kBlockSize) + "]"),
            std::string::npos)
      << source;
}

} // namespace
} // namespace tessiq::codegen
