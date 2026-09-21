// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Loading a compiled kernel and running it. This is where generated code first produces a number,
// so the tests compare against one computed on the host rather than checking that nothing threw.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/program.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"
#include "tessiq/runtime/device.h"

namespace tessiq::codegen {
namespace {

constexpr std::size_t kCount = 4096;

#define TESSIQ_SKIP_WITHOUT_GPU()                                                                  \
  do {                                                                                             \
    if (!launcher_available() || device_count() < 2) {                                             \
      GTEST_SKIP() << "needs a CUDA device";                                                       \
    }                                                                                              \
  } while (false)

/// The compute capability of the first GPU, which is what a cubin has to be built for.
///
/// Asked rather than assumed. A cubin built for the wrong architecture fails at load with an
/// invalid image, and there is no forward compatibility that would let a low guess work anywhere.
std::pair<int, int> capability() {
  const auto found = device_capability(0);
  EXPECT_TRUE(found.has_value()) << (found.has_value() ? "" : found.error().to_string());
  return found.has_value() ? *found : std::pair<int, int>{0, 0};
}

std::string source_for(const std::string& ir) {
  auto graph = ir::parse(ir);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  passes::PassManager manager;
  EXPECT_TRUE(manager.add_from_pipeline("mark-supported,cluster,fuse-elementwise").has_value());
  EXPECT_TRUE(manager.run(**graph).has_value());

  auto kernel = extract_kernel(**graph, 0);
  EXPECT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  auto source = cuda_emitter().emit(*kernel);
  EXPECT_TRUE(source.has_value());
  return source.has_value() ? *source : std::string{};
}

/// Compiles, loads and launches, returning what the kernel wrote.
///
/// Inputs and the output live in device memory taken from the runtime's own allocator, which is
/// the same memory a tensor would sit in.
std::vector<float>
run(const std::string& ir, const std::string& name, const std::vector<std::vector<float>>& inputs) {
  const auto [major, minor] = capability();
  auto cubin = compile_cached(source_for(ir), name, major, minor);
  EXPECT_TRUE(cubin.has_value()) << (cubin.has_value() ? "" : cubin.error().to_string());
  if (!cubin.has_value()) {
    return {};
  }

  auto program = load_program(*cubin, name, /*ordinal=*/0);
  EXPECT_TRUE(program.has_value()) << (program.has_value() ? "" : program.error().to_string());
  if (!program.has_value()) {
    return {};
  }

  auto device = device_at(1);
  EXPECT_TRUE(device.has_value());
  Device& gpu = **device;

  std::vector<void*> buffers;
  for (const std::vector<float>& input : inputs) {
    auto memory = gpu.raw_allocate(input.size() * sizeof(float));
    EXPECT_TRUE(memory.has_value());
    EXPECT_TRUE(
        gpu.copy_host_to_device(*memory, input.data(), input.size() * sizeof(float)).has_value());
    buffers.push_back(*memory);
  }
  auto out = gpu.raw_allocate(kCount * sizeof(float));
  EXPECT_TRUE(out.has_value());
  buffers.push_back(*out);

  EXPECT_TRUE((*program)->launch(buffers, static_cast<std::int64_t>(kCount)).has_value());
  EXPECT_TRUE(gpu.synchronize().has_value());

  std::vector<float> result(kCount);
  EXPECT_TRUE(gpu.copy_device_to_host(result.data(), *out, kCount * sizeof(float)).has_value());

  for (void* buffer : buffers) {
    gpu.raw_deallocate(buffer, kCount * sizeof(float));
  }
  return result;
}

TEST(ProgramTest, TheMilestoneChainProducesTheSameNumbersAsTheHost) {
  TESSIQ_SKIP_WITHOUT_GPU();

  std::vector<float> a(kCount);
  std::vector<float> b(kCount);
  std::vector<float> c(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    a[i] = static_cast<float>(i % 17) - 8.0F;
    b[i] = static_cast<float>(i % 5) - 2.0F;
    c[i] = static_cast<float>(i % 9) - 4.0F;
  }

  const std::vector<float> got =
      run(R"(graph @g(%a: f32[4096], %b: f32[4096], %c: f32[4096]) -> f32[4096] {
  %m = aten.mul.Tensor(%a, %b) : f32[4096]
  %s = aten.add.Tensor(%m, %c) : f32[4096]
  %r = aten.relu.default(%s) : f32[4096]
  %g = aten.sigmoid.default(%r) : f32[4096]
  return %g
}
)",
          "tessiq_fused_0",
          {a, b, c});
  ASSERT_EQ(got.size(), kCount);

  for (std::size_t i = 0; i < kCount; ++i) {
    const float relu = std::max((a[i] * b[i]) + c[i], 0.0F);
    const float want = 1.0F / (1.0F + std::exp(-relu));
    EXPECT_NEAR(got[i], want, 1e-6F) << "at " << i;
  }
}

TEST(ProgramTest, ElementsThatDoNotFillTheLastBlockAreStillWritten) {
  TESSIQ_SKIP_WITHOUT_GPU();

  // 4096 divides the block size exactly, which is the case that would hide a missing bounds check.
  // This one does not.
  constexpr std::size_t kOdd = 4095;
  std::vector<float> a(kCount, 3.0F);
  for (std::size_t i = 0; i < kOdd; ++i) {
    a[i] = static_cast<float>(i);
  }

  const auto [major, minor] = capability();
  const std::string ir = R"(graph @g(%a: f32[4096]) -> f32[4096] {
  %n = aten.neg.default(%a) : f32[4096]
  %r = aten.abs.default(%n) : f32[4096]
  return %r
}
)";
  auto cubin = compile_cached(source_for(ir), "tessiq_fused_0", major, minor);
  ASSERT_TRUE(cubin.has_value());
  auto program = load_program(*cubin, "tessiq_fused_0", 0);
  ASSERT_TRUE(program.has_value());

  auto device = device_at(1);
  ASSERT_TRUE(device.has_value());
  Device& gpu = **device;

  auto input = gpu.raw_allocate(kCount * sizeof(float));
  auto output = gpu.raw_allocate(kCount * sizeof(float));
  ASSERT_TRUE(input.has_value());
  ASSERT_TRUE(output.has_value());
  ASSERT_TRUE(gpu.copy_host_to_device(*input, a.data(), kCount * sizeof(float)).has_value());

  ASSERT_TRUE((*program)->launch({*input, *output}, static_cast<std::int64_t>(kOdd)).has_value());
  ASSERT_TRUE(gpu.synchronize().has_value());

  std::vector<float> got(kCount);
  ASSERT_TRUE(gpu.copy_device_to_host(got.data(), *output, kCount * sizeof(float)).has_value());
  for (std::size_t i = 0; i < kOdd; ++i) {
    EXPECT_NEAR(got[i], std::fabs(-a[i]), 1e-6F) << "at " << i;
  }

  gpu.raw_deallocate(*input, kCount * sizeof(float));
  gpu.raw_deallocate(*output, kCount * sizeof(float));
}

TEST(ProgramTest, LoadingTwiceReusesTheModule) {
  TESSIQ_SKIP_WITHOUT_GPU();

  const auto [major, minor] = capability();
  auto cubin = compile_cached(source_for(R"(graph @g(%a: f32[4096]) -> f32[4096] {
  %n = aten.neg.default(%a) : f32[4096]
  %r = aten.relu.default(%n) : f32[4096]
  return %r
}
)"),
                              "tessiq_fused_0",
                              major,
                              minor);
  ASSERT_TRUE(cubin.has_value());

  const std::size_t before = loaded_program_count();
  auto first = load_program(*cubin, "tessiq_fused_0", 0);
  auto second = load_program(*cubin, "tessiq_fused_0", 0);
  ASSERT_TRUE(first.has_value()) << (first.has_value() ? "" : first.error().to_string());
  ASSERT_TRUE(second.has_value());

  EXPECT_EQ(*first, *second) << "the same cubin loaded twice";
  EXPECT_LE(loaded_program_count(), before + 1);
}

TEST(ProgramTest, AMissingFunctionIsReportedByName) {
  TESSIQ_SKIP_WITHOUT_GPU();

  const auto [major, minor] = capability();
  auto cubin = compile_cached(source_for(R"(graph @g(%a: f32[4096]) -> f32[4096] {
  %n = aten.neg.default(%a) : f32[4096]
  %r = aten.relu.default(%n) : f32[4096]
  return %r
}
)"),
                              "tessiq_fused_0",
                              major,
                              minor);
  ASSERT_TRUE(cubin.has_value());

  const auto program = load_program(*cubin, "not_in_this_module", 0);
  ASSERT_FALSE(program.has_value());
  EXPECT_NE(program.error().message().find("not_in_this_module"), std::string::npos)
      << program.error().message();
}

} // namespace
} // namespace tessiq::codegen
