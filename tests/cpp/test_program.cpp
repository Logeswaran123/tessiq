// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Loading a compiled kernel and running it, checked against results computed on the host.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/stream.h"
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
/// A cubin built for another major architecture fails to load with an invalid image.
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
/// Buffers come from the runtime's allocator, as a tensor's would.
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

  EXPECT_TRUE(
      (*program)->launch(buffers, static_cast<std::int64_t>(kCount), kDefaultStream).has_value());
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

  // Not a multiple of the block size, so a missing bounds check would show.
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

  ASSERT_TRUE((*program)
                  ->launch({*input, *output}, static_cast<std::int64_t>(kOdd), kDefaultStream)
                  .has_value());
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

TEST(ProgramTest, TwoDifferentKernelsWithTheSameNameAreNotConfused) {
  TESSIQ_SKIP_WITHOUT_GPU();

  // Fused kernels are named tessiq_fused_<group>, so unrelated graphs often produce kernels with
  // the same name. These two also compile to the same length, so the loaded-module map has to key
  // on the contents.
  const auto [major, minor] = capability();
  const auto cubin_for = [major = major, minor = minor](const char* text) {
    return compile_cached(source_for(text), "tessiq_fused_0", major, minor);
  };

  auto negated = cubin_for(R"(graph @g(%a: f32[4096]) -> f32[4096] {
  %n = aten.neg.default(%a) : f32[4096]
  %r = aten.relu.default(%n) : f32[4096]
  return %r
}
)");
  auto absolute = cubin_for(R"(graph @g(%a: f32[4096]) -> f32[4096] {
  %n = aten.abs.default(%a) : f32[4096]
  %r = aten.relu.default(%n) : f32[4096]
  return %r
}
)");
  ASSERT_TRUE(negated.has_value());
  ASSERT_TRUE(absolute.has_value());
  ASSERT_NE(negated->data, absolute->data) << "the two kernels compiled to the same cubin";

  auto first = load_program(*negated, "tessiq_fused_0", 0);
  auto second = load_program(*absolute, "tessiq_fused_0", 0);
  ASSERT_TRUE(first.has_value()) << first.error().to_string();
  ASSERT_TRUE(second.has_value()) << second.error().to_string();
  EXPECT_NE(*first, *second) << "two different cubins shared one loaded module";
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

/// A kernel written by hand, the way a custom operation brings one.
///
/// It stages each element through dynamic shared memory, so a launch that gave it none would read
/// garbage, and it takes three scalars of different types, so a wrongly typed argument shows up
/// as a wrong number.
constexpr std::string_view kScaleRows = R"(
extern "C" __global__ void scale_rows(const float* in, float* out, long long columns,
                                      float factor, int bias) {
  extern __shared__ float staged[];
  const long long row = blockIdx.y;
  const long long column = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (column < columns) {
    staged[threadIdx.x] = in[row * columns + column] * factor;
  }
  __syncthreads();
  if (column < columns) {
    out[row * columns + column] = staged[threadIdx.x] + (float)bias;
  }
}
)";

TEST(ProgramTest, AHandWrittenKernelGetsItsGridSharedMemoryAndTypedArguments) {
  TESSIQ_SKIP_WITHOUT_GPU();

  constexpr std::uint32_t kRows = 3;
  constexpr std::uint32_t kColumns = 300;
  constexpr std::uint32_t kThreads = 128;
  std::vector<float> in(static_cast<std::size_t>(kRows) * kColumns);
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(i) * 0.5F;
  }

  auto program = compile_program(kScaleRows, "scale_rows", 0);
  ASSERT_TRUE(program.has_value()) << program.error().to_string();
  auto device = device_at(1);
  ASSERT_TRUE(device.has_value());
  Device& gpu = **device;
  const std::size_t bytes = in.size() * sizeof(float);
  auto input = gpu.raw_allocate(bytes);
  auto output = gpu.raw_allocate(bytes);
  ASSERT_TRUE(input.has_value() && output.has_value());
  ASSERT_TRUE(gpu.copy_host_to_device(*input, in.data(), bytes).has_value());

  LaunchShape shape;
  shape.grid = {(kColumns + kThreads - 1) / kThreads, kRows, 1};
  shape.block = {kThreads, 1, 1};
  shape.shared_bytes = kThreads * sizeof(float);
  const std::vector<KernelArgument> arguments{
      std::int64_t{kColumns}, KernelArgument{2.5F}, KernelArgument{std::int32_t{7}}};
  ASSERT_TRUE((*program)->launch({*input, *output}, arguments, shape, kDefaultStream).has_value());
  ASSERT_TRUE(gpu.synchronize().has_value());

  std::vector<float> got(in.size());
  ASSERT_TRUE(gpu.copy_device_to_host(got.data(), *output, bytes).has_value());
  for (std::size_t i = 0; i < in.size(); ++i) {
    ASSERT_FLOAT_EQ(got[i], (in[i] * 2.5F) + 7.0F) << "at " << i;
  }
  gpu.raw_deallocate(*input, bytes);
  gpu.raw_deallocate(*output, bytes);
}

TEST(ProgramTest, SharedMemoryPastWhatEveryKernelGetsIsAskedFor) {
  TESSIQ_SKIP_WITHOUT_GPU();

  // 64 KB, past the 48 KB default. Each thread reads back what another thread wrote, so all of it
  // has to exist.
  constexpr std::string_view kSource = R"(
extern "C" __global__ void through_shared(float* out, long long count) {
  extern __shared__ float staged[];
  for (long long i = threadIdx.x; i < count; i += blockDim.x) { staged[i] = (float)i; }
  __syncthreads();
  for (long long i = threadIdx.x; i < count; i += blockDim.x) { out[i] = staged[count - 1 - i]; }
}
)";
  constexpr std::size_t kFloats = 16384;
  auto program = compile_program(kSource, "through_shared", 0);
  ASSERT_TRUE(program.has_value()) << program.error().to_string();
  auto device = device_at(1);
  ASSERT_TRUE(device.has_value());
  Device& gpu = **device;
  auto output = gpu.raw_allocate(kFloats * sizeof(float));
  ASSERT_TRUE(output.has_value());

  LaunchShape shape;
  shape.block = {256, 1, 1};
  shape.shared_bytes = kFloats * sizeof(float);
  const auto launched =
      (*program)->launch({*output}, {std::int64_t{kFloats}}, shape, kDefaultStream);
  ASSERT_TRUE(launched.has_value()) << launched.error().to_string();
  ASSERT_TRUE(gpu.synchronize().has_value());

  std::vector<float> got(kFloats);
  ASSERT_TRUE(gpu.copy_device_to_host(got.data(), *output, kFloats * sizeof(float)).has_value());
  EXPECT_FLOAT_EQ(got.front(), static_cast<float>(kFloats - 1));
  EXPECT_FLOAT_EQ(got.back(), 0.0F);
  gpu.raw_deallocate(*output, kFloats * sizeof(float));
}

TEST(ProgramTest, SynchronizeWaitsForAKernelOnTheDefaultStream) {
  TESSIQ_SKIP_WITHOUT_GPU();

  // Spins on the device's own clock. It goes on the legacy default stream, as every launch does,
  // and the device's transfer stream stays idle, so a synchronize that waited only for transfers
  // would return at once.
  constexpr std::string_view kSource = R"(
extern "C" __global__ void spin(long long nanoseconds) {
  unsigned long long start;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  unsigned long long now = start;
  while (now - start < (unsigned long long)nanoseconds) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  }
}
)";
  constexpr std::int64_t kSpinNanoseconds = 50'000'000;
  auto program = compile_program(kSource, "spin", 0);
  ASSERT_TRUE(program.has_value()) << program.error().to_string();
  auto device = device_at(1);
  ASSERT_TRUE(device.has_value());
  Device& gpu = **device;
  ASSERT_TRUE(gpu.synchronize().has_value());

  const auto launched = (*program)->launch({}, {kSpinNanoseconds}, 1, kDefaultStream);
  ASSERT_TRUE(launched.has_value()) << launched.error().to_string();
  const auto start = std::chrono::steady_clock::now();
  ASSERT_TRUE(gpu.synchronize().has_value());
  const std::chrono::duration<double, std::milli> waited = std::chrono::steady_clock::now() - start;
  EXPECT_GE(waited.count(), 25.0) << "synchronize returned before a 50 ms kernel could finish";
}

TEST(ProgramTest, ALaunchShapeNoDeviceRunsIsRefused) {
  TESSIQ_SKIP_WITHOUT_GPU();

  auto program = compile_program(kScaleRows, "scale_rows", 0);
  ASSERT_TRUE(program.has_value()) << program.error().to_string();

  LaunchShape too_many;
  too_many.block = {2048, 1, 1};
  const auto refused = (*program)->launch({}, {}, too_many, kDefaultStream);
  ASSERT_FALSE(refused.has_value());
  EXPECT_NE(refused.error().message().find("1024"), std::string::npos);

  LaunchShape empty;
  empty.grid = {0, 1, 1};
  EXPECT_FALSE((*program)->launch({}, {}, empty, kDefaultStream).has_value());
}

} // namespace
} // namespace tessiq::codegen
