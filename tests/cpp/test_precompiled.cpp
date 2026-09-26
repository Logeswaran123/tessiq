// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Precompiled host code: what a call hands it, the registry, and the check that the code was
// built for the card. With nvcc, test_scale.nvcc.cu is built into this binary and runs for real.

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

#include "tessiq/codegen/precompiled.h"
#include "tessiq/codegen/program.h"
#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"

namespace tessiq::codegen {
namespace {

DeviceTensor tensor(std::vector<std::int64_t> sizes, void* data = nullptr) {
  DeviceTensor made;
  made.data = data;
  made.sizes = std::move(sizes);
  made.type = ElementType::kFloat32;
  return made;
}

TEST(DeviceTensorTest, CountsElementsAndReadsAxesFromEitherEnd) {
  const DeviceTensor t = tensor({2, 3, 4});
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.size(0), 2);
  EXPECT_EQ(t.size(-1), 4);
  EXPECT_EQ(t.size(3), 0);
  EXPECT_EQ(tensor({}).numel(), 1);
}

TEST(CallTest, NumbersTensorsAndScalarsSeparately) {
  // (Tensor x, float eps, Tensor w, int n, bool b) arrives as two tensors and three scalars.
  const Call call({tensor({4}), tensor({8})},
                  {Scalar{0.5}, Scalar{std::int64_t{7}}, Scalar{true}},
                  tensor({4}),
                  nullptr,
                  0,
                  {});
  EXPECT_EQ(call.input_count(), 2U);
  EXPECT_EQ(call.input(1).size(0), 8);
  EXPECT_DOUBLE_EQ(call.real(0), 0.5);
  EXPECT_EQ(call.integer(1), 7);
  EXPECT_TRUE(call.flag(2));
}

TEST(CallTest, KeepsTheWorkspaceItTakesForTheCallerToGiveBack) {
  std::vector<std::size_t> asked;
  std::vector<char> memory(64);
  Call call({}, {}, tensor({1}), nullptr, 0, [&](std::size_t bytes) -> void* {
    asked.push_back(bytes);
    return memory.data();
  });
  EXPECT_EQ(call.workspace(64), memory.data());
  EXPECT_EQ(asked, std::vector<std::size_t>{64});
  EXPECT_EQ(call.workspaces(), std::vector<void*>{memory.data()});
  EXPECT_FALSE(call.failed());
}

TEST(CallTest, WorkspaceThatCannotBeHadFailsTheCall) {
  Call call({}, {}, tensor({1}), nullptr, 3, [](std::size_t) -> void* { return nullptr; });
  EXPECT_EQ(call.workspace(1024), nullptr);
  EXPECT_TRUE(call.failed());
  EXPECT_NE(call.reason().find("1024 bytes"), std::string::npos) << call.reason();
  EXPECT_NE(call.reason().find("CUDA device 3"), std::string::npos) << call.reason();
  EXPECT_TRUE(call.workspaces().empty());
}

TEST(CallTest, DecliningKeepsTheReason) {
  Call call({}, {}, tensor({1}), nullptr, 0, {});
  call.decline("a weight that broadcasts");
  EXPECT_TRUE(call.declined());
  EXPECT_FALSE(call.failed());
  EXPECT_EQ(call.reason(), "a weight that broadcasts");
}

TEST(PrecompiledTest, ArchitectureNamesReadNvccsList) {
  EXPECT_EQ(architecture_names("890"), "sm_89");
  EXPECT_EQ(architecture_names("800,1200"), "sm_80, sm_120");
  EXPECT_EQ(architecture_names(""), "no architecture");
}

TEST(PrecompiledTest, HostCodeThatDoesNotLoadIsRefusedNamingBothArchitectures) {
  const Precompiled entry{"somewhere_else", nullptr, [](int) { return false; }, "800"};
  const Result<void> checked = check_runs_on(entry, 0);
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().code(), ErrorCode::kExternal);
  const std::string& message = checked.error().message();
  EXPECT_NE(message.find("tessiq::somewhere_else was built for sm_80"), std::string::npos)
      << message;
  EXPECT_NE(message.find("CUDA device 0 is "), std::string::npos) << message;
  if (launcher_available() && device_count() > 1) {
    const auto capability = device_capability(0);
    ASSERT_TRUE(capability.has_value());
    const std::string card = std::to_string(capability->first) + std::to_string(capability->second);
    EXPECT_NE(message.find("is sm_" + card), std::string::npos) << message;
    EXPECT_NE(message.find("CMAKE_CUDA_ARCHITECTURES including " + card), std::string::npos)
        << message;
  }
}

TEST(PrecompiledTest, UnknownHostCodeIsNotFound) {
  EXPECT_EQ(find_precompiled("no_such_operation"), nullptr);
}

#ifdef TESSIQ_TEST_PRECOMPILED

#define TESSIQ_SKIP_WITHOUT_GPU()                                                                  \
  do {                                                                                             \
    if (!launcher_available() || device_count() < 2) {                                             \
      GTEST_SKIP() << "needs a CUDA device";                                                       \
    }                                                                                              \
  } while (false)

TEST(PrecompiledTest, TheMacroRegistersTheFileAndWhatItWasBuiltFor) {
  const Precompiled* entry = find_precompiled("test_scale");
  ASSERT_NE(entry, nullptr);
  EXPECT_NE(entry->run, nullptr);
  EXPECT_NE(entry->loads, nullptr);
  EXPECT_FALSE(entry->architectures.empty());
}

TEST(PrecompiledTest, HostCodeBuiltForThisCardLoadsOnIt) {
  TESSIQ_SKIP_WITHOUT_GPU();
  const Precompiled* entry = find_precompiled("test_scale");
  ASSERT_NE(entry, nullptr);
  const Result<void> checked = check_runs_on(*entry, 0);
  EXPECT_TRUE(checked.has_value()) << checked.error().to_string();
}

/// Runs test_scale over `values` with `factor`, returning the call and what it wrote.
std::pair<std::vector<float>, bool> scale(const std::vector<float>& values, double factor) {
  Device& gpu = **device_at(1);
  CachingAllocator& allocator = allocator_at(1);
  const std::size_t bytes = values.size() * sizeof(float);
  void* in = *allocator.allocate(bytes);
  void* out = *allocator.allocate(bytes);
  EXPECT_TRUE(gpu.copy_host_to_device(in, values.data(), bytes).has_value());

  const auto count = static_cast<std::int64_t>(values.size());
  Call call({tensor({count}, in)},
            {Scalar{factor}},
            tensor({count}, out),
            nullptr,
            0,
            [&allocator](std::size_t size) -> void* {
              const Result<void*> block = allocator.allocate(size);
              return block.has_value() ? *block : nullptr;
            });
  const Result<void> ran = run_precompiled(*find_precompiled("test_scale"), call);
  EXPECT_TRUE(ran.has_value()) << ran.error().to_string();
  EXPECT_TRUE(gpu.synchronize().has_value());
  EXPECT_EQ(call.workspaces().size(), call.declined() ? 0U : 1U);
  for (void* block : call.workspaces()) {
    allocator.deallocate(block);
  }

  std::vector<float> written(values.size());
  EXPECT_TRUE(gpu.copy_device_to_host(written.data(), out, bytes).has_value());
  allocator.deallocate(in);
  allocator.deallocate(out);
  return {written, call.declined()};
}

TEST(PrecompiledTest, HostCodeRunsItsLaunchesThroughWorkspace) {
  TESSIQ_SKIP_WITHOUT_GPU();
  std::vector<float> values(1000);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
  }
  const auto [written, declined] = scale(values, 2.5);
  EXPECT_FALSE(declined);
  for (std::size_t i = 0; i < values.size(); ++i) {
    ASSERT_FLOAT_EQ(written[i], values[i] * 2.5F) << "at " << i;
  }
}

TEST(PrecompiledTest, HostCodeCanDeclineACall) {
  TESSIQ_SKIP_WITHOUT_GPU();
  const auto [written, declined] = scale({1.0F, 2.0F}, -1.0);
  EXPECT_TRUE(declined);
}

#endif // TESSIQ_TEST_PRECOMPILED

} // namespace
} // namespace tessiq::codegen
