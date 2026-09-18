// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <gtest/gtest.h>

#include "tessera/runtime/device.h"

namespace tessera {
namespace {

TEST(DeviceTest, CpuBackendIsAlwaysAvailable) {
  auto device = get_device(DeviceKind::kCpu);
  ASSERT_TRUE(device.has_value()) << device.error().to_string();
  EXPECT_EQ((*device)->kind(), DeviceKind::kCpu);
  EXPECT_GE((*device)->device_count(), 1);
}

TEST(DeviceTest, DefaultsToCpu) { EXPECT_EQ(current_device().kind(), DeviceKind::kCpu); }

TEST(DeviceTest, KindNamesAreStable) {
  EXPECT_EQ(device_kind_name(DeviceKind::kCpu), "cpu");
  EXPECT_EQ(device_kind_name(DeviceKind::kCuda), "cuda");
}

TEST(DeviceTest, UnavailableBackendReportsWhyRatherThanCrashing) {
  auto device = get_device(DeviceKind::kCuda);
  if (!device.has_value()) {
    EXPECT_EQ(device.error().code(), ErrorCode::kUnsupported);
    EXPECT_NE(device.error().message().find("cuda"), std::string::npos);
  }
}

TEST(DeviceTest, SwitchingToAnUnavailableBackendLeavesTheCurrentOneInPlace) {
  const DeviceKind before = current_device().kind();

  if (!get_device(DeviceKind::kCuda).has_value()) {
    EXPECT_FALSE(set_current_device(DeviceKind::kCuda).has_value());
    EXPECT_EQ(current_device().kind(), before) << "a failed switch must not change the backend";
  }
}

TEST(DeviceTest, RoundTripsDataThroughDeviceMemory) {
  Device& device = current_device();

  auto ptr = device.raw_allocate(256);
  ASSERT_TRUE(ptr.has_value()) << ptr.error().to_string();

  std::array<std::uint32_t, 4> source{1, 2, 3, 4};
  std::array<std::uint32_t, 4> destination{};

  ASSERT_TRUE(device.copy_host_to_device(*ptr, source.data(), sizeof(source)).has_value());
  ASSERT_TRUE(device.copy_device_to_host(destination.data(), *ptr, sizeof(source)).has_value());
  ASSERT_TRUE(device.synchronize().has_value());

  EXPECT_EQ(source, destination);
  device.raw_deallocate(*ptr, 256);
}

TEST(DeviceTest, ZeroByteCopyIsANoOpRatherThanAnError) {
  EXPECT_TRUE(current_device().copy_host_to_device(nullptr, nullptr, 0).has_value());
}

TEST(DeviceTest, NullPointerCopyIsRejected) {
  auto result = current_device().copy_device_to_host(nullptr, nullptr, 16);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace tessera
