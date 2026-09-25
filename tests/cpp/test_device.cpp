// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"

namespace tessiq {
namespace {

/// Every device index, so a test written once runs against both backends.
std::vector<int> all_indices() {
  std::vector<int> indices;
  indices.reserve(static_cast<std::size_t>(device_count()));
  for (int i = 0; i < device_count(); ++i) {
    indices.push_back(i);
  }
  return indices;
}

TEST(RegistryTest, IndexZeroIsAlwaysTheCpuReference) {
  ASSERT_GE(device_count(), 1);
  auto device = device_at(0);
  ASSERT_TRUE(device.has_value()) << device.error().to_string();
  EXPECT_EQ((*device)->kind(), DeviceKind::kCpu);
  EXPECT_EQ((*device)->ordinal(), 0);
}

TEST(RegistryTest, CudaDevicesFollowInDriverOrder) {
  for (int index = 1; index < device_count(); ++index) {
    auto device = device_at(index);
    ASSERT_TRUE(device.has_value());
    EXPECT_EQ((*device)->kind(), DeviceKind::kCuda);
    EXPECT_EQ((*device)->ordinal(), index - 1) << "index " << index;
  }
}

TEST(RegistryTest, OutOfRangeIndexIsRejectedWithACountInTheMessage) {
  auto device = device_at(device_count());
  ASSERT_FALSE(device.has_value());
  EXPECT_EQ(device.error().code(), ErrorCode::kInvalidArgument);
  EXPECT_NE(device.error().message().find("cpu reference"), std::string::npos)
      << device.error().message();
}

TEST(RegistryTest, NegativeIndexIsRejected) { EXPECT_FALSE(device_at(-1).has_value()); }

TEST(RegistryTest, KindNamesAreStable) {
  EXPECT_EQ(device_kind_name(DeviceKind::kCpu), "cpu");
  EXPECT_EQ(device_kind_name(DeviceKind::kCuda), "cuda");
}

// --- the current index -------------------------------------------------------------------------

class CurrentIndexTest : public ::testing::Test {
 protected:
  void TearDown() override { ASSERT_TRUE(set_current_device_index(0).has_value()); }
};

TEST_F(CurrentIndexTest, DefaultsToZero) {
  EXPECT_EQ(current_device_index(), 0);
  EXPECT_EQ(current_device().kind(), DeviceKind::kCpu);
}

TEST_F(CurrentIndexTest, FollowsTheIndexItIsSetTo) {
  for (const int index : all_indices()) {
    ASSERT_TRUE(set_current_device_index(index).has_value());
    EXPECT_EQ(current_device_index(), index);
    EXPECT_EQ(&current_device(), *device_at(index));
  }
}

TEST_F(CurrentIndexTest, AFailedSwitchLeavesTheIndexAlone) {
  ASSERT_TRUE(set_current_device_index(0).has_value());
  EXPECT_FALSE(set_current_device_index(device_count()).has_value());
  EXPECT_EQ(current_device_index(), 0) << "a rejected index must not take effect";
}

// --- backends ------------------------------------------------------------------------------------

TEST(BackendTest, RoundTripsDataThroughEveryDevice) {
  const std::array<std::uint32_t, 4> source{1, 2, 3, 4};

  for (const int index : all_indices()) {
    Device& device = **device_at(index);
    auto ptr = device.raw_allocate(256);
    ASSERT_TRUE(ptr.has_value()) << "index " << index << ": " << ptr.error().to_string();

    std::array<std::uint32_t, 4> destination{};
    ASSERT_TRUE(device.copy_host_to_device(*ptr, source.data(), sizeof(source)).has_value());
    ASSERT_TRUE(device.copy_device_to_host(destination.data(), *ptr, sizeof(source)).has_value());
    ASSERT_TRUE(device.synchronize().has_value());

    EXPECT_EQ(source, destination) << "index " << index;
    device.raw_deallocate(*ptr, 256);
  }
}

TEST(BackendTest, EveryDeviceReportsItsMemory) {
  for (const int index : all_indices()) {
    const auto info = (*device_at(index))->memory_info();
    ASSERT_TRUE(info.has_value()) << "index " << index << ": " << info.error().to_string();
    EXPECT_GT(info->total_bytes, 0U) << "index " << index;
    EXPECT_LE(info->free_bytes, info->total_bytes) << "index " << index;
  }
}

TEST(BackendTest, DeviceToDeviceCopyPreservesData) {
  const std::array<std::uint32_t, 4> source{9, 8, 7, 6};

  for (const int index : all_indices()) {
    Device& device = **device_at(index);
    auto first = device.raw_allocate(256);
    auto second = device.raw_allocate(256);
    ASSERT_TRUE(first.has_value() && second.has_value());

    std::array<std::uint32_t, 4> destination{};
    ASSERT_TRUE(device.copy_host_to_device(*first, source.data(), sizeof(source)).has_value());
    ASSERT_TRUE(device.copy_device_to_device(*second, *first, sizeof(source)).has_value());
    ASSERT_TRUE(
        device.copy_device_to_host(destination.data(), *second, sizeof(source)).has_value());
    ASSERT_TRUE(device.synchronize().has_value());

    EXPECT_EQ(source, destination) << "index " << index;
    device.raw_deallocate(*first, 256);
    device.raw_deallocate(*second, 256);
  }
}

TEST(BackendTest, AllocationsMeetTheAlignmentContract) {
  for (const int index : all_indices()) {
    Device& device = **device_at(index);
    auto ptr = device.raw_allocate(1024);
    ASSERT_TRUE(ptr.has_value());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*ptr) % kDeviceAlignment, 0U) << "index " << index;
    device.raw_deallocate(*ptr, 1024);
  }
}

TEST(BackendTest, ZeroByteCopyIsANoOpRatherThanAnError) {
  for (const int index : all_indices()) {
    EXPECT_TRUE((*device_at(index))->copy_host_to_device(nullptr, nullptr, 0).has_value());
  }
}

TEST(BackendTest, NullPointerCopyIsRejected) {
  for (const int index : all_indices()) {
    auto result = (*device_at(index))->copy_device_to_host(nullptr, nullptr, 16);
    ASSERT_FALSE(result.has_value()) << "index " << index;
    EXPECT_EQ(result.error().code(), ErrorCode::kInvalidArgument);
  }
}

// --- allocators --------------------------------------------------------------------------------

TEST(PerDeviceAllocatorTest, EachDeviceHasItsOwn) {
  for (const int index : all_indices()) {
    for (const int other : all_indices()) {
      if (index != other) {
        EXPECT_NE(&allocator_at(index), &allocator_at(other));
      }
    }
  }
}

TEST(PerDeviceAllocatorTest, StatisticsDoNotLeakBetweenDevices) {
  if (device_count() < 2) {
    GTEST_SKIP() << "needs a second device";
  }

  const std::size_t before = allocator_at(0).stats().allocated_bytes;

  auto ptr = allocator_at(1).allocate(8192);
  ASSERT_TRUE(ptr.has_value()) << ptr.error().to_string();
  EXPECT_EQ(allocator_at(0).stats().allocated_bytes, before)
      << "allocating on one device changed another's statistics";
  EXPECT_GT(allocator_at(1).stats().allocated_bytes, 0U);

  allocator_at(1).deallocate(*ptr);
}

TEST(PerDeviceAllocatorTest, CurrentAllocatorFollowsTheCurrentIndex) {
  for (const int index : all_indices()) {
    ASSERT_TRUE(set_current_device_index(index).has_value());
    EXPECT_EQ(&current_allocator(), &allocator_at(index));
  }
  ASSERT_TRUE(set_current_device_index(0).has_value());
}

} // namespace
} // namespace tessiq
