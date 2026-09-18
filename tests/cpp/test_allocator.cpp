// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <vector>

#include "tessera/runtime/allocator.h"
#include "tessera/runtime/device.h"

namespace tessera {
namespace {

/// Counts driver calls, so tests can assert caching behaviour rather than infer it, and can force
/// allocation failure without needing to exhaust real memory.
class FakeDevice final : public Device {
 public:
  [[nodiscard]] DeviceKind kind() const override { return DeviceKind::kCpu; }

  [[nodiscard]] int device_count() const override { return 1; }

  Result<void*> raw_allocate(std::size_t bytes) override {
    if (fail_next_allocation) {
      fail_next_allocation = false;
      return resource_exhausted("fake device is out of memory");
    }
    raw_allocate_calls++;
    live_bytes += bytes;
    // Honours the kDeviceAlignment contract, like any real backend must.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc)
    return std::aligned_alloc(kDeviceAlignment, bytes);
  }

  void raw_deallocate(void* ptr, std::size_t bytes) noexcept override {
    raw_deallocate_calls++;
    live_bytes -= bytes;
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc)
    std::free(ptr);
  }

  Result<void> copy_host_to_device(void* dst, const void* src, std::size_t bytes) override {
    std::memcpy(dst, src, bytes);
    return ok();
  }

  Result<void> copy_device_to_host(void* dst, const void* src, std::size_t bytes) override {
    std::memcpy(dst, src, bytes);
    return ok();
  }

  Result<void> copy_device_to_device(void* dst, const void* src, std::size_t bytes) override {
    std::memcpy(dst, src, bytes);
    return ok();
  }

  Result<void> synchronize() override { return ok(); }

  int raw_allocate_calls = 0;
  int raw_deallocate_calls = 0;
  std::size_t live_bytes = 0;
  bool fail_next_allocation = false;
};

TEST(RoundSizeTest, SmallRequestsRoundToAlignment) {
  EXPECT_EQ(CachingAllocator::round_size(1), CachingAllocator::kAlignment);
  EXPECT_EQ(CachingAllocator::round_size(CachingAllocator::kAlignment),
            CachingAllocator::kAlignment);
  EXPECT_EQ(CachingAllocator::round_size(CachingAllocator::kAlignment + 1),
            2 * CachingAllocator::kAlignment);
}

TEST(RoundSizeTest, ZeroGetsARealBlock) {
  // A zero-element tensor still needs a distinct address, so a zero request cannot round to zero.
  EXPECT_EQ(CachingAllocator::round_size(0), CachingAllocator::kAlignment);
}

TEST(RoundSizeTest, LargeRequestsRoundToTheLargeGranularity) {
  EXPECT_EQ(CachingAllocator::round_size(CachingAllocator::kLargeThreshold),
            CachingAllocator::kLargeGranularity);
  EXPECT_EQ(CachingAllocator::round_size(CachingAllocator::kLargeGranularity + 1),
            2 * CachingAllocator::kLargeGranularity);
}

TEST(RoundSizeTest, RoundingIsMonotonic) {
  std::size_t previous = 0;
  for (std::size_t bytes = 0; bytes < 8192; bytes += 97) {
    const std::size_t rounded = CachingAllocator::round_size(bytes);
    EXPECT_GE(rounded, bytes);
    EXPECT_GE(rounded, previous) << "rounding went backwards at " << bytes;
    previous = rounded;
  }
}

TEST(AllocatorTest, ReturnsUsableAlignedMemory) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  auto ptr = allocator.allocate(1024);
  ASSERT_TRUE(ptr.has_value()) << ptr.error().to_string();
  ASSERT_NE(*ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*ptr) % kDeviceAlignment, 0U)
      << "Device::raw_allocate must return kDeviceAlignment-aligned memory";

  std::memset(*ptr, 0xAB, 1024);
  allocator.deallocate(*ptr);
}

TEST(AllocatorTest, ReusesAFreedBlockInsteadOfCallingTheDriver) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  auto first = allocator.allocate(4096);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(device.raw_allocate_calls, 1);

  allocator.deallocate(*first);

  auto second = allocator.allocate(4096);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(device.raw_allocate_calls, 1) << "a cached block should not hit the driver";
  EXPECT_EQ(*second, *first) << "the cached block should be the one handed back";

  EXPECT_EQ(allocator.stats().cache_hits, 1U);
  EXPECT_EQ(allocator.stats().cache_misses, 1U);
  allocator.deallocate(*second);
}

TEST(AllocatorTest, RequestsInTheSameSizeClassShareABlock) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  auto first = allocator.allocate(100);
  ASSERT_TRUE(first.has_value());
  allocator.deallocate(*first);

  // 200 rounds to the same class as 100, so this must reuse rather than allocate.
  auto second = allocator.allocate(200);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(device.raw_allocate_calls, 1);
  allocator.deallocate(*second);
}

TEST(AllocatorTest, DistinctLiveAllocationsDoNotOverlap) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  std::vector<void*> pointers;
  std::set<void*> unique;
  for (int i = 0; i < 16; ++i) {
    auto ptr = allocator.allocate(1024);
    ASSERT_TRUE(ptr.has_value());
    pointers.push_back(*ptr);
    unique.insert(*ptr);
  }
  EXPECT_EQ(unique.size(), pointers.size()) << "two live allocations returned the same address";

  for (void* ptr : pointers) {
    allocator.deallocate(ptr);
  }
}

TEST(AllocatorTest, StatsTrackAllocatedAndReserved) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  const std::size_t block = CachingAllocator::round_size(4096);

  auto ptr = allocator.allocate(4096);
  ASSERT_TRUE(ptr.has_value());

  AllocatorStats stats = allocator.stats();
  EXPECT_EQ(stats.allocated_bytes, block);
  EXPECT_EQ(stats.reserved_bytes, block);
  EXPECT_EQ(stats.allocation_count, 1U);

  allocator.deallocate(*ptr);

  stats = allocator.stats();
  EXPECT_EQ(stats.allocated_bytes, 0U) << "freed memory is no longer allocated";
  EXPECT_EQ(stats.reserved_bytes, block) << "but is still reserved, because it is cached";
  EXPECT_EQ(stats.peak_allocated_bytes, block);
  EXPECT_EQ(stats.deallocation_count, 1U);
}

TEST(AllocatorTest, EmptyCacheReturnsMemoryToTheDevice) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  auto ptr = allocator.allocate(4096);
  ASSERT_TRUE(ptr.has_value());
  allocator.deallocate(*ptr);

  const std::size_t released = allocator.empty_cache();
  EXPECT_EQ(released, CachingAllocator::round_size(4096));
  EXPECT_EQ(device.raw_deallocate_calls, 1);
  EXPECT_EQ(allocator.stats().reserved_bytes, 0U);
  EXPECT_EQ(device.live_bytes, 0U);
}

TEST(AllocatorTest, EmptyCacheLeavesLiveBlocksAlone) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  auto live = allocator.allocate(4096);
  ASSERT_TRUE(live.has_value());

  auto cached = allocator.allocate(8192);
  ASSERT_TRUE(cached.has_value());
  allocator.deallocate(*cached);

  EXPECT_EQ(allocator.empty_cache(), CachingAllocator::round_size(8192));

  // The live block must survive and stay usable.
  std::memset(*live, 0x5A, 4096);
  EXPECT_EQ(allocator.stats().allocated_bytes, CachingAllocator::round_size(4096));
  allocator.deallocate(*live);
}

TEST(AllocatorTest, FreesTheCacheAndRetriesWhenTheDeviceIsFull) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  auto cached = allocator.allocate(4096);
  ASSERT_TRUE(cached.has_value());
  allocator.deallocate(*cached);

  // A request in a different size class cannot reuse the cached block, so it reaches the driver.
  device.fail_next_allocation = true;
  auto ptr = allocator.allocate(std::size_t{64} * 1024);
  ASSERT_TRUE(ptr.has_value()) << "should have dropped the cache and retried: "
                               << ptr.error().to_string();
  EXPECT_EQ(allocator.stats().reserved_bytes, CachingAllocator::round_size(std::size_t{64} * 1024));
  allocator.deallocate(*ptr);
}

TEST(AllocatorTest, ReportsFailureWhenThereIsNothingLeftToRelease) {
  FakeDevice device;
  CachingAllocator allocator(&device);

  device.fail_next_allocation = true;
  auto ptr = allocator.allocate(4096);

  ASSERT_FALSE(ptr.has_value());
  EXPECT_EQ(ptr.error().code(), ErrorCode::kResourceExhausted);
  EXPECT_NE(ptr.error().message().find("4096"), std::string::npos)
      << "the message should name the size that failed: " << ptr.error().message();
}

TEST(AllocatorTest, DeallocateIgnoresNull) {
  FakeDevice device;
  CachingAllocator allocator(&device);
  allocator.deallocate(nullptr);
  EXPECT_EQ(allocator.stats().deallocation_count, 0U);
}

} // namespace
} // namespace tessera
