// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tessiq/runtime/device.h"
#include "tessiq/support/result.h"

namespace tessiq {

/// Counters for the Python memory-stats API and for leak assertions in tests.
struct AllocatorStats {
  /// Bytes currently handed out to callers.
  std::size_t allocated_bytes = 0;
  /// Bytes held by this allocator, in use or cached. Never shrinks except via empty_cache().
  std::size_t reserved_bytes = 0;
  std::size_t peak_allocated_bytes = 0;
  std::size_t peak_reserved_bytes = 0;
  std::uint64_t allocation_count = 0;
  std::uint64_t deallocation_count = 0;
  /// Requests served from a cached block.
  std::uint64_t cache_hits = 0;
  /// Requests that had to call Device::raw_allocate.
  std::uint64_t cache_misses = 0;
};

/// Sub-allocates device memory, keeping freed blocks for reuse.
///
/// Driver allocation is slow, so freed blocks are cached rather than returned. Sizes are rounded to
/// a size class so that a block freed by one tensor can serve the next of similar size.
///
/// Thread-safe.
class CachingAllocator {
 public:
  /// Size class granularity for small requests. Address alignment comes from the Device.
  static constexpr std::size_t kAlignment = kDeviceAlignment;

  /// Requests at or above this are rounded to kLargeGranularity instead of kAlignment, so large
  /// blocks are reusable across slightly different sizes.
  static constexpr std::size_t kLargeThreshold = 1U << 20U;

  static constexpr std::size_t kLargeGranularity = 2U << 20U;

  explicit CachingAllocator(Device* device);

  ~CachingAllocator();

  CachingAllocator(const CachingAllocator&) = delete;
  CachingAllocator& operator=(const CachingAllocator&) = delete;
  CachingAllocator(CachingAllocator&&) = delete;
  CachingAllocator& operator=(CachingAllocator&&) = delete;

  /// Returns a pointer to at least `bytes` of device memory, or kResourceExhausted.
  ///
  /// A zero-byte request returns a distinct non-null pointer.
  Result<void*> allocate(std::size_t bytes);

  /// Returns a pointer from allocate() to the cache. Passing null is a no-op; passing anything
  /// else this allocator did not hand out is a programmer error.
  void deallocate(void* ptr);

  /// Releases cached blocks back to the device. Blocks still in use are untouched.
  ///
  /// Returns the number of bytes released.
  std::size_t empty_cache();

  [[nodiscard]] AllocatorStats stats() const;

  /// Rounds `bytes` up to its size class. Public for tests.
  [[nodiscard]] static std::size_t round_size(std::size_t bytes);

 private:
  struct Block {
    void* ptr = nullptr;
    std::size_t size = 0;
  };

  Device* device_;

  mutable std::mutex mutex_;

  /// Cached blocks, keyed by rounded size. Every block in here is owned by this allocator and
  /// not handed out.
  std::map<std::size_t, std::vector<void*>> free_blocks_;

  /// Every block this allocator has taken from the device, in use or cached. Also the lookup
  /// deallocate() uses to recover a pointer's size.
  std::unordered_map<void*, Block> live_blocks_;

  AllocatorStats stats_;
};

/// The allocator for `index`. One per device.
CachingAllocator& allocator_at(int index);

/// The allocator for the index this thread is on.
CachingAllocator& current_allocator();

} // namespace tessiq
