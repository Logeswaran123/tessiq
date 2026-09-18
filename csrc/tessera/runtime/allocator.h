// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tessera/runtime/device.h"
#include "tessera/support/result.h"

namespace tessera {

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
/// Driver allocation is slow enough that a training step which allocates per iteration will spend
/// most of its time in the driver. Freed blocks are therefore cached rather than returned.
///
/// Sizes are rounded to a size class so that a block freed by one tensor can serve the next tensor
/// of similar size. Without rounding the free list fills with blocks of odd sizes that never match
/// a later request.
///
/// Thread-safe.
class CachingAllocator {
 public:
  /// Size class granularity for small requests. Address alignment is the Device's contract
  /// (kDeviceAlignment), not something rounding the size can provide.
  static constexpr std::size_t kAlignment = kDeviceAlignment;

  /// Requests at or above this are rounded to kLargeGranularity instead of kAlignment. Rounding
  /// a 100 MB tensor to 512 bytes leaves a free list that never matches; rounding to 2 MB does.
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
  /// A zero-byte request returns a distinct non-null pointer, matching the convention that a
  /// zero-element tensor still has an identity.
  Result<void*> allocate(std::size_t bytes);

  /// Returns a pointer from allocate() to the cache. Passing null is a no-op; passing anything
  /// else this allocator did not hand out is a programmer error.
  void deallocate(void* ptr);

  /// Releases cached blocks back to the device. Blocks still in use are untouched.
  ///
  /// Returns the number of bytes released. Used to hand memory back to another process, and by
  /// tests to assert the cache is doing what it claims.
  std::size_t empty_cache();

  [[nodiscard]] AllocatorStats stats() const;

  /// Rounds `bytes` up to its size class. Exposed for tests, which assert the policy directly
  /// rather than inferring it from allocation behaviour.
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

} // namespace tessera
