// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/runtime/allocator.h"

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>

#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq {

namespace {

constexpr std::string_view kLogModule = "allocator";

/// Rounds `bytes` up to a multiple of `granularity`, which must be a power of two.
std::size_t round_up(std::size_t bytes, std::size_t granularity) {
  return (bytes + granularity - 1) & ~(granularity - 1);
}

} // namespace

std::size_t CachingAllocator::round_size(std::size_t bytes) {
  if (bytes == 0) {
    return kAlignment;
  }
  if (bytes < kLargeThreshold) {
    return round_up(bytes, kAlignment);
  }
  return round_up(bytes, kLargeGranularity);
}

CachingAllocator::CachingAllocator(Device* device) : device_(device) {
  TESSIQ_CHECK(device != nullptr, "CachingAllocator requires a device");
}

CachingAllocator::~CachingAllocator() {
  // Blocks still in use at teardown are usually a leak, but at interpreter shutdown this allocator
  // can be destroyed before the tensors holding its memory, so this is logged at debug.
  if (stats_.allocated_bytes != 0) {
    TESSIQ_LOG(kDebug, kLogModule)
        << "destroyed with " << stats_.allocated_bytes << " bytes still allocated across "
        << stats_.live_blocks << " blocks";
  }

  for (const auto& [ptr, block] : live_blocks_) {
    device_->raw_deallocate(block.ptr, block.size);
  }
}

Result<void*> CachingAllocator::allocate(std::size_t bytes) {
  const std::size_t rounded = round_size(bytes);

  const std::scoped_lock guard(mutex_);

  if (auto it = free_blocks_.find(rounded); it != free_blocks_.end() && !it->second.empty()) {
    void* ptr = it->second.back();
    it->second.pop_back();

    stats_.cache_hits++;
    handed_out(rounded);

    TESSIQ_LOG(kTrace, kLogModule) << "reused " << rounded << " bytes at " << ptr;
    return ptr;
  }

  auto raw = device_->raw_allocate(rounded);
  if (!raw) {
    // Release the cache and retry once before reporting failure.
    const std::size_t released = release_cached();
    if (released == 0) {
      stats_.out_of_memory++;
      return std::move(raw).error().with_context("allocating " + std::to_string(rounded) +
                                                 " bytes");
    }
    stats_.retries++;

    TESSIQ_LOG(kInfo, kLogModule) << "allocation of " << rounded << " bytes failed; released "
                                  << released << " cached bytes and retrying";

    raw = device_->raw_allocate(rounded);
    if (!raw) {
      stats_.out_of_memory++;
      return std::move(raw).error().with_context("allocating " + std::to_string(rounded) +
                                                 " bytes after releasing " +
                                                 std::to_string(released) + " cached bytes");
    }
  }

  void* ptr = *raw;
  live_blocks_.emplace(ptr, Block{ptr, rounded});

  stats_.cache_misses++;
  stats_.reserved_bytes += rounded;
  stats_.reserved_bytes_total += rounded;
  stats_.reserved_blocks++;
  stats_.peak_reserved_bytes = std::max(stats_.peak_reserved_bytes, stats_.reserved_bytes);
  stats_.peak_reserved_blocks = std::max(stats_.peak_reserved_blocks, stats_.reserved_blocks);
  handed_out(rounded);

  TESSIQ_LOG(kTrace, kLogModule) << "allocated " << rounded << " bytes at " << ptr;
  return ptr;
}

void CachingAllocator::deallocate(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  const std::scoped_lock guard(mutex_);

  const auto it = live_blocks_.find(ptr);
  TESSIQ_CHECK(it != live_blocks_.end(),
               "deallocate() called on a pointer this allocator did not return");

  const std::size_t size = it->second.size;
  free_blocks_[size].push_back(ptr);

  stats_.deallocation_count++;
  stats_.allocated_bytes -= size;
  stats_.freed_bytes_total += size;
  stats_.live_blocks--;

  TESSIQ_LOG(kTrace, kLogModule) << "cached " << size << " bytes at " << ptr;
}

std::size_t CachingAllocator::empty_cache() {
  const std::scoped_lock guard(mutex_);
  const std::size_t released = release_cached();
  TESSIQ_LOG(kDebug, kLogModule) << "released " << released << " cached bytes";
  return released;
}

std::size_t CachingAllocator::release_cached() {
  std::size_t released = 0;
  for (auto& [size, blocks] : free_blocks_) {
    for (void* ptr : blocks) {
      device_->raw_deallocate(ptr, size);
      live_blocks_.erase(ptr);
      released += size;
      stats_.device_frees++;
      stats_.reserved_blocks--;
    }
    blocks.clear();
  }
  free_blocks_.clear();

  stats_.reserved_bytes -= released;
  stats_.released_bytes_total += released;
  return released;
}

void CachingAllocator::handed_out(std::size_t bytes) {
  stats_.allocation_count++;
  stats_.allocated_bytes += bytes;
  stats_.allocated_bytes_total += bytes;
  stats_.live_blocks++;
  stats_.peak_allocated_bytes = std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);
  stats_.peak_live_blocks = std::max(stats_.peak_live_blocks, stats_.live_blocks);
}

AllocatorStats CachingAllocator::stats() const {
  const std::scoped_lock guard(mutex_);
  return stats_;
}

void CachingAllocator::reset_peak_stats() {
  const std::scoped_lock guard(mutex_);
  stats_.peak_allocated_bytes = stats_.allocated_bytes;
  stats_.peak_reserved_bytes = stats_.reserved_bytes;
  stats_.peak_live_blocks = stats_.live_blocks;
  stats_.peak_reserved_blocks = stats_.reserved_blocks;
}

void CachingAllocator::reset_accumulated_stats() {
  const std::scoped_lock guard(mutex_);
  stats_.allocation_count = 0;
  stats_.deallocation_count = 0;
  stats_.cache_hits = 0;
  stats_.cache_misses = 0;
  stats_.device_frees = 0;
  stats_.retries = 0;
  stats_.out_of_memory = 0;
  stats_.allocated_bytes_total = 0;
  stats_.freed_bytes_total = 0;
  stats_.reserved_bytes_total = 0;
  stats_.released_bytes_total = 0;
}

} // namespace tessiq
