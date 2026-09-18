// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The device table and the per-thread current index.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/cuda_device.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq {

namespace {

/// Owns every device and its allocator, built once on first use.
///
/// Index 0 is the CPU reference and always exists. CUDA devices follow in driver order, so a
/// machine with two GPUs exposes tessiq:0, tessiq:1 and tessiq:2.
class Registry {
 public:
  static Registry& instance() {
    static Registry registry;
    return registry;
  }

  [[nodiscard]] int size() const { return static_cast<int>(devices_.size()); }

  [[nodiscard]] Device* at(int index) const {
    if (index < 0 || index >= size()) {
      return nullptr;
    }
    return devices_.at(static_cast<std::size_t>(index)).get();
  }

  [[nodiscard]] CachingAllocator& allocator(int index) const {
    // Held in a named variable because the check macro negates its argument, and clang-tidy then
    // reads the negation of a conjunction as a DeMorgan smell in code nobody wrote that way.
    const bool in_range = index >= 0 && index < size();
    TESSIQ_CHECK(in_range, "allocator requested for device index " + std::to_string(index));
    return *allocators_.at(static_cast<std::size_t>(index));
  }

 private:
  Registry() {
    devices_.push_back(make_cpu_device());
    for (auto& device : discover_cuda_devices()) {
      devices_.push_back(std::move(device));
    }

    // One allocator per device. A single pooled allocator would report a memory total that
    // belongs to no particular device, which is worse than no number at all.
    allocators_.reserve(devices_.size());
    for (const auto& device : devices_) {
      allocators_.push_back(std::make_unique<CachingAllocator>(device.get()));
    }

    TESSIQ_LOG(kInfo, "runtime") << "registered " << devices_.size() << " device(s)";
  }

  std::vector<std::unique_ptr<Device>> devices_;
  std::vector<std::unique_ptr<CachingAllocator>> allocators_;
};

/// The index this thread allocates on.
///
/// Thread-local because PyTorch's device guard is: two threads can be mid-op on two devices, and
/// c10's allocator interface takes no device argument, so this is how an allocation knows where
/// it belongs.
int& thread_device_index() {
  static thread_local int index = 0;
  return index;
}

} // namespace

int device_count() { return Registry::instance().size(); }

Result<Device*> device_at(int index) {
  Device* device = Registry::instance().at(index);
  if (device == nullptr) {
    return invalid_argument("no tessiq device at index " + std::to_string(index) +
                            "; this machine has " + std::to_string(device_count()) +
                            " (index 0 is the cpu reference)");
  }
  return device;
}

int current_device_index() { return thread_device_index(); }

namespace detail {

void set_current_device_index_unchecked(int index) noexcept { thread_device_index() = index; }

} // namespace detail

Result<void> set_current_device_index(int index) {
  TESSIQ_RETURN_IF_ERROR(device_at(index));
  thread_device_index() = index;
  return ok();
}

Device& current_device() {
  auto device = device_at(current_device_index());
  TESSIQ_CHECK(device.has_value(), "the current device index names no device");
  return **device;
}

CachingAllocator& allocator_at(int index) { return Registry::instance().allocator(index); }

CachingAllocator& current_allocator() { return allocator_at(current_device_index()); }

} // namespace tessiq
