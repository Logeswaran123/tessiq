// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Everything PyTorch needs in order to treat tessiq as a device.
//
// PyTorch reserves one dispatch key for out-of-tree backends. Its internal identifier is fixed in
// PyTorch's own enum and is confined to this file; the name users see is set
// by rename_privateuse1_backend("tessiq") on the Python side.
//
// The file most exposed to changes in torch's internal interfaces.

#include "tessiq/torch/backend.h"

#include <algorithm>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Storage.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cstddef>
#include <mutex>
#include <unordered_map>

#include <torch/library.h>

#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "backend";

constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;

/// Adapts CachingAllocator to the interface c10 expects.
///
/// c10 reports allocation failure by throwing, so this is one of the few places outside the
/// bindings where an Error becomes an exception.
struct TessiqAllocator final : c10::Allocator {
  c10::DataPtr allocate(std::size_t bytes) override {
    // c10's interface takes no device. The index comes from the device guard torch sets around
    // every op, as it does for CUDA's allocator.
    const int index = current_device_index();
    auto ptr = current_allocator().allocate(bytes);
    if (!ptr) {
      TORCH_CHECK(false, ptr.error().to_string());
    }
    // The deleter gets only the pointer, so record() maps it back to its device.
    record(*ptr, index);
    return {*ptr, *ptr, &release, c10::Device(kDeviceType, static_cast<c10::DeviceIndex>(index))};
  }

  [[nodiscard]] c10::DeleterFnPtr raw_deleter() const override { return &release; }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    const auto result = current_device().copy_device_to_device(dest, src, count);
    TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
  }

  static void release(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
    allocator_at(forget(ptr)).deallocate(ptr);
  }

 private:
  /// Which allocator handed out each live pointer.
  ///
  /// c10 frees through a function pointer that takes only the address, and freeing on the wrong
  /// device's allocator would silently corrupt both pools.
  static std::unordered_map<void*, int>& owners() {
    static std::unordered_map<void*, int> map;
    return map;
  }

  static std::mutex& owners_mutex() {
    static std::mutex mutex;
    return mutex;
  }

  static void record(void* ptr, int index) {
    const std::scoped_lock guard(owners_mutex());
    owners()[ptr] = index;
  }

  static int forget(void* ptr) {
    const std::scoped_lock guard(owners_mutex());
    const auto it = owners().find(ptr);
    TESSIQ_CHECK(it != owners().end(), "freeing a pointer tessiq did not allocate");
    const int index = it->second;
    owners().erase(it);
    return index;
  }
};

// REGISTER_ALLOCATOR takes a pointer to a mutable allocator, and c10 owns the lifetime from there.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TessiqAllocator g_allocator;

/// Device and stream bookkeeping for the dispatcher.
///
/// Streams are not exposed: every launch and copy goes on the legacy default stream, so torch
/// sees only the default stream.
///
/// Every signature below is fixed by c10, down to the spelling of `static_type`. The suppressions
/// cover the naming rules this interface cannot follow.
// NOLINTBEGIN(readability-identifier-naming, readability-named-parameter,
//             modernize-use-nodiscard, clang-diagnostic-unused-const-variable)
struct TessiqGuardImpl final : c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = kDeviceType;

  c10::DeviceType type() const override { return kDeviceType; }

  c10::Device exchangeDevice(c10::Device device) const override {
    const c10::Device previous = getDevice();
    setDevice(device);
    return previous;
  }

  c10::Device getDevice() const override {
    return {kDeviceType, static_cast<c10::DeviceIndex>(current_device_index())};
  }

  void setDevice(c10::Device device) const override {
    TORCH_CHECK(device.type() == kDeviceType, "expected a tessiq device, got ", device);
    if (device.index() < 0) {
      return; // -1 means the current device.
    }
    const auto result = set_current_device_index(static_cast<int>(device.index()));
    TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    if (device.index() >= 0) {
      static_cast<void>(set_current_device_index(static_cast<int>(device.index())));
    }
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(kDeviceType, 0));
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return static_cast<c10::DeviceIndex>(device_count());
  }
};

// NOLINTEND(readability-identifier-naming, readability-named-parameter,
//           modernize-use-nodiscard, clang-diagnostic-unused-const-variable)

C10_REGISTER_GUARD_IMPL(PrivateUse1, TessiqGuardImpl);

} // namespace

// REGISTER_ALLOCATOR opens its own anonymous namespace, so it has to sit at namespace scope.
REGISTER_ALLOCATOR(kDeviceType, &g_allocator)

c10::Allocator* get_allocator() { return &g_allocator; }

AllocatorStats allocator_stats(int index) { return allocator_at(index).stats(); }

std::size_t empty_cache(int index) { return allocator_at(index).empty_cache(); }

void reallocate_storage(const c10::Storage& storage, std::size_t bytes) {
  // Allocated under the storage's own index. An op with no device guard, such as resize_, runs
  // with whatever index the thread last set, which may be another device.
  const DeviceIndexGuard guard(storage.device().index());
  at::DataPtr data = g_allocator.allocate(bytes);
  const std::size_t kept = std::min(bytes, storage.nbytes());
  if (kept > 0 && storage.data() != nullptr) {
    const auto copied = current_device().copy_device_to_device(data.get(), storage.data(), kept);
    TORCH_CHECK(copied.has_value(), copied.has_value() ? "" : copied.error().to_string());
  }
  storage.set_data_ptr_noswap(std::move(data));
  storage.set_nbytes(bytes);
}

void register_allocator() {
  // Registration happens at static initialisation via REGISTER_ALLOCATOR above. Calling this
  // from the Python module keeps this translation unit linked in.
  TESSIQ_LOG(kInfo, kLogModule) << "tessiq backend ready with " << device_count() << " device(s)";
}

} // namespace tessiq::torch_backend
