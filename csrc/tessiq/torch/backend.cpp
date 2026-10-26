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

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <algorithm>
#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Storage.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/library.h>

#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/cuda_device.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "backend";

constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;

/// The allocator for `index` if there is a device there, as torch's calls need.
CachingAllocator& checked_allocator(c10::DeviceIndex index) {
  const auto device = device_at(index);
  TORCH_CHECK(device.has_value(), device.has_value() ? "" : device.error().to_string());
  return allocator_at(index);
}

/// tessiq's counters in torch's layout.
///
/// There is one pool and a block is never split, so only the "all" figures are filled, and the
/// split-block counts and requested bytes stay zero.
c10::CachingDeviceAllocator::DeviceStats device_stats(const AllocatorStats& from) {
  using c10::CachingAllocator::Stat;
  constexpr auto kAll = static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  const auto set = [](Stat& stat,
                      std::uint64_t current,
                      std::uint64_t peak,
                      std::uint64_t increased,
                      std::uint64_t decreased) {
    stat.current = static_cast<std::int64_t>(current);
    stat.peak = static_cast<std::int64_t>(peak);
    stat.allocated = static_cast<std::int64_t>(increased);
    stat.freed = static_cast<std::int64_t>(decreased);
  };

  c10::CachingDeviceAllocator::DeviceStats stats;
  set(stats.allocation[kAll],
      from.live_blocks,
      from.peak_live_blocks,
      from.allocation_count,
      from.deallocation_count);
  set(stats.allocated_bytes[kAll],
      from.allocated_bytes,
      from.peak_allocated_bytes,
      from.allocated_bytes_total,
      from.freed_bytes_total);
  set(stats.segment[kAll],
      from.reserved_blocks,
      from.peak_reserved_blocks,
      from.cache_misses,
      from.device_frees);
  set(stats.reserved_bytes[kAll],
      from.reserved_bytes,
      from.peak_reserved_bytes,
      from.reserved_bytes_total,
      from.released_bytes_total);
  // A block is active exactly while it is handed out.
  stats.active = stats.allocation;
  stats.active_bytes = stats.allocated_bytes;
  stats.num_device_alloc = static_cast<std::int64_t>(from.cache_misses);
  stats.num_device_free = static_cast<std::int64_t>(from.device_frees);
  stats.num_alloc_retries = static_cast<std::int64_t>(from.retries);
  stats.num_ooms = static_cast<std::int64_t>(from.out_of_memory);
  return stats;
}

/// Adapts CachingAllocator to the interface c10 expects.
///
/// c10's interface takes no device. The instance registered with c10 allocates on the current
/// device, which torch's device guard sets around an op, as for CUDA's allocator. The instances
/// from get_allocator(index) allocate on their own device, so a storage made with one clones onto
/// that device whatever the current one is.
///
/// It is a DeviceAllocator so that torch.accelerator's memory functions reach tessiq's counters.
///
/// c10 reports allocation failure by throwing, so this is one of the few places outside the
/// bindings where an Error becomes an exception.
struct TessiqAllocator final : c10::DeviceAllocator {
  explicit TessiqAllocator(std::optional<int> index = std::nullopt) noexcept : index_(index) {}

  c10::DataPtr allocate(std::size_t bytes) override {
    const int index = index_.value_or(current_device_index());
    auto ptr = allocator_at(index).allocate(bytes);
    if (!ptr) {
      TORCH_CHECK_WITH(OutOfMemoryError,
                       ptr.error().code() != ErrorCode::kResourceExhausted,
                       ptr.error().to_string());
      TORCH_CHECK(false, ptr.error().to_string());
    }
    // The deleter gets only the pointer, so record() maps it back to its device.
    record(*ptr, index);
    return {*ptr, *ptr, &release, c10::Device(kDeviceType, static_cast<c10::DeviceIndex>(index))};
  }

  [[nodiscard]] c10::DeleterFnPtr raw_deleter() const override { return &release; }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    // c10 calls this from clone(), which allocates `dest` on the current device. Copy on the
    // device that owns both, since the current one may be another.
    const std::optional<int> to = owner_of(dest);
    const std::optional<int> from = owner_of(src);
    TORCH_CHECK(to.has_value() && from.has_value(), "tessiq did not allocate ", dest, " and ", src);
    TORCH_CHECK(
        *to == *from, "tessiq cannot clone an allocation on device ", *from, " onto device ", *to);
    const auto device = device_at(*to);
    TORCH_CHECK(device.has_value(), device.has_value() ? "" : device.error().to_string());
    const auto result = (*device)->copy_device_to_device(dest, src, count);
    TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
  }

  static void release(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
    allocator_at(forget(ptr)).deallocate(ptr);
  }

  bool initialized() override { return true; }

  void emptyCache(c10::MempoolId_t /*mempool_id*/) override {
    for (int index = 0; index < device_count(); ++index) {
      static_cast<void>(allocator_at(index).empty_cache());
    }
  }

  // Nothing to record. Launches go on the legacy default stream and copies on the device's own
  // blocking stream, which the legacy stream orders against, so a block freed after its last use
  // is not handed out before that use finishes.
  void recordStream(const c10::DataPtr& /*ptr*/, c10::Stream /*stream*/) override {}

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    return device_stats(checked_allocator(device).stats());
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    checked_allocator(device).reset_accumulated_stats();
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    checked_allocator(device).reset_peak_stats();
  }

  std::pair<std::size_t, std::size_t> getMemoryInfo(c10::DeviceIndex device) override {
    const auto found = device_at(device);
    TORCH_CHECK(found.has_value(), found.has_value() ? "" : found.error().to_string());
    const auto info = (*found)->memory_info();
    TORCH_CHECK(info.has_value(), info.has_value() ? "" : info.error().to_string());
    return {info->free_bytes, info->total_bytes};
  }

  /// The device index that allocated `ptr`, or nothing if tessiq did not. `ptr` has to be the
  /// start of the allocation.
  static std::optional<int> owner_of(const void* ptr) {
    const std::scoped_lock guard(owners_mutex());
    const auto it = owners().find(ptr);
    return it == owners().end() ? std::nullopt : std::optional<int>(it->second);
  }

 private:
  std::optional<int> index_;

  /// Which allocator handed out each live pointer.
  ///
  /// c10 frees through a function pointer that takes only the address, and freeing on the wrong
  /// device's allocator would silently corrupt both pools.
  static std::unordered_map<const void*, int>& owners() {
    static std::unordered_map<const void*, int> map;
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
// c10 defines DeviceAllocator's constructor out of line without noexcept, which is all the second
// check sees.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
// NOLINTBEGIN(bugprone-throwing-static-initialization)
TessiqAllocator g_allocator;
// NOLINTEND(bugprone-throwing-static-initialization)
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// Host memory for `pin_memory()` and a DataLoader with `pin_memory=True`.
///
/// Page-locked when there is a GPU, so a copy from it is a direct DMA and one made with
/// `non_blocking=True` returns before it finishes. With no GPU every tessiq device is host memory,
/// so ordinary memory is as good, and pinning works rather than failing.
struct PinnedAllocator final : c10::Allocator {
  c10::DataPtr allocate(std::size_t bytes) override {
    if (bytes == 0) {
      return {nullptr, nullptr, &release, c10::Device(c10::DeviceType::CPU)};
    }
    void* ptr = nullptr;
    if (page_locked()) {
      auto locked = allocate_page_locked(bytes);
      TORCH_CHECK(locked.has_value(), locked.has_value() ? "" : locked.error().to_string());
      ptr = *locked;
    } else {
      ptr = c10::alloc_cpu(bytes);
    }
    {
      const std::scoped_lock guard(mutex());
      pinned().insert(ptr);
    }
    return {ptr, ptr, &release, c10::Device(c10::DeviceType::CPU)};
  }

  [[nodiscard]] c10::DeleterFnPtr raw_deleter() const override { return &release; }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    default_copy_data(dest, src, count);
  }

  /// Whether `ptr` is the start of a block this allocator handed out.
  static bool owns(const void* ptr) {
    const std::scoped_lock guard(mutex());
    return pinned().contains(ptr);
  }

  static void release(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
    {
      const std::scoped_lock guard(mutex());
      pinned().erase(ptr);
    }
    if (!page_locked()) {
      c10::free_cpu(ptr);
      return;
    }
    // A copy made with non_blocking=True may still be reading or writing it.
    for (int index = 1; index < device_count(); ++index) {
      if (const auto device = device_at(index); device.has_value()) {
        static_cast<void>((*device)->synchronize());
      }
    }
    free_page_locked(ptr);
  }

 private:
  /// Whether this process pins by page-locking, which it does when there is a GPU.
  static bool page_locked() {
    static const bool locked = device_count() > 1;
    return locked;
  }

  static std::unordered_set<const void*>& pinned() {
    static std::unordered_set<const void*> set;
    return set;
  }

  static std::mutex& mutex() {
    static std::mutex mutex;
    return mutex;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables): c10 takes it as mutable.
PinnedAllocator g_pinned_allocator;

/// Device and stream bookkeeping for the dispatcher.
///
/// Streams are not exposed. Launches go on the legacy default stream and copies on each device's
/// own stream, which the legacy stream orders against, so torch sees only a default stream.
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

  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    const auto device = device_at(device_index);
    TORCH_CHECK(device.has_value(), device.has_value() ? "" : device.error().to_string());
    const auto done = (*device)->synchronize();
    TORCH_CHECK(done.has_value(), done.has_value() ? "" : done.error().to_string());
  }
};

// NOLINTEND(readability-identifier-naming, readability-named-parameter,
//           modernize-use-nodiscard, clang-diagnostic-unused-const-variable)

C10_REGISTER_GUARD_IMPL(PrivateUse1, TessiqGuardImpl);

/// What torch asks of an out-of-tree backend besides running its operations: pinned host memory,
/// generators, resizing a storage, and which device a pointer is on.
///
/// Generators are CPU generators. A random operation on a tessiq tensor runs on the host through
/// the fallback, and a CPU generator is the one the host kernel accepts.
// NOLINTBEGIN(readability-identifier-naming, readability-named-parameter, modernize-use-nodiscard)
struct TessiqHooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }

  bool isAvailable() const override { return device_count() > 0; }

  // A device counts as set up once it exists. tessiq does not track whether a GPU's context has
  // been created yet.
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return device_at(device_index).has_value();
  }

  void init() const override {}

  c10::DeviceIndex deviceCount() const override {
    return static_cast<c10::DeviceIndex>(device_count());
  }

  void setCurrentDevice(c10::DeviceIndex device) const override {
    const auto result = set_current_device_index(device);
    TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
  }

  c10::DeviceIndex getCurrentDevice() const override {
    return static_cast<c10::DeviceIndex>(current_device_index());
  }

  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
    const c10::DeviceIndex previous = getCurrentDevice();
    setCurrentDevice(device);
    return previous;
  }

  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
    return exchangeDevice(device);
  }

  // Page-locked memory from anywhere counts, since torch pins with CUDA's allocator rather than
  // this one whenever CUDA is present, as for `torch.empty(n, pin_memory=True)`.
  bool isPinnedPtr(const void* data) const override {
    return PinnedAllocator::owns(data) || (device_count() > 1 && is_page_locked(data));
  }

  c10::Allocator* getPinnedMemoryAllocator() const override { return &g_pinned_allocator; }

  const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override {
    return at::detail::getDefaultCPUGenerator();
  }

  at::Generator getNewGenerator(c10::DeviceIndex) const override {
    return at::detail::createCPUGenerator();
  }

  at::Device getDeviceFromPtr(void* data) const override {
    const std::optional<int> index = TessiqAllocator::owner_of(data);
    TORCH_CHECK(index.has_value(), "tessiq did not allocate ", data);
    return {kDeviceType, static_cast<c10::DeviceIndex>(*index)};
  }

  void resizePrivateUse1Bytes(const c10::Storage& storage, std::size_t newsize) const override {
    TORCH_CHECK(storage.resizable(), "tessiq cannot resize a storage that is not resizable");
    reallocate_storage(storage, newsize);
  }
};

// NOLINTEND(readability-identifier-naming, readability-named-parameter, modernize-use-nodiscard)

} // namespace

// REGISTER_ALLOCATOR opens its own anonymous namespace, so it has to sit at namespace scope.
REGISTER_ALLOCATOR(kDeviceType, &g_allocator)

c10::Allocator* get_allocator(int index) {
  static const std::vector<std::unique_ptr<TessiqAllocator>> per_device = [] {
    std::vector<std::unique_ptr<TessiqAllocator>> made;
    made.reserve(static_cast<std::size_t>(device_count()));
    for (int each = 0; each < device_count(); ++each) {
      made.push_back(std::make_unique<TessiqAllocator>(each));
    }
    return made;
  }();
  TORCH_CHECK(
      index >= 0 && std::cmp_less(index, per_device.size()), "no tessiq device at index ", index);
  return per_device[static_cast<std::size_t>(index)].get();
}

AllocatorStats allocator_stats(int index) { return allocator_at(index).stats(); }

std::optional<int> cuda_ordinal_of(const c10::Device& device) {
  if (device.type() != c10::DeviceType::PrivateUse1) {
    return std::nullopt;
  }
  const Result<Device*> found = device_at(device.index());
  if (!found.has_value() || (*found)->kind() != DeviceKind::kCuda) {
    return std::nullopt;
  }
  return (*found)->ordinal();
}

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

void register_backend() {
  // The allocator and the guard register at static initialisation, above. The hooks register
  // here, once.
  static const bool hooks_registered = [] {
    static TessiqHooks hooks;
    at::RegisterPrivateUse1HooksInterface(&hooks);
    return true;
  }();
  static_cast<void>(hooks_registered);
  TESSIQ_LOG(kInfo, kLogModule) << "tessiq backend ready with " << device_count() << " device(s)";
}

} // namespace tessiq::torch_backend
