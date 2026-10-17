// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/runtime/cuda_device.h"

#ifdef TESSIQ_WITH_CUDA

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq {

namespace {

constexpr std::string_view kLogModule = "cuda";

Error cuda_error(cudaError_t status, std::string_view what) {
  return external_error(std::string(what) + ": " + cudaGetErrorName(status) + " (" +
                        cudaGetErrorString(status) + ")");
}

/// Wraps the driver, one instance per physical device.
///
/// Every entry point sets the CUDA device first. The runtime's current device is per thread and
/// separate from tessiq's current index, so nothing here relies on whichever thread called last.
class CudaDevice final : public Device {
 public:
  explicit CudaDevice(int ordinal) : ordinal_(ordinal) {}

  TESSIQ_DISALLOW_COPY_AND_MOVE(CudaDevice);

  ~CudaDevice() override {
    if (stream_ != nullptr) {
      static_cast<void>(cudaSetDevice(ordinal_));
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  [[nodiscard]] DeviceKind kind() const override { return DeviceKind::kCuda; }

  [[nodiscard]] int ordinal() const override { return ordinal_; }

  Result<void*> raw_allocate(std::size_t bytes) override {
    TESSIQ_RETURN_IF_ERROR(activate());
    void* ptr = nullptr;
    // cudaMalloc documents 256-byte alignment and in practice returns page-aligned memory, which
    // meets kDeviceAlignment.
    if (const cudaError_t status = cudaMalloc(&ptr, bytes); status != cudaSuccess) {
      if (status == cudaErrorMemoryAllocation) {
        return resource_exhausted("cuda device " + std::to_string(ordinal_) +
                                  " could not allocate " + std::to_string(bytes) + " bytes");
      }
      return cuda_error(status, "cudaMalloc");
    }
    return ptr;
  }

  void raw_deallocate(void* ptr, std::size_t /*bytes*/) noexcept override {
    // Runs from destructors, so errors are ignored: a failure here means the context is already
    // gone.
    static_cast<void>(cudaSetDevice(ordinal_));
    static_cast<void>(cudaFree(ptr));
  }

  Result<void> copy_host_to_device(void* dst, const void* src, std::size_t bytes) override {
    return copy(dst, src, bytes, cudaMemcpyHostToDevice);
  }

  Result<void> copy_device_to_host(void* dst, const void* src, std::size_t bytes) override {
    return copy(dst, src, bytes, cudaMemcpyDeviceToHost);
  }

  Result<void> copy_device_to_device(void* dst, const void* src, std::size_t bytes) override {
    return copy(dst, src, bytes, cudaMemcpyDeviceToDevice);
  }

  Result<void> synchronize() override {
    TESSIQ_RETURN_IF_ERROR(activate());
    // Transfers go on stream_ and every launch on the legacy default stream. Each stream waits for
    // work queued earlier on the other, so waiting for both covers everything, whichever came last.
    if (const cudaError_t status = cudaStreamSynchronize(stream_); status != cudaSuccess) {
      return cuda_error(status, "cudaStreamSynchronize");
    }
    if (const cudaError_t status = cudaStreamSynchronize(cudaStreamLegacy); status != cudaSuccess) {
      return cuda_error(status, "cudaStreamSynchronize(cudaStreamLegacy)");
    }
    return ok();
  }

  Result<MemoryInfo> memory_info() override {
    TESSIQ_RETURN_IF_ERROR(activate());
    MemoryInfo info;
    if (const cudaError_t status = cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
        status != cudaSuccess) {
      return cuda_error(status, "cudaMemGetInfo");
    }
    return info;
  }

 private:
  Result<void> activate() {
    if (const cudaError_t status = cudaSetDevice(ordinal_); status != cudaSuccess) {
      return cuda_error(status, "cudaSetDevice(" + std::to_string(ordinal_) + ")");
    }
    const std::scoped_lock lock(stream_mutex_);
    if (stream_ == nullptr) {
      if (const cudaError_t status = cudaStreamCreate(&stream_); status != cudaSuccess) {
        return cuda_error(status, "cudaStreamCreate");
      }
      TESSIQ_LOG(kDebug, kLogModule) << "created stream for device " << ordinal_;
    }
    return ok();
  }

  Result<void> copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind direction) {
    if (bytes == 0) {
      return ok();
    }
    if (dst == nullptr || src == nullptr) {
      return invalid_argument("copy with a null pointer: dst and src must both be valid");
    }
    TESSIQ_RETURN_IF_ERROR(activate());

    // Queued on this device's stream and synchronised by the caller. The stream is a blocking one,
    // so the copy is ordered against kernels on the legacy default stream.
    if (const cudaError_t status = cudaMemcpyAsync(dst, src, bytes, direction, stream_);
        status != cudaSuccess) {
      return cuda_error(status, "cudaMemcpyAsync");
    }
    return ok();
  }

  int ordinal_;
  std::mutex stream_mutex_;
  cudaStream_t stream_ = nullptr; // Created on first use, then only read.
};

} // namespace

std::vector<std::unique_ptr<Device>> discover_cuda_devices() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    // No driver, no device, or a driver/runtime mismatch: no CUDA devices, and the CPU reference
    // is still there.
    TESSIQ_LOG(kInfo, kLogModule) << "no cuda devices: " << cudaGetErrorName(status) << " ("
                                  << cudaGetErrorString(status) << ")";
    return {};
  }

  std::vector<std::unique_ptr<Device>> devices;
  devices.reserve(static_cast<std::size_t>(count));
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    devices.push_back(std::make_unique<CudaDevice>(ordinal));
  }
  TESSIQ_LOG(kInfo, kLogModule) << "found " << count << " cuda device(s)";
  return devices;
}

Result<void*> allocate_page_locked(std::size_t bytes) {
  void* ptr = nullptr;
  if (const cudaError_t status = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
      status != cudaSuccess) {
    if (status == cudaErrorMemoryAllocation) {
      return resource_exhausted("could not page-lock " + std::to_string(bytes) + " host bytes");
    }
    return cuda_error(status, "cudaHostAlloc");
  }
  return ptr;
}

void free_page_locked(void* ptr) noexcept { static_cast<void>(cudaFreeHost(ptr)); }

bool is_page_locked(const void* ptr) noexcept {
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  return attributes.type == cudaMemoryTypeHost;
}

} // namespace tessiq

#else // TESSIQ_WITH_CUDA

namespace tessiq {

std::vector<std::unique_ptr<Device>> discover_cuda_devices() { return {}; }

Result<void*> allocate_page_locked(std::size_t /*bytes*/) {
  return unsupported("page-locked memory needs a build with CUDA");
}

void free_page_locked(void* /*ptr*/) noexcept {}

bool is_page_locked(const void* /*ptr*/) noexcept { return false; }

} // namespace tessiq

#endif // TESSIQ_WITH_CUDA
