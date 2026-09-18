// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/runtime/cuda_device.h"

#ifdef TESSIQ_WITH_CUDA

#include <cstddef>
#include <memory>
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
/// Every entry point sets the device first. The CUDA runtime keeps a per-thread current device,
/// and tessiq's own current index is a separate notion, so relying on ambient state here would
/// make correctness depend on which thread called last.
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
    // cudaMalloc returns memory aligned to at least 256 bytes, and in practice to the page. That
    // satisfies kDeviceAlignment, which the Device contract requires us to guarantee.
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
    // Runs from destructors, so it cannot fail. A failure here means the context is already gone,
    // which is not something the caller can act on.
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
    if (stream_ == nullptr) {
      return ok();
    }
    if (const cudaError_t status = cudaStreamSynchronize(stream_); status != cudaSuccess) {
      return cuda_error(status, "cudaStreamSynchronize");
    }
    return ok();
  }

 private:
  Result<void> activate() {
    if (const cudaError_t status = cudaSetDevice(ordinal_); status != cudaSuccess) {
      return cuda_error(status, "cudaSetDevice(" + std::to_string(ordinal_) + ")");
    }
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

    // Async on our own stream, then synchronised by the caller. Nothing above this layer overlaps
    // transfers with compute yet, so the win is not latency; it is that every transfer is ordered
    // against the same stream the kernels will use once they exist.
    if (const cudaError_t status = cudaMemcpyAsync(dst, src, bytes, direction, stream_);
        status != cudaSuccess) {
      return cuda_error(status, "cudaMemcpyAsync");
    }
    return ok();
  }

  int ordinal_;
  cudaStream_t stream_ = nullptr;
};

} // namespace

std::vector<std::unique_ptr<Device>> discover_cuda_devices() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    // No driver, no device, or a driver/runtime mismatch. All of them mean the same thing here:
    // there is no CUDA to offer, and the CPU reference carries on.
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

} // namespace tessiq

#else // TESSIQ_WITH_CUDA

namespace tessiq {

std::vector<std::unique_ptr<Device>> discover_cuda_devices() { return {}; }

} // namespace tessiq

#endif // TESSIQ_WITH_CUDA
