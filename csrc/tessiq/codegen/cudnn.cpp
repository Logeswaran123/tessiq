// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/cudnn.h"

#include <memory>
#include <string>

#include "tessiq/codegen/conv.h"
#include "tessiq/support/result.h"

#ifdef TESSIQ_WITH_CUDNN

#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

#include <cuda_runtime.h>

namespace tessiq::codegen {

namespace {

/// What a failed call says, built once so every return type can use it.
std::string cudnn_message(cudnnStatus_t status, const char* what) {
  return std::string(what) + ": " + cudnnGetErrorString(status);
}

/// Held from choosing a handle's stream until the calls on it are queued.
std::mutex& launch_mutex() {
  static std::mutex mutex;
  return mutex;
}

/// One handle per device, created on first use and kept.
///
/// A handle carries a context and is not free to create, and every plan on a device can share
/// one. Its stream is set on every launch to the one the caller names, which is what orders a
/// convolution against the kernels either side of it without anyone synchronising.
Result<cudnnHandle_t> handle_for(int ordinal) {
  static std::mutex mutex;
  static std::map<int, cudnnHandle_t> handles;
  const std::scoped_lock guard(mutex);
  if (const auto found = handles.find(ordinal); found != handles.end()) {
    return found->second;
  }
  if (cudaSetDevice(ordinal) != cudaSuccess) {
    return external_error("cudaSetDevice(" + std::to_string(ordinal) + ") failed");
  }
  cudnnHandle_t handle = nullptr;
  if (const cudnnStatus_t status = cudnnCreate(&handle); status != CUDNN_STATUS_SUCCESS) {
    return external_error(cudnn_message(status, "cudnnCreate"));
  }
  handles.emplace(ordinal, handle);
  return handle;
}

/// A packed NCHW tensor descriptor for these four extents.
Result<cudnnTensorDescriptor_t> tensor_descriptor(const std::array<std::int64_t, 4>& shape) {
  cudnnTensorDescriptor_t descriptor = nullptr;
  if (const cudnnStatus_t status = cudnnCreateTensorDescriptor(&descriptor);
      status != CUDNN_STATUS_SUCCESS) {
    return external_error(cudnn_message(status, "cudnnCreateTensorDescriptor"));
  }
  if (const cudnnStatus_t status = cudnnSetTensor4dDescriptor(descriptor,
                                                              CUDNN_TENSOR_NCHW,
                                                              CUDNN_DATA_FLOAT,
                                                              static_cast<int>(shape[0]),
                                                              static_cast<int>(shape[1]),
                                                              static_cast<int>(shape[2]),
                                                              static_cast<int>(shape[3]));
      status != CUDNN_STATUS_SUCCESS) {
    static_cast<void>(cudnnDestroyTensorDescriptor(descriptor));
    return external_error(cudnn_message(status, "cudnnSetTensor4dDescriptor"));
  }
  return descriptor;
}

class CudnnConvPlan final : public ConvPlan {
 public:
  ~CudnnConvPlan() override {
    if (workspace_ != nullptr) {
      static_cast<void>(cudaFree(workspace_));
    }
    if (bias_ != nullptr) {
      static_cast<void>(cudnnDestroyTensorDescriptor(bias_));
    }
    if (output_ != nullptr) {
      static_cast<void>(cudnnDestroyTensorDescriptor(output_));
    }
    if (input_ != nullptr) {
      static_cast<void>(cudnnDestroyTensorDescriptor(input_));
    }
    if (filter_ != nullptr) {
      static_cast<void>(cudnnDestroyFilterDescriptor(filter_));
    }
    if (convolution_ != nullptr) {
      static_cast<void>(cudnnDestroyConvolutionDescriptor(convolution_));
    }
  }

  CudnnConvPlan(const CudnnConvPlan&) = delete;
  CudnnConvPlan& operator=(const CudnnConvPlan&) = delete;
  CudnnConvPlan(CudnnConvPlan&&) = delete;
  CudnnConvPlan& operator=(CudnnConvPlan&&) = delete;

  static Result<std::unique_ptr<CudnnConvPlan>> build(const ConvKernel& kernel, int ordinal) {
    // Constructed first and filled in, so that a failure part way through still destroys
    // whatever was created before it.
    std::unique_ptr<CudnnConvPlan> plan(new CudnnConvPlan(ordinal));
    TESSIQ_ASSIGN_OR_RETURN(plan->handle_, handle_for(ordinal));
    TESSIQ_ASSIGN_OR_RETURN(plan->input_, tensor_descriptor(kernel.input_shape));
    TESSIQ_ASSIGN_OR_RETURN(plan->output_, tensor_descriptor(kernel.output_shape));
    if (kernel.bias.has_value()) {
      // One value per output channel, broadcast over the batch and both spatial axes.
      TESSIQ_ASSIGN_OR_RETURN(plan->bias_, tensor_descriptor({1, kernel.output_shape[1], 1, 1}));
    }
    TESSIQ_RETURN_IF_ERROR(plan->describe_filter(kernel));
    TESSIQ_RETURN_IF_ERROR(plan->describe_convolution(kernel));
    TESSIQ_RETURN_IF_ERROR(plan->check_output(kernel));
    TESSIQ_RETURN_IF_ERROR(plan->choose_algorithm());
    return plan;
  }

  [[nodiscard]] Result<void> launch(const void* input,
                                    const void* weight,
                                    const void* bias,
                                    void* output,
                                    Stream stream) const override {
    if (cudaSetDevice(ordinal_) != cudaSuccess) {
      return external_error("cudaSetDevice(" + std::to_string(ordinal_) + ") failed");
    }
    // The handle is shared by every plan on the device and carries one stream at a time, so the
    // stream and the calls queued on it are kept together. Queuing does not wait for the device.
    const std::scoped_lock guard(launch_mutex());
    if (const cudnnStatus_t status =
            cudnnSetStream(handle_, static_cast<cudaStream_t>(stream.handle));
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnSetStream"));
    }
    const float one = 1.0F;
    const float zero = 0.0F;
    if (const cudnnStatus_t status = cudnnConvolutionForward(handle_,
                                                             &one,
                                                             input_,
                                                             input,
                                                             filter_,
                                                             weight,
                                                             convolution_,
                                                             algorithm_,
                                                             workspace_,
                                                             workspace_bytes_,
                                                             &zero,
                                                             output_,
                                                             output);
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnConvolutionForward"));
    }
    if (bias_ != nullptr && bias != nullptr) {
      // Added into the result, which is what beta of one means: output = 1 * bias + 1 * output.
      if (const cudnnStatus_t status =
              cudnnAddTensor(handle_, &one, bias_, bias, &one, output_, output);
          status != CUDNN_STATUS_SUCCESS) {
        return external_error(cudnn_message(status, "cudnnAddTensor"));
      }
    }
    return ok();
  }

  [[nodiscard]] int algorithm() const override { return static_cast<int>(algorithm_); }

  [[nodiscard]] std::size_t workspace_bytes() const override { return workspace_bytes_; }

 private:
  explicit CudnnConvPlan(int ordinal) : ordinal_(ordinal) {}

  Result<void> describe_filter(const ConvKernel& kernel) {
    if (const cudnnStatus_t status = cudnnCreateFilterDescriptor(&filter_);
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnCreateFilterDescriptor"));
    }
    const auto& w = kernel.weight_shape;
    if (const cudnnStatus_t status = cudnnSetFilter4dDescriptor(filter_,
                                                                CUDNN_DATA_FLOAT,
                                                                CUDNN_TENSOR_NCHW,
                                                                static_cast<int>(w[0]),
                                                                static_cast<int>(w[1]),
                                                                static_cast<int>(w[2]),
                                                                static_cast<int>(w[3]));
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnSetFilter4dDescriptor"));
    }
    return ok();
  }

  Result<void> describe_convolution(const ConvKernel& kernel) {
    if (const cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&convolution_);
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnCreateConvolutionDescriptor"));
    }
    // Cross-correlation, which is what every deep learning framework calls convolution. The
    // other mode flips the filter and gives an answer nobody asked for.
    if (const cudnnStatus_t status =
            cudnnSetConvolution2dDescriptor(convolution_,
                                            static_cast<int>(kernel.padding[0]),
                                            static_cast<int>(kernel.padding[1]),
                                            static_cast<int>(kernel.stride[0]),
                                            static_cast<int>(kernel.stride[1]),
                                            static_cast<int>(kernel.dilation[0]),
                                            static_cast<int>(kernel.dilation[1]),
                                            CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT);
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnSetConvolution2dDescriptor"));
    }
    if (const cudnnStatus_t status =
            cudnnSetConvolutionGroupCount(convolution_, static_cast<int>(kernel.groups));
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnSetConvolutionGroupCount"));
    }
    // Plain fp32 rather than TF32 tensor cores. See make_conv_plan.
    if (const cudnnStatus_t status = cudnnSetConvolutionMathType(convolution_, CUDNN_FMA_MATH);
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnSetConvolutionMathType"));
    }
    return ok();
  }

  /// Asks cuDNN what shape the result is and checks it against the graph's.
  ///
  /// The two should agree, since both came from the same stride, padding and dilation. A
  /// disagreement means this read an attribute differently from how ATen did, and would write
  /// the wrong number of elements into a buffer sized for the right number.
  Result<void> check_output(const ConvKernel& kernel) const {
    int batch = 0;
    int channels = 0;
    int height = 0;
    int width = 0;
    if (const cudnnStatus_t status = cudnnGetConvolution2dForwardOutputDim(
            convolution_, input_, filter_, &batch, &channels, &height, &width);
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnGetConvolution2dForwardOutputDim"));
    }
    const std::array<std::int64_t, 4> reported{batch, channels, height, width};
    if (reported != kernel.output_shape) {
      return internal_error("cuDNN and the graph disagree about the shape of " + kernel.output);
    }
    return ok();
  }

  /// Picks the algorithm cuDNN's heuristic ranks first among those that fit in memory.
  ///
  /// The heuristic rather than a timed search. A search runs every candidate and is the right
  /// thing for a long-lived deployment; here it would run on every compile, and the heuristic's
  /// first choice is usually the one a search would find.
  Result<void> choose_algorithm() {
    constexpr int kCandidates = 8;
    std::array<cudnnConvolutionFwdAlgoPerf_t, kCandidates> candidates{};
    int returned = 0;
    if (const cudnnStatus_t status = cudnnGetConvolutionForwardAlgorithm_v7(handle_,
                                                                            input_,
                                                                            filter_,
                                                                            convolution_,
                                                                            output_,
                                                                            kCandidates,
                                                                            &returned,
                                                                            candidates.data());
        status != CUDNN_STATUS_SUCCESS) {
      return external_error(cudnn_message(status, "cudnnGetConvolutionForwardAlgorithm_v7"));
    }
    auto* const ranked_end = candidates.begin() + std::min(returned, kCandidates);
    for (auto* ranked = candidates.begin(); ranked != ranked_end; ++ranked) {
      const cudnnConvolutionFwdAlgoPerf_t& candidate = *ranked;
      if (candidate.status != CUDNN_STATUS_SUCCESS) {
        continue;
      }
      void* workspace = nullptr;
      if (candidate.memory > 0 && cudaMalloc(&workspace, candidate.memory) != cudaSuccess) {
        continue;
      }
      algorithm_ = candidate.algo;
      workspace_ = workspace;
      workspace_bytes_ = candidate.memory;
      return ok();
    }
    return unsupported("cuDNN has no algorithm for this convolution that fits in memory");
  }

  int ordinal_ = 0;
  cudnnHandle_t handle_ = nullptr;
  cudnnTensorDescriptor_t input_ = nullptr;
  cudnnTensorDescriptor_t output_ = nullptr;
  cudnnTensorDescriptor_t bias_ = nullptr;
  cudnnFilterDescriptor_t filter_ = nullptr;
  cudnnConvolutionDescriptor_t convolution_ = nullptr;
  cudnnConvolutionFwdAlgo_t algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  void* workspace_ = nullptr;
  std::size_t workspace_bytes_ = 0;
};

} // namespace

bool cudnn_available() { return true; }

Result<std::unique_ptr<ConvPlan>> make_conv_plan(const ConvKernel& kernel, int ordinal) {
  if (kernel.dtype != ir::DType::kF32) {
    return unsupported("cuDNN convolution is float32 only for now");
  }
  TESSIQ_ASSIGN_OR_RETURN(std::unique_ptr<CudnnConvPlan> plan,
                          CudnnConvPlan::build(kernel, ordinal));
  return std::unique_ptr<ConvPlan>(std::move(plan));
}

} // namespace tessiq::codegen

#else // TESSIQ_WITH_CUDNN

namespace tessiq::codegen {

bool cudnn_available() { return false; }

Result<std::unique_ptr<ConvPlan>> make_conv_plan(const ConvKernel& /*kernel*/, int /*ordinal*/) {
  return unsupported("this build has no cuDNN, so convolution runs on the host");
}

} // namespace tessiq::codegen

#endif // TESSIQ_WITH_CUDNN
