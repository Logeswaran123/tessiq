// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The host code for tessiq::layer_norm, compiled by nvcc when tessiq is built.
//
// Two launches on the stream the call is handed. The first runs a block per row and writes the
// row's mean and reciprocal standard deviation into workspace. The second runs a thread per
// element and applies them, with the weight and bias. The variance is taken over deviations from
// the mean rather than as a mean of squares less a squared mean, which loses precision when the
// mean is large against the spread.

#include <climits>
#include <cstddef>
#include <cstdint>

#include "tessiq/codegen/precompiled.h"

namespace {

using tessiq::codegen::Call;
using tessiq::codegen::DeviceTensor;

/// Threads per block. A power of two, because block_sum halves the block at every step.
constexpr unsigned int kThreads = 256;

/// The sum of `value` over the block, returned to every thread. `partial` holds kThreads floats.
__device__ float block_sum(float value, float* partial) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float total = partial[0];
  // Before anyone writes `partial` for the next sum.
  __syncthreads();
  return total;
}

/// One block per row: the row's mean and reciprocal standard deviation, into `statistics`.
__global__ void row_statistics(const float* x, float2* statistics, long long columns, float eps) {
  __shared__ float partial[kThreads];
  const float* row = x + static_cast<long long>(blockIdx.x) * columns;

  float sum = 0.0F;
  for (long long column = threadIdx.x; column < columns; column += kThreads) {
    sum += row[column];
  }
  const float mean = block_sum(sum, partial) / static_cast<float>(columns);

  float squares = 0.0F;
  for (long long column = threadIdx.x; column < columns; column += kThreads) {
    const float deviation = row[column] - mean;
    squares += deviation * deviation;
  }
  const float variance = block_sum(squares, partial) / static_cast<float>(columns);

  if (threadIdx.x == 0) {
    statistics[blockIdx.x] = make_float2(mean, rsqrtf(variance + eps));
  }
}

/// One thread per element: normalise with the row's statistics, then scale and shift.
__global__ void apply(const float* x,
                      const float2* statistics,
                      const float* weight,
                      const float* bias,
                      float* out,
                      long long columns,
                      long long count) {
  const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float2 row = statistics[i / columns];
  const long long column = i % columns;
  out[i] = (x[i] - row.x) * row.y * weight[column] + bias[column];
}

void run(Call& call) {
  const DeviceTensor& x = call.input(0);
  const DeviceTensor& weight = call.input(1);
  const DeviceTensor& bias = call.input(2);
  if (x.sizes.empty()) {
    call.decline("x is a scalar, with no axis to normalise over");
    return;
  }
  const long long columns = x.size(-1);
  const long long count = x.numel();
  const long long rows = count / columns;
  // The reference broadcasts any weight and bias that fit, but the kernel reads one of each per
  // column.
  if (weight.numel() != columns || bias.numel() != columns) {
    call.decline("the kernel reads one weight and one bias per column");
    return;
  }
  const long long blocks = (count + kThreads - 1) / kThreads;
  if (rows > INT_MAX || blocks > INT_MAX) {
    call.decline("more blocks than a grid holds");
    return;
  }

  auto* statistics =
      static_cast<float2*>(call.workspace(static_cast<std::size_t>(rows) * sizeof(float2)));
  if (statistics == nullptr) {
    return;
  }
  const auto stream = static_cast<cudaStream_t>(call.stream());
  row_statistics<<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
      x.as<const float>(), statistics, columns, static_cast<float>(call.real(0)));
  apply<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(x.as<const float>(),
                                                                    statistics,
                                                                    weight.as<const float>(),
                                                                    bias.as<const float>(),
                                                                    call.output().as<float>(),
                                                                    columns,
                                                                    count);
}

} // namespace

TESSIQ_PRECOMPILED(layer_norm, run)
