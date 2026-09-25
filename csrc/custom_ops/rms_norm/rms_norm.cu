// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// RMSNorm over the last axis, one block per row.
//
// Each block sums the squares of its row, shares the partial sums through shared memory, and then
// scales the row by the reciprocal of its root mean square and by the weight. Both passes read the
// row, so it is read twice and written once, in one launch. Written out in PyTorch the same
// operation is a reduction and several elementwise kernels.
//
// Launched by rms_norm.cpp: a grid of one block per row, a block of kThreads threads, and
// kThreads floats of shared memory. The block size is a power of two, because the sum halves the
// block at every step.

extern "C" __global__ void rms_norm(const float* __restrict__ x,
                                    const float* __restrict__ weight,
                                    float* __restrict__ out,
                                    long long columns,
                                    float eps) {
  extern __shared__ float partial[];
  const float* row = x + (long long)blockIdx.x * columns;
  float* result = out + (long long)blockIdx.x * columns;

  float sum = 0.0f;
  for (long long c = threadIdx.x; c < columns; c += blockDim.x) {
    sum += row[c] * row[c];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale = rsqrtf(partial[0] / (float)columns + eps);
  for (long long c = threadIdx.x; c < columns; c += blockDim.x) {
    result[c] = row[c] * scale * weight[c];
  }
}
