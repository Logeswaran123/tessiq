// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax along one dimension, one block per softmax.
//
// The tensor is packed and read as [outer, extent, inner], where extent is the dimension the
// softmax runs along. Block b takes the elements at outer index b / inner and inner index b %
// inner, which sit `inner` apart. Along the last dimension inner is 1, so a block reads one
// contiguous row.
//
// Three passes over the elements: the largest, then the sum of exponentials of each element less
// the largest, then each exponential divided by that sum. Subtracting the largest keeps every
// exponential at most 1, so none overflows. Each pass shares its partial results through shared
// memory, and the block size is a power of two because the sums halve the block at every step.
//
// Launched by softmax.cpp: a grid of outer * inner blocks, a block of kThreads threads, and
// kThreads floats of shared memory.

extern "C" __global__ void softmax(const float* __restrict__ self,
                                   float* __restrict__ out,
                                   long long extent,
                                   long long inner) {
  extern __shared__ float partial[];
  const long long block = blockIdx.x;
  const long long first = (block / inner) * extent * inner + block % inner;
  const float* x = self + first;
  float* y = out + first;

  float largest = __int_as_float(0xff800000); // negative infinity
  for (long long i = threadIdx.x; i < extent; i += blockDim.x) {
    largest = fmaxf(largest, x[i * inner]);
  }
  partial[threadIdx.x] = largest;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  largest = partial[0];
  __syncthreads();

  float sum = 0.0f;
  for (long long i = threadIdx.x; i < extent; i += blockDim.x) {
    sum += expf(x[i * inner] - largest);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale = 1.0f / partial[0];
  for (long long i = threadIdx.x; i < extent; i += blockDim.x) {
    y[i * inner] = expf(x[i * inner] - largest) * scale;
  }
}
