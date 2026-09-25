// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax along any dimension, in one launch, standing in for torch's own.
//
// It replaces aten::_softmax, which torch.softmax, F.softmax and nn.Softmax all reach. Without it,
// tessiq breaks a softmax into a max, a subtract, an exp, a sum and a divide, which is two
// reductions and a pass over the tensor for each of the elementwise steps. The kernel is
// softmax.cu, beside this file. Float32 only; any other type runs torch's softmax.

#include <ATen/ATen.h>
#include <cstdint>
#include <limits>

#include "tessiq/torch/custom_op.h"

namespace {

using tessiq::custom_ops::Launch;

/// Threads per block. A power of two, because the kernel's reductions halve the block each step.
constexpr std::uint32_t kThreads = 256;

/// What the kernel computes, which is torch's own softmax.
at::Tensor softmax(const at::Tensor& self, std::int64_t dim, bool half_to_float) {
  return at::_softmax(self, dim, half_to_float);
}

/// One block per softmax, with one float of shared memory per thread for its partial results.
Launch launch(const at::Tensor& self,
              std::int64_t dim,
              bool /*half_to_float*/,
              const at::Tensor& /*out*/) {
  // A tensor with no dimensions is one softmax of one element.
  const std::int64_t along = self.dim() == 0 ? 0 : at::maybe_wrap_dim(dim, self.dim());
  const std::int64_t extent = self.dim() == 0 ? 1 : self.size(along);
  std::int64_t inner = 1;
  for (std::int64_t d = along + 1; d < self.dim(); ++d) {
    inner *= self.size(d);
  }
  const std::int64_t blocks = self.numel() / extent;
  TORCH_CHECK(blocks <= std::numeric_limits<std::int32_t>::max(),
              "the softmax kernel runs a block per softmax, and ",
              blocks,
              " is more blocks than a grid holds");

  Launch launch;
  launch.grid = {static_cast<std::uint32_t>(blocks), 1, 1};
  launch.block = {kThreads, 1, 1};
  launch.shared_bytes = kThreads * sizeof(float);
  launch.arguments = {extent, inner};
  return launch;
}

} // namespace

TESSIQ_CUSTOM_OP(softmax, "(Tensor self, int dim, bool half_to_float) -> Tensor")
    .reference(softmax)
    .launch(launch)
    .dtypes({at::kFloat})
    .replaces("aten::_softmax");
