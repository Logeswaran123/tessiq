// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "tessiq/codegen/conv.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// A convolution with its descriptors built and its algorithm chosen, ready to run.
///
/// Built once per compiled convolution and kept. Building the descriptors and choosing an algorithm
/// costs more than most convolutions, and neither depends on the data.
///
/// Abstract so that no cuDNN type reaches this header, which code built without cuDNN includes.
class ConvPlan {
 public:
  virtual ~ConvPlan() = default;

  ConvPlan(const ConvPlan&) = delete;
  ConvPlan& operator=(const ConvPlan&) = delete;
  ConvPlan(ConvPlan&&) = delete;
  ConvPlan& operator=(ConvPlan&&) = delete;

  /// Runs it asynchronously on `stream`. `bias` is null when the convolution has none.
  [[nodiscard]] virtual Result<void> launch(const void* input,
                                            const void* weight,
                                            const void* bias,
                                            void* output,
                                            Stream stream) const = 0;

  /// The `cudnnConvolutionFwdAlgo_t` cuDNN chose, for reports and tests.
  [[nodiscard]] virtual int algorithm() const = 0;

  /// Scratch bytes the chosen algorithm asked for.
  [[nodiscard]] virtual std::size_t workspace_bytes() const = 0;

 protected:
  ConvPlan() = default;
};

/// Whether this build can call cuDNN at all.
[[nodiscard]] bool cudnn_available();

/// Describes `kernel` to cuDNN on device `ordinal` and chooses how to run it.
///
/// Uses fp32 math, not TF32. torch lets cuDNN use TF32 by default, which is faster but agrees
/// with fp32 to only about three digits, and results here are checked against the host. A timing
/// against eager CUDA therefore compares fp32 with TF32.
Result<std::unique_ptr<ConvPlan>> make_conv_plan(const ConvKernel& kernel, int ordinal);

} // namespace tessiq::codegen
