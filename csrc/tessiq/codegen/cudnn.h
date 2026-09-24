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
/// Built once per compiled convolution and kept, because the expensive part of calling cuDNN is
/// not the call. It is describing the problem and choosing among the algorithms that could solve
/// it, and neither depends on the values being convolved. Doing that per call would cost more
/// than most of the convolutions it describes.
///
/// Abstract so that no cuDNN type reaches this header, which is included by code that has to
/// compile without the library.
class ConvPlan {
 public:
  virtual ~ConvPlan() = default;

  ConvPlan(const ConvPlan&) = delete;
  ConvPlan& operator=(const ConvPlan&) = delete;
  ConvPlan(ConvPlan&&) = delete;
  ConvPlan& operator=(ConvPlan&&) = delete;

  /// Runs it on `stream`. `bias` is null when the convolution has none. Asynchronous, like a
  /// kernel launch.
  [[nodiscard]] virtual Result<void> launch(const void* input,
                                            const void* weight,
                                            const void* bias,
                                            void* output,
                                            Stream stream) const = 0;

  /// The algorithm cuDNN chose, for reports and tests. Its own enumeration, as an integer.
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
/// The arithmetic is kept in fp32. torch lets cuDNN use TF32 tensor cores for convolutions by
/// default, which is faster and agrees with fp32 to about three decimal places. This does not,
/// because the answer is checked against the host and the host does not round that way. The
/// consequence is that a timing against eager CUDA compares an fp32 convolution with a TF32 one,
/// and says so where it is reported.
Result<std::unique_ptr<ConvPlan>> make_conv_plan(const ConvKernel& kernel, int ordinal);

} // namespace tessiq::codegen
