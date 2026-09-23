// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// A two-dimensional convolution, described as the library's arguments rather than as a loop.
///
/// Like a matrix multiply this is a binding and not generated code. cuDNN picks the algorithm,
/// which is the whole value of calling it: a convolution has half a dozen reasonable
/// implementations and which is fastest depends on the shapes, the card and the library version.
/// What the compiler owns is recognising one, and saying exactly what it is.
///
/// Every tensor is NCHW and packed, which is the layout the imported graph gives a convolution's
/// input and weight in eval mode and the only one described here. A convolution handed anything
/// else is not one this describes, and runs on the host.
struct ConvKernel {
  /// The IR names of what it reads and writes.
  std::string input;
  std::string weight;
  /// Present when the convolution adds a per-channel bias. `nn.Conv2d(bias=False)` has none,
  /// which is what a convolution followed by a batch norm always is.
  std::optional<std::string> bias;
  std::string output;

  ir::DType dtype = ir::DType::kF32;

  /// [batch, channels, height, width] of the input, weight and output.
  std::array<std::int64_t, 4> input_shape{};
  std::array<std::int64_t, 4> weight_shape{};
  std::array<std::int64_t, 4> output_shape{};

  /// Per spatial axis, height then width.
  std::array<std::int64_t, 2> stride{1, 1};
  std::array<std::int64_t, 2> padding{0, 0};
  std::array<std::int64_t, 2> dilation{1, 1};

  /// Channel groups. One for an ordinary convolution, the channel count for a depthwise one.
  std::int64_t groups = 1;

  /// The IR names of the values read, in the order a launch takes them: input, weight, then the
  /// bias when there is one.
  [[nodiscard]] std::vector<std::string> inputs() const;
};

/// Describes `node` as a convolution, or nothing if it cannot be one here.
///
/// Nothing is not a failure. A transposed convolution, a one- or three-dimensional one, a shape
/// that is not known until the call and a layout other than packed NCHW all have no description,
/// and run on the host. An error means the node claims to be a convolution and contradicts itself.
Result<std::optional<ConvKernel>> extract_conv(const ir::Node& node);

/// The nodes a convolution could be built from, in graph order.
[[nodiscard]] std::vector<const ir::Node*> conv_nodes(const ir::Graph& graph);

} // namespace tessiq::codegen
