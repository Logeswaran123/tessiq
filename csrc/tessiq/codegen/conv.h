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
/// Like a matmul, this is a library call rather than generated code. cuDNN picks the algorithm,
/// since the fastest one depends on the shapes, the card and the library version.
///
/// Every tensor is packed NCHW. A convolution with any other layout runs on the host.
struct ConvKernel {
  /// The IR names of what it reads and writes.
  std::string input;
  std::string weight;
  /// The per-channel bias. Absent for `nn.Conv2d(bias=False)`, as is usual before a batch norm.
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
/// Nothing for a transposed convolution, a 1-D or 3-D one, an unknown shape, or a layout other
/// than packed NCHW; those run on the host. An error means the node is malformed.
Result<std::optional<ConvKernel>> extract_conv(const ir::Node& node);

/// The nodes a convolution could be built from, in graph order.
[[nodiscard]] std::vector<const ir::Node*> conv_nodes(const ir::Graph& graph);

} // namespace tessiq::codegen
