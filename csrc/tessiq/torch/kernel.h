// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/codegen/stream.h"
#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Compiled work the executor can run, whatever produced it.
///
/// A fused loop, a reduction, a cuBLAS multiply, a cuDNN convolution and a custom operation all
/// answer the same questions: what it reads, what it writes, how big the result is, and how to run
/// it on a stream. A new kind of kernel is a new implementation of this.
///
/// Names are IR value names, which the importer took from the FX nodes, so the executor can find
/// the tensors a kernel reads and the node it replaces.
class Kernel {
 public:
  virtual ~Kernel() = default;

  /// What kind of kernel this is, for reports and tests only.
  [[nodiscard]] virtual std::string_view kind() const = 0;

  /// The values it reads, in the order `launch` takes them.
  [[nodiscard]] virtual std::vector<std::string> inputs() const = 0;

  /// The value it writes, which names the node the executor runs it in place of.
  [[nodiscard]] virtual const std::string& output() const = 0;

  /// Every value it computes, the output included. The executor skips the others when it walks
  /// the graph. A fused loop computes its whole group; most kernels compute only their output.
  [[nodiscard]] virtual std::vector<std::string> members() const { return {output()}; }

  /// Whether the executor has to pack each input before the launch.
  ///
  /// True for a kernel that assumes packed buffers. False for one that is told each operand's
  /// layout, as a matmul is, since packing would invalidate that description.
  [[nodiscard]] virtual bool packs_inputs() const = 0;

  /// Whether some extent it needs arrives with the tensors rather than being known here.
  [[nodiscard]] virtual bool has_dynamic_extent() const = 0;

  /// The shape of the tensor it writes, given the ones it will read.
  ///
  /// The executor asks this only when the graph was traced for more than one shape. Otherwise it
  /// sizes the result from the node's fake tensor.
  [[nodiscard]] virtual Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const = 0;

  /// Runs it, reading `tensors` in the order `inputs()` names them and writing `out`, on `stream`.
  ///
  /// Asynchronous; the caller orders readers of the result through `stream`.
  [[nodiscard]] virtual Result<void> launch(const std::vector<at::Tensor>& tensors,
                                            const at::Tensor& out,
                                            codegen::Stream stream) const = 0;

 protected:
  Kernel() = default;
  Kernel(const Kernel&) = default;
  Kernel& operator=(const Kernel&) = default;
  Kernel(Kernel&&) = default;
  Kernel& operator=(Kernel&&) = default;
};

} // namespace tessiq::torch_backend
