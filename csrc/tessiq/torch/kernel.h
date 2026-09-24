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
/// A fused loop, a reduction, a cuBLAS multiply and a cuDNN convolution differ in almost
/// everything, but the executor asks each of them the same few questions: what it reads, what it
/// writes, how big the result is, and how to run it on a stream. This is those questions. The
/// executor holds a list of kernels without knowing how many kinds exist, so a new kind is a new
/// implementation of this rather than a new branch somewhere downstream.
///
/// Names are IR value names, which the importer took from the FX nodes, so the executor can find
/// the tensors a kernel reads and the node it stands in for in the graph it started with.
class Kernel {
 public:
  virtual ~Kernel() = default;

  /// What kind of kernel this is, for reports and tests. Nothing decides anything on it.
  [[nodiscard]] virtual std::string_view kind() const = 0;

  /// The values it reads, in the order `launch` takes them.
  [[nodiscard]] virtual std::vector<std::string> inputs() const = 0;

  /// The value it writes, which is the node the executor runs it in place of.
  [[nodiscard]] virtual const std::string& output() const = 0;

  /// Every value it computes, the output included.
  ///
  /// The executor skips the others when it walks the graph, because this computes them on the way.
  /// Most kernels compute only their output. A fused loop computes the whole group.
  [[nodiscard]] virtual std::vector<std::string> members() const { return {output()}; }

  /// Whether the executor has to pack each input before the launch.
  ///
  /// A kernel that walks one index across its buffers reads the right elements only when they are
  /// laid out the same way. A kernel that is told each operand's layout, as a matmul is, must not
  /// have them packed, because packing would leave that description pointing at data that moved.
  [[nodiscard]] virtual bool packs_inputs() const = 0;

  /// Whether some extent it needs arrives with the tensors rather than being known here.
  [[nodiscard]] virtual bool has_dynamic_extent() const = 0;

  /// The shape of the tensor it writes, given the ones it will read.
  ///
  /// The executor usually sizes the result from the node's fake tensor. It asks this instead when
  /// the graph was traced for more than one shape, since the fake tensor then has no numbers.
  [[nodiscard]] virtual Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const = 0;

  /// Runs it, reading `tensors` in the order `inputs()` names them and writing `out`, on `stream`.
  ///
  /// Asynchronous, like any launch. The result is ready once the stream has reached it, and the
  /// caller orders anything that reads it by choosing the stream.
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
