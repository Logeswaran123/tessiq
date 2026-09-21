// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/reduction.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// Turns a fused kernel into source for one target.
///
/// The interface exists so that fusion produces a description and something else decides what code
/// that becomes. Keeping those apart is what lets a second target be a second class rather than a
/// second path through the same function, and it is what makes two targets comparable: point them
/// at the same `FusedKernel` and the only difference is the code.
///
/// Thread-compatible. An emitter holds no state between calls, so one instance may be shared.
class Emitter {
 public:
  virtual ~Emitter() = default;

  Emitter(const Emitter&) = delete;
  Emitter& operator=(const Emitter&) = delete;
  Emitter(Emitter&&) = delete;
  Emitter& operator=(Emitter&&) = delete;

  /// The name this emitter is selected by.
  [[nodiscard]] virtual std::string_view name() const = 0;

  /// Whether this emitter can produce code for `kernel`. A false answer is not a failure: the
  /// caller falls back to running the operations one at a time.
  [[nodiscard]] virtual bool supports(const FusedKernel& kernel) const = 0;

  /// Whether it can write this reduction.
  [[nodiscard]] virtual bool supports(const ReductionKernel& kernel) const = 0;

  /// Source text for `kernel`. Only called when `supports` is true.
  [[nodiscard]] virtual Result<std::string> emit(const FusedKernel& kernel) const = 0;

  /// The source for one reduction. A reduction is its own kernel rather than a step in a fused
  /// loop, so it is a separate call rather than another kind of step.
  [[nodiscard]] virtual Result<std::string> emit(const ReductionKernel& kernel) const = 0;

 protected:
  Emitter() = default;
};

/// Emits CUDA C++ for NVRTC.
const Emitter& cuda_emitter();

} // namespace tessiq::codegen
