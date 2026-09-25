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
/// Fusion produces a description and an emitter decides what code it becomes, so a second target
/// is a second class.
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

  /// Source text for a reduction, which is its own kernel rather than a step in a fused loop.
  [[nodiscard]] virtual Result<std::string> emit(const ReductionKernel& kernel) const = 0;

 protected:
  Emitter() = default;
};

/// Emits CUDA C++ for NVRTC.
const Emitter& cuda_emitter();

} // namespace tessiq::codegen
