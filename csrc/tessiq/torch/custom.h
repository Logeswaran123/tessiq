// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string_view>

#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/kernel.h"

namespace tessiq::torch_backend {

/// A kernel running registered elementwise operation `op` on its own, for the eager path.
///
/// Built by the compiler itself rather than beside it: a one-node graph of `arity` operands of
/// `dtype` with every extent unknown goes through the same passes and the same emitter a compiled
/// graph does. So a call made outside `torch.compile` runs exactly the code a compiled graph would
/// fuse, and the one kernel serves every shape, because it takes its element count at the launch.
/// Every operand has to be the result's shape, which the caller arranges by broadcasting first.
///
/// Refused when `op` is not an elementwise operation the compiler knows, and when there is no
/// kernel for `dtype`, which includes a build with no CUDA toolkit.
[[nodiscard]] Result<std::shared_ptr<Kernel>>
compile_elementwise(std::string_view op, int arity, ir::DType dtype, int ordinal);

} // namespace tessiq::torch_backend
