// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <cstdint>
#include <map>
#include <string>

namespace tessiq::torch_backend {

/// Runs an op with no tessiq kernel by moving its tensors to the host, calling PyTorch's own
/// kernel, and moving the results back. Counts the call.
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

/// Counts a call of `op` that ran on the host, for an operation whose host path is its own rather
/// than `cpu_fallback`.
void count_cpu_fallback(const c10::OperatorName& op);

/// How many calls of each operation ran on the host, by operator name such as `aten::add.Tensor`.
/// An operation that never fell back is absent.
std::map<std::string, std::int64_t> cpu_fallback_counts();

} // namespace tessiq::torch_backend
