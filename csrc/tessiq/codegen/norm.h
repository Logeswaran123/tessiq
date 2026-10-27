// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

#include "tessiq/ir/types.h"

namespace tessiq::codegen {

/// Threads per block in the normalisation kernels. A multiple of the warp size, which the layer
/// norm's reduction needs.
inline constexpr unsigned int kNormBlockSize = 256;

/// Elements of a row each layer norm thread holds in registers. A row up to
/// `kNormHeldPerThread * kNormBlockSize` wide is read from memory once.
inline constexpr int kNormHeldPerThread = 8;

/// Source for a layer norm over rows of `width` packed elements, one block of `kNormBlockSize`
/// threads per row.
///
/// Parameters: input, weight, bias, output, mean, rstd, then `long long width` and `scalar_t eps`.
/// A null weight or bias is skipped. Each row's mean and reciprocal deviation are written for the
/// backward pass.
///
/// Two passes over the row, the second over centred values, as torch's CPU kernel computes the
/// variance, rather than one pass of sums and squares, which loses digits when the mean is large.
/// A row the block can hold stays in registers between the passes.
[[nodiscard]] std::string layer_norm_source(ir::DType dtype, std::string_view name);

/// Source for an eval-mode batch norm of packed `[batch, channels, inner]` elements, with two entry
/// points over the same parameters, input, weight, bias, running mean, running variance and output:
///
/// - `<name>_by_plane`: a block of `kNormBlockSize` threads per `inner` elements of one channel,
///   which finds its channel without dividing. Then `long long channels, long long inner` and
///   `scalar_t eps`. For an `inner` at least a block wide.
/// - `<name>_by_element`: a thread per element. Then `unsigned int total, channels, inner` and
///   `scalar_t eps`. For a narrower `inner`, such as `BatchNorm1d`'s 1, with fewer than 2^31
///   elements.
///
/// A null weight or bias is skipped. Each element computes `x * alpha + beta` with
/// `alpha = weight / sqrt(var + eps)` and `beta = bias - mean * alpha`, as torch's CPU kernel does.
[[nodiscard]] std::string batch_norm_source(ir::DType dtype, std::string_view name);

} // namespace tessiq::codegen
