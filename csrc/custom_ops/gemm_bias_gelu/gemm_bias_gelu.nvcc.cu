// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The host code for tessiq::gemm_bias_gelu: one CUTLASS GEMM, compiled by nvcc when tessiq is
// built.
//
// The epilogue, the stage that turns each block's accumulators into output elements, computes
// gelu(accumulator + bias) and writes that, so the product is never written out on its own. The
// bias is the GEMM's C operand with a row stride of zero, which reads the same row, one value per
// column, for every row of the output.
//
// The GEMM is CUTLASS's SIMT kernel: float32 multiplies and adds, like cuBLAS's SGEMM, rather than
// TF32 tensor cores, which round the inputs to 10 bits of mantissa. Three TF32 products per
// multiply recover float32's accuracy, and on an RTX 5080 were slower than this. When the output
// has too few tiles to give every multiprocessor work, K is split into slices that run side by side
// and add into the output in turn, and only the last applies GELU. The slices take their turns
// through semaphores in workspace.

#include <algorithm>
#include <climits>
#include <cstddef>
#include <string>

#include "tessiq/codegen/precompiled.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/gemm/device/gemm.h"

namespace {

using tessiq::codegen::Call;
using tessiq::codegen::DeviceTensor;

/// gelu(alpha * accumulator + beta * bias), one element at a time.
using Epilogue = cutlass::epilogue::thread::
    LinearCombinationGeneric<cutlass::epilogue::thread::GELU, float, 1, float, float>;

/// Rows and columns of the output one block writes.
constexpr long long kTileRows = 64;
constexpr long long kTileColumns = 128;

/// x is row-major [m, k]. The weight, row-major [n, k], is the column-major [k, n] matrix the GEMM
/// multiplies by, so it is read as it is stored.
///
/// A block of four warps writes a 64 x 128 tile, which on an RTX 5080 did better overall than
/// 128 x 128 or 32 x 128: a few percent behind 128 x 128 on the largest outputs, well ahead on a
/// few rows through a wide layer.
using Gemm =
    cutlass::gemm::device::Gemm<float,
                                cutlass::layout::RowMajor,
                                float,
                                cutlass::layout::ColumnMajor,
                                float,
                                cutlass::layout::RowMajor,
                                float,
                                cutlass::arch::OpClassSimt,
                                cutlass::arch::Sm80,
                                cutlass::gemm::GemmShape<kTileRows, kTileColumns, 8>,
                                cutlass::gemm::GemmShape<32, 64, 8>,
                                cutlass::gemm::GemmShape<1, 1, 1>,
                                Epilogue,
                                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                /*Stages=*/4,
                                /*AlignmentA=*/1,
                                /*AlignmentB=*/1,
                                /*SplitKSerial=*/true>;

/// The most slices K is split into. Past this the slices wait on each other more than they gain.
constexpr long long kMostSlices = 4;

/// The fewest elements of K a slice gets.
constexpr long long kLeastSliceDepth = 256;

/// Tiles per multiprocessor that splitting K aims for. Three blocks fit on a multiprocessor, so
/// this is four waves of them. Measured on an RTX 5080 over nine shapes, splitting up to this
/// did as well as the best slice count or close to it, and aiming for fewer tiles did worse.
constexpr long long kTilesPerMultiprocessor = 12;

/// How many slices to split K into: enough for kTilesPerMultiprocessor, at most kMostSlices, and
/// each keeping kLeastSliceDepth of K.
int slices_for(long long m, long long n, long long k, int multiprocessors) {
  const long long tiles =
      ((m + kTileRows - 1) / kTileRows) * ((n + kTileColumns - 1) / kTileColumns);
  const long long wanted = (kTilesPerMultiprocessor * multiprocessors + tiles - 1) / tiles;
  return static_cast<int>(std::max(1LL, std::min({wanted, kMostSlices, k / kLeastSliceDepth})));
}

void run(Call& call) {
  const DeviceTensor& x = call.input(0);
  const DeviceTensor& weight = call.input(1);
  const DeviceTensor& bias = call.input(2);
  if (x.sizes.size() != 2 || weight.sizes.size() != 2) {
    call.decline("x and weight have to be matrices");
    return;
  }
  const long long m = x.size(0);
  const long long k = x.size(1);
  const long long n = weight.size(0);
  if (weight.size(1) != k) {
    call.decline("weight's rows are not as long as x's");
    return;
  }
  // The reference broadcasts any bias that fits the output, but the epilogue reads one per column.
  if (bias.sizes.size() != 1 || bias.numel() != n) {
    call.decline("the epilogue reads one bias per column");
    return;
  }
  if (k == 0) {
    call.decline("there is nothing to multiply");
    return;
  }
  if (m > INT_MAX || n > INT_MAX || k > INT_MAX || (n + kTileColumns - 1) / kTileColumns > 65535) {
    call.decline("the GEMM takes its extents as int, and a grid holds 65535 tiles of columns");
    return;
  }

  int multiprocessors = 0;
  if (const cudaError_t status =
          cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, call.device());
      status != cudaSuccess) {
    call.fail(std::string("cudaDeviceGetAttribute: ") + cudaGetErrorString(status));
    return;
  }

  const Gemm::Arguments arguments({static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
                                  {x.as<const float>(), k},
                                  {weight.as<const float>(), k},
                                  {bias.as<const float>(), 0},
                                  {call.output().as<float>(), n},
                                  {/*alpha=*/1.0F, /*beta=*/1.0F},
                                  slices_for(m, n, k, multiprocessors));

  Gemm gemm;
  if (const cutlass::Status status = Gemm::can_implement(arguments);
      status != cutlass::Status::kSuccess) {
    call.decline(std::string("CUTLASS cannot run it: ") + cutlass::cutlassGetStatusString(status));
    return;
  }
  void* workspace = nullptr;
  if (const std::size_t bytes = Gemm::get_workspace_size(arguments); bytes > 0) {
    workspace = call.workspace(bytes);
    if (workspace == nullptr) {
      return;
    }
  }
  const auto stream = static_cast<cudaStream_t>(call.stream());
  // Zeroes the slices' semaphores on the stream, when there is more than one slice.
  if (const cutlass::Status status = gemm.initialize(arguments, workspace, stream);
      status != cutlass::Status::kSuccess) {
    call.fail(std::string("CUTLASS initialize: ") + cutlass::cutlassGetStatusString(status));
    return;
  }
  if (const cutlass::Status status = gemm.run(stream); status != cutlass::Status::kSuccess) {
    call.fail(std::string("CUTLASS run: ") + cutlass::cutlassGetStatusString(status));
  }
}

} // namespace

TESSIQ_PRECOMPILED(gemm_bias_gelu, run)
