// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Matrix multiplies called outside a compiled graph.

#include "tessiq/torch/eager_matmul.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/addmm_ops.h>
#include <ATen/ops/bmm_ops.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mm_ops.h>
#include <array>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>

#include "tessiq/codegen/blas.h"
#include "tessiq/codegen/gemm.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/fallback.h"
#include "tessiq/torch/importer.h"
#include "tessiq/torch/pack.h"

namespace tessiq::torch_backend {

namespace {

/// One operand's last two dimensions as cuBLAS reads them.
struct Matrix {
  /// The tensor read: the operand itself, or a packed copy when it had no layout cuBLAS reads.
  at::Tensor tensor;
  codegen::GemmOperand operand;
};

/// `tensor` as cuBLAS reads it. Packed rows are read as they are, and packed columns through the
/// transpose flag, which is how `nn.Linear`'s weight and the matrices autograd transposes arrive.
/// Anything else is packed on the device first.
///
/// A batch has to be packed matrices one after another, since the batched call takes one stride.
Matrix as_matrix(const at::Tensor& tensor, const std::string& name) {
  if (tensor.is_contiguous()) {
    return {.tensor = tensor,
            .operand = {.name = name, .transposed = false, .leading = tensor.size(-1)}};
  }
  if (tensor.transpose(-2, -1).is_contiguous()) {
    return {.tensor = tensor,
            .operand = {.name = name, .transposed = true, .leading = tensor.size(-2)}};
  }
  at::Tensor packed = tensor.contiguous();
  return {.tensor = packed,
          .operand = {.name = name, .transposed = false, .leading = packed.size(-1)}};
}

/// Whether every tensor is on one GPU, in a type the operation's ops.toml row lists, and the GPU's
/// CUDA ordinal if so.
std::optional<int> runs_on(const char* spelling, std::initializer_list<const at::Tensor*> tensors) {
  const passes::OpInfo* info = passes::find_builtin_op(spelling);
  const at::Tensor& first = **tensors.begin();
  const Result<ir::DType> type = to_ir_dtype(first.scalar_type());
  if (!codegen::blas_available() || info == nullptr || !info->has_kernel || !type.has_value() ||
      !info->accepts(*type)) {
    return std::nullopt;
  }
  for (const at::Tensor* tensor : tensors) {
    if (tensor->device() != first.device() || tensor->scalar_type() != first.scalar_type()) {
      return std::nullopt;
    }
  }
  return cuda_ordinal_of(first.device());
}

/// Whether cuBLAS can take these extents, which it reads as `int`.
bool fits(std::initializer_list<std::int64_t> extents) {
  for (const std::int64_t extent : extents) {
    if (extent <= 0 || extent > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  return true;
}

/// `lhs` times `rhs` into `out`, which already holds what the product is added to.
///
/// False, having launched nothing, when cuBLAS cannot take the call.
bool multiply(const at::Tensor& lhs, const at::Tensor& rhs, const at::Tensor& out, int ordinal) {
  const Matrix left = as_matrix(lhs, "lhs");
  const Matrix right = as_matrix(rhs, "rhs");
  codegen::GemmKernel kernel;
  kernel.batch = lhs.dim() == 3 ? lhs.size(0) : 1;
  kernel.m = lhs.size(-2);
  kernel.k = lhs.size(-1);
  kernel.n = rhs.size(-1);
  kernel.lhs = left.operand;
  kernel.rhs = right.operand;
  kernel.output = "out";
  kernel.dtype = ir::DType::kF32;
  if (!fits({kernel.batch,
             kernel.m,
             kernel.n,
             kernel.k,
             left.operand.leading,
             right.operand.leading})) {
    return false;
  }
  const codegen::GemmBuffers buffers{.lhs = left.tensor.const_data_ptr(),
                                     .rhs = right.tensor.const_data_ptr(),
                                     .output = out.data_ptr()};
  const Result<void> launched =
      codegen::launch_gemm(kernel, kernel.extents(), buffers, ordinal, codegen::kDefaultStream);
  TORCH_CHECK(launched.has_value(), launched.has_value() ? "" : launched.error().to_string());
  return true;
}

/// Zeroes `out`, which a plain multiply adds its product to.
void zero(const at::Tensor& out, int ordinal) {
  const Result<void> cleared =
      codegen::zero_fill(out.data_ptr(), out.nbytes(), ordinal, codegen::kDefaultStream);
  TORCH_CHECK(cleared.has_value(), cleared.has_value() ? "" : cleared.error().to_string());
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  const std::optional<int> ordinal = runs_on("aten.mm.default", {&self, &mat2});
  if (ordinal.has_value() && self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0) &&
      fits({self.size(0), self.size(1), mat2.size(1)})) {
    at::Tensor out = at::empty({self.size(0), mat2.size(1)}, self.options());
    zero(out, *ordinal);
    if (multiply(self, mat2, out, *ordinal)) {
      return out;
    }
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::mm>::call(self, mat2);
}

at::Tensor bmm(const at::Tensor& self, const at::Tensor& mat2) {
  const std::optional<int> ordinal = runs_on("aten.bmm.default", {&self, &mat2});
  if (ordinal.has_value() && self.dim() == 3 && mat2.dim() == 3 && self.size(0) == mat2.size(0) &&
      self.size(2) == mat2.size(1) &&
      fits({self.size(0), self.size(1), self.size(2), mat2.size(2)})) {
    at::Tensor out = at::empty({self.size(0), self.size(1), mat2.size(2)}, self.options());
    zero(out, *ordinal);
    if (multiply(self, mat2, out, *ordinal)) {
      return out;
    }
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::bmm>::call(self, mat2);
}

/// `beta * self + alpha * (mat1 @ mat2)`, for the beta and alpha of 1 `nn.Linear` uses. The bias
/// is laid into the result first, broadcast however torch would, and the product added to it.
at::Tensor addmm(const at::Tensor& self,
                 const at::Tensor& mat1,
                 const at::Tensor& mat2,
                 const at::Scalar& beta,
                 const at::Scalar& alpha) {
  const std::optional<int> ordinal = runs_on("aten.addmm.default", {&mat1, &mat2, &self});
  if (ordinal.has_value() && beta.equal(1) && alpha.equal(1) && mat1.dim() == 2 &&
      mat2.dim() == 2 && mat1.size(1) == mat2.size(0) &&
      fits({mat1.size(0), mat1.size(1), mat2.size(1)})) {
    const std::array<std::int64_t, 2> shape{mat1.size(0), mat2.size(1)};
    if (at::is_expandable_to(self.sizes(), shape)) {
      at::Tensor out = at::empty(shape, mat1.options());
      const Result<void> laid = pack_on_device(self.expand(shape), out);
      if (laid.has_value() && multiply(mat1, mat2, out, *ordinal)) {
        return out;
      }
    }
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::addmm>::call(
      self, mat1, mat2, beta, alpha);
}

} // namespace

Result<void> register_eager_matmuls(torch::Library& library) {
  try {
    library.impl("mm", &mm);
    library.impl("bmm", &bmm);
    library.impl("addmm", &addmm);
  } catch (const c10::Error& error) {
    return invalid_argument(error.what_without_backtrace());
  }
  return ok();
}

} // namespace tessiq::torch_backend
