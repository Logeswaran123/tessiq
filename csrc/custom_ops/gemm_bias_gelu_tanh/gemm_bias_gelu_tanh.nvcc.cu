// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The host code for tessiq::gemm_bias_gelu_tanh: one cuBLASLt matmul, compiled by nvcc when tessiq
// is built. It launches no kernel of its own.
//
// cuBLASLt reads column-major. The output, row-major [m, n], is its transpose read column-major, so
// the call computes out^T = weight x^T. The weight, row-major [n, k], is column-major [k, n] read
// transposed, and x, row-major [m, k], is column-major [k, m] read as it is. The GELU_BIAS epilogue
// adds one bias per row of that [n, m] result, which is one per column of the output, and applies
// the tanh GELU before writing.
//
// cuBLASLt takes the stream with each call rather than keeping one in its handle, so one handle
// serves every call on a device and there is no stream to set on it. Choosing an algorithm takes
// longer than running a small one, so the choice is kept for each shape and pointer alignment.

#include <cublasLt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "tessiq/codegen/precompiled.h"

namespace {

using tessiq::codegen::Call;
using tessiq::codegen::DeviceTensor;

/// The most workspace cuBLASLt may use. The algorithm it chooses says how much it needs, and only
/// that much is taken.
constexpr std::size_t kMostWorkspace = std::size_t{4} << 20;

/// Fails the call with `what` and cuBLAS's name for `status`, unless it is a success.
bool succeeded(Call& call, cublasStatus_t status, const char* what) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  call.fail(std::string(what) + ": " + cublasGetStatusName(status));
  return false;
}

/// The cuBLASLt handle for the call's device, created the first time and kept, since creating one
/// costs more than a multiply. Null, having failed the call, when it cannot be created.
cublasLtHandle_t handle_for(Call& call) {
  static std::mutex mutex;
  static std::map<int, cublasLtHandle_t> handles;
  const std::scoped_lock guard(mutex);
  if (const auto found = handles.find(call.device()); found != handles.end()) {
    return found->second;
  }
  cublasLtHandle_t handle = nullptr;
  if (!succeeded(call, cublasLtCreate(&handle), "cublasLtCreate")) {
    return nullptr;
  }
  handles.emplace(call.device(), handle);
  return handle;
}

/// The largest power of two, up to the 256 cuBLASLt assumes, that divides `pointer`'s address.
std::uint32_t alignment_of(const void* pointer) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  std::uint32_t alignment = 256;
  while (alignment > 1 && address % alignment != 0) {
    alignment /= 2;
  }
  return alignment;
}

/// A cuBLASLt descriptor, destroyed with `Destroy` when this goes.
template <typename Descriptor, cublasStatus_t (*Destroy)(Descriptor)>
class Owned {
 public:
  Owned() = default;
  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;

  ~Owned() {
    if (value != nullptr) {
      static_cast<void>(Destroy(value));
    }
  }

  Descriptor value = nullptr;
};

using MatmulDescription = Owned<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy>;
using Layout = Owned<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy>;
using Preference = Owned<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy>;

/// The device, m, n and k, and the alignment every pointer has.
using PlanKey = std::tuple<int, long long, long long, long long, std::uint32_t>;

/// The algorithm cuBLASLt chose, and the workspace it needs.
struct Plan {
  cublasLtMatmulAlgo_t algorithm{};
  std::size_t workspace = 0;
};

/// The matmul as `run` describes it to cuBLASLt.
struct Matmul {
  MatmulDescription description;
  Layout weight;
  Layout x;
  Layout out;
};

/// Describes gelu(weight x^T + bias) for cuBLASLt. False, having failed the call, when it cannot.
bool describe(
    Call& call, Matmul& matmul, long long m, long long n, long long k, const float* bias) {
  const cublasOperation_t transposed = CUBLAS_OP_T;
  const cublasOperation_t as_is = CUBLAS_OP_N;
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
  // Float32 accumulation and scaling, as cuBLAS's SGEMM does, rather than TF32.
  return succeeded(
             call,
             cublasLtMatmulDescCreate(&matmul.description.value, CUBLAS_COMPUTE_32F, CUDA_R_32F),
             "cublasLtMatmulDescCreate") &&
         succeeded(call,
                   cublasLtMatmulDescSetAttribute(matmul.description.value,
                                                  CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &transposed,
                                                  sizeof(transposed)),
                   "setting TRANSA") &&
         succeeded(
             call,
             cublasLtMatmulDescSetAttribute(
                 matmul.description.value, CUBLASLT_MATMUL_DESC_TRANSB, &as_is, sizeof(as_is)),
             "setting TRANSB") &&
         succeeded(call,
                   cublasLtMatmulDescSetAttribute(matmul.description.value,
                                                  CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue,
                                                  sizeof(epilogue)),
                   "setting the epilogue") &&
         succeeded(
             call,
             cublasLtMatmulDescSetAttribute(
                 matmul.description.value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)),
             "setting the bias") &&
         succeeded(call,
                   cublasLtMatrixLayoutCreate(&matmul.weight.value, CUDA_R_32F, k, n, k),
                   "the weight's layout") &&
         succeeded(call,
                   cublasLtMatrixLayoutCreate(&matmul.x.value, CUDA_R_32F, k, m, k),
                   "x's layout") &&
         succeeded(call,
                   cublasLtMatrixLayoutCreate(&matmul.out.value, CUDA_R_32F, n, m, n),
                   "the output's layout");
}

/// The algorithm for `matmul`, chosen once for each key and kept. False, having declined or failed
/// the call, when there is none.
bool plan_for(
    Call& call, cublasLtHandle_t handle, const Matmul& matmul, const PlanKey& key, Plan& plan) {
  static std::mutex mutex;
  static std::map<PlanKey, Plan> plans;
  const std::scoped_lock guard(mutex);
  if (const auto found = plans.find(key); found != plans.end()) {
    plan = found->second;
    return true;
  }

  Preference preference;
  if (!succeeded(call, cublasLtMatmulPreferenceCreate(&preference.value), "the preference")) {
    return false;
  }
  const std::uint32_t alignment = std::get<4>(key);
  const std::size_t most = kMostWorkspace;
  // cuBLASLt otherwise assumes 256-byte pointers, and a tensor viewed at an offset has fewer.
  for (const cublasLtMatmulPreferenceAttributes_t attribute :
       {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    if (!succeeded(call,
                   cublasLtMatmulPreferenceSetAttribute(
                       preference.value, attribute, &alignment, sizeof(alignment)),
                   "setting an alignment")) {
      return false;
    }
  }
  if (!succeeded(
          call,
          cublasLtMatmulPreferenceSetAttribute(
              preference.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &most, sizeof(most)),
          "setting the workspace")) {
    return false;
  }

  cublasLtMatmulHeuristicResult_t result{};
  int found = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle,
                                                               matmul.description.value,
                                                               matmul.weight.value,
                                                               matmul.x.value,
                                                               matmul.out.value,
                                                               matmul.out.value,
                                                               preference.value,
                                                               1,
                                                               &result,
                                                               &found);
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || (status == CUBLAS_STATUS_SUCCESS && found == 0)) {
    call.decline("cuBLASLt has no algorithm for these extents and alignments");
    return false;
  }
  if (!succeeded(call, status, "cublasLtMatmulAlgoGetHeuristic")) {
    return false;
  }
  plan = Plan{result.algo, result.workspaceSize};
  plans.emplace(key, plan);
  return true;
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

  const cublasLtHandle_t handle = handle_for(call);
  if (handle == nullptr) {
    return;
  }
  Matmul matmul;
  if (!describe(call, matmul, m, n, k, bias.as<const float>())) {
    return;
  }
  std::uint32_t alignment = alignment_of(x.data);
  for (const void* pointer : {weight.data, bias.data, call.output().data}) {
    alignment = std::min(alignment, alignment_of(pointer));
  }
  Plan plan;
  if (!plan_for(call, handle, matmul, {call.device(), m, n, k, alignment}, plan)) {
    return;
  }
  void* workspace = nullptr;
  if (plan.workspace > 0) {
    workspace = call.workspace(plan.workspace);
    if (workspace == nullptr) {
      return;
    }
  }

  const float one = 1.0F;
  const float zero = 0.0F;
  succeeded(call,
            cublasLtMatmul(handle,
                           matmul.description.value,
                           &one,
                           weight.data,
                           matmul.weight.value,
                           x.data,
                           matmul.x.value,
                           &zero,
                           call.output().data,
                           matmul.out.value,
                           call.output().data,
                           matmul.out.value,
                           &plan.algorithm,
                           workspace,
                           plan.workspace,
                           static_cast<cudaStream_t>(call.stream())),
            "cublasLtMatmul");
}

} // namespace

TESSIQ_PRECOMPILED(gemm_bias_gelu_tanh, run)
