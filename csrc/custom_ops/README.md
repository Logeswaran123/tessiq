# Custom operations

Each directory here is one operation. It is compiled into the extension, registered on
`import tessiq`, and callable as `torch.ops.tessiq.<name>`, eagerly on tessiq tensors and inside
`torch.compile`. Adding one is a new directory and a rebuild; nothing else in tessiq changes.

```
custom_ops/
    CMakeLists.txt                      picks up every directory; never needs editing
    rms_norm/
        rms_norm.cpp                    schema, reference, launch function
        rms_norm.cu                     kernel, compiled at run time with NVRTC
        rms_norm.pattern.tsr            RMSNorm written out, for its rewrite
    softmax/                            .cu kernel; replaces aten::_softmax
    fast_gelu/                          elementwise expression, plus a pattern
    swiglu/                             elementwise expression with two operands
    layer_norm/
        layer_norm.cpp                  declared .precompiled()
        layer_norm.nvcc.cu              host code, compiled by nvcc at build time
    gemm_bias_gelu/                     CUTLASS host code, plus a pattern
    gemm_bias_gelu_tanh/                cuBLASLt host code, plus a pattern
```

An operation's kernel is one of:

| Kind | Declared with | Built |
|---|---|---|
| CUDA kernel in `<name>.cu` | `.launch(fn)` | NVRTC, on first use on each GPU |
| Elementwise expression | `.expression(text)` | Fused into the compiler's generated loops |
| Host code in `<name>.nvcc.cu` | `.precompiled()` | nvcc, at build time |

## A CUDA kernel

Take `scale_rows`, which multiplies each row of a matrix by its own factor.

`scale_rows/scale_rows.cu` holds an `extern "C" __global__` function named after the operation.
Its parameters are a pointer per tensor argument in schema order, then the result, then the launch
arguments:

```cuda
extern "C" __global__ void scale_rows(const float* x, const float* factors, float* out,
                                      long long columns) {
  const long long row = blockIdx.x;
  for (long long c = threadIdx.x; c < columns; c += blockDim.x) {
    out[row * columns + c] = x[row * columns + c] * factors[row];
  }
}
```

Tensors arrive packed, so index from the first element. The file is compiled on its own and can't
`#include` other project files.

`scale_rows/scale_rows.cpp` gives the schema, the ATen reference, and how to launch:

```cpp
#include <ATen/ATen.h>
#include <cstdint>

#include "tessiq/torch/custom_op.h"

namespace {

using tessiq::custom_ops::Launch;

at::Tensor scale_rows(const at::Tensor& x, const at::Tensor& factors) {
  return x * factors.unsqueeze(-1);
}

// One 256-thread block per row.
Launch launch(const at::Tensor& x, const at::Tensor& /*factors*/, const at::Tensor& /*out*/) {
  Launch launch;
  launch.grid = {static_cast<std::uint32_t>(x.size(0)), 1, 1};
  launch.block = {256, 1, 1};
  launch.arguments = {x.size(1)};
  return launch;
}

} // namespace

TESSIQ_CUSTOM_OP(scale_rows, "(Tensor x, Tensor factors) -> Tensor")
    .reference(scale_rows)
    .launch(launch);
```

The kernel is found by name. Use `.kernel("other.cu", "entry")` if it lives elsewhere. After
`pip install -e . --no-build-isolation`, `tessiq.library.check("tessiq::scale_rows")` compiles the
kernel and raises with the NVRTC log on failure. The test suite checks every operation this way.

Launch arguments are typed: each entry must have the C++ type of the kernel parameter it fills
(`std::int64_t` for `long long`, `std::int32_t` for `int`, `float`, `double`). Pass
`static_cast<float>(eps)` for a `float` parameter.

## An elementwise expression

An operation that computes each output element from the inputs at the same position needs no
kernel. Its expression is fused into the surrounding loop:

```cpp
at::Tensor fast_gelu(const at::Tensor& x) { return x * (1.702 * x).sigmoid(); }

TESSIQ_CUSTOM_OP(fast_gelu, "(Tensor x) -> Tensor")
    .reference(fast_gelu)
    .expression("$0 * (scalar_t(1) / (scalar_t(1) + exp(scalar_t(-1.702) * $0)))")
    .dtypes({at::kFloat, at::kDouble});
```

`$0`, `$1` are the operands. Write constants as `scalar_t(1.702)`; a bare `1.702` is a double and
would promote float32 math, so registration rejects it. All arguments are tensors and broadcast
like `torch.add`'s.

## Host code

NVRTC only compiles device code. An operation that needs host code (several launches, workspace
between them, a library call, headers NVRTC lacks) declares `.precompiled()` and puts the code in
`<name>.nvcc.cu`, which nvcc compiles at build time:

```cpp
TESSIQ_CUSTOM_OP(layer_norm, "(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor")
    .reference(layer_norm)
    .precompiled()
    .dtypes({at::kFloat});
```

```cuda
#include "tessiq/codegen/precompiled.h"

namespace {

void run(tessiq::codegen::Call& call) {
  const tessiq::codegen::DeviceTensor& x = call.input(0);
  ...
  if (weight.numel() != columns || bias.numel() != columns) {
    call.decline("the kernel reads one weight and one bias per column");
    return;
  }
  auto* statistics = static_cast<float2*>(call.workspace(rows * sizeof(float2)));
  if (statistics == nullptr) {
    return;
  }
  const auto stream = static_cast<cudaStream_t>(call.stream());
  row_statistics<<<rows, kThreads, 0, stream>>>(x.as<const float>(), statistics, columns, eps);
  apply<<<blocks, kThreads, 0, stream>>>(...);
}

} // namespace

TESSIQ_PRECOMPILED(layer_norm, run)
```

`precompiled.h` has no PyTorch dependency. The `Call` provides:

| | |
|---|---|
| `input(i)`, `output()` | Tensor arguments (counted separately from scalars) and the result: packed, on the call's device |
| `real(i)`, `integer(i)`, `flag(i)` | `float`, `int` and `bool` arguments, counted separately from tensors |
| `stream()` | The `cudaStream_t` all work must go on |
| `device()` | The CUDA device, which is current when `run` is called |
| `workspace(bytes)` | Memory from tessiq's allocator, valid until the queued work finishes. Null, with the call failed, if unavailable |
| `decline(reason)` | Run the reference instead |
| `fail(reason)` | Fail the call |

A launch error left after `run` returns fails the call with CUDA's message.

Precompiled code is built for `CMAKE_CUDA_ARCHITECTURES`, `native` by default. On a GPU it wasn't
built for, the operation is refused before it runs, eagerly, from `tessiq.library.check()` and in
`torch.compile`:

```
RuntimeError: external: tessiq::layer_norm was built for sm_89, and CUDA device 0 is sm_120; rebuild tessiq with CMAKE_CUDA_ARCHITECTURES including 120
```

An architecture listed without `-real` also embeds PTX, which the driver can JIT for newer GPUs.
Without nvcc, `.nvcc.cu` files are skipped and their operations always run the reference.
`tessiq.library.precompiled()` lists the ones that were built.

**CUTLASS.** A `.nvcc.cu` that includes a `cutlass/` header gets CUTLASS 4.8.0 (header-only)
downloaded at configure time. Offline, point the build at a checkout with
`-Ccmake.define.FETCHCONTENT_SOURCE_DIR_CUTLASS=/path/to/cutlass`. `gemm_bias_gelu` uses it: a
float32 SIMT GEMM whose epilogue applies bias and GELU before writing. It splits K across slices,
synchronised by per-tile semaphores in workspace, when the output has too few tiles to fill the
GPU.

**Libraries.** A `.nvcc.cu` that includes `cublas_v2.h` or `cublasLt.h` is linked against cuBLAS
and cuBLASLt. `gemm_bias_gelu_tanh` is a single `cublasLtMatmul` with the GELU_BIAS epilogue:
- It passes `call.stream()` to each call. A cuBLAS handle holds a stream, so set it before each
  call.
- It takes the workspace cuBLASLt's chosen algorithm asks for from `call.workspace()`.
- It keeps one handle per device.
- It caches the chosen algorithm by shape and pointer alignment. cuBLASLt assumes 256-byte
  alignment unless told otherwise, and a tensor at an offset has less.

## Replacing a PyTorch operation

`.replaces()` makes a custom operation the implementation of a PyTorch one, so models get it
without calling it by name. `softmax` replaces `aten::_softmax`, which `torch.softmax`,
`F.softmax` and `nn.Softmax` all reach:

```cpp
at::Tensor softmax(const at::Tensor& self, std::int64_t dim, bool half_to_float) {
  return at::_softmax(self, dim, half_to_float);
}

TESSIQ_CUSTOM_OP(softmax, "(Tensor self, int dim, bool half_to_float) -> Tensor")
    .reference(softmax)
    .launch(launch)
    .dtypes({at::kFloat})
    .replaces("aten::_softmax");
```

The schema must match the PyTorch operation's argument names and types. The replacement applies
in three places:

- **Eager.** On tessiq devices it is the operation's implementation. For undeclared types it runs
  the PyTorch op on the host and counts a fallback.
- **Compiled graphs.** An `override-softmax` pass turns `aten._softmax.default` nodes into
  `tessiq.softmax.default` wherever the types are declared.
- **Decomposition.** tessiq's decomposition table keeps `_softmax` whole where the custom
  operation applies, and decomposes it elsewhere.

The PyTorch operation keeps its backward, so training works. An op PyTorch decomposes before any
backend sees it, such as `aten::softmax.int`, can't be replaced; replace what it decomposes into.
`tessiq.library.overrides()` lists the replacements.

## Rewriting a pattern

Models usually write RMSNorm out rather than calling `tessiq.rms_norm`:

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

`rms_norm.pattern.tsr` is that expression as AOTAutograd emits it, in textual IR, and `.pattern()`
registers a `rewrite-rms-norm` pass that replaces each match with one call:

```
graph @rms_norm(%x: f32[?], %weight: f32[?]) -> f32[?] {
  %pow = aten.pow.Tensor_Scalar(%x) {arg1 = 2} : f32[?]
  %mean = aten.mean.dim(%pow) {arg1 = [-1], arg2 = true} : f32[?]
  %add = aten.add.Tensor(%mean) {arg1 = "$eps"} : f32[?]
  %rsqrt = aten.rsqrt.default(%add) : f32[?]
  %normed = aten.mul.Tensor(%x, %rsqrt) : f32[?]
  %out = aten.mul.Tensor(%normed, %weight) : f32[?]
  return %out
}
```

Rules for patterns:

- **Inputs** are named after the operation's tensor arguments.
- **Scalars.** `"$name"` captures a scalar argument.
- **Attributes.** Every other attribute must match exactly (`2` equals `2.0`), and a node may not
  carry attributes the pattern lacks.
- **Types** aren't compared. A match with undeclared dtypes is left alone.
- **Operand order.** `add` and `mul` operands match in either order.
- **Write what AOTAutograd emits.** `tessiq.import_graph(tessiq.capture(model, x))` shows it; for
  example `1.702 * x` arrives as `mul(x) {arg1 = 1.702}`.

A match is refused if any interior value is used outside it or returned. Training forward graphs
return the values backward needs, so rewrites don't fire there.

Rewrites change numerics, so they only run when requested:

```python
torch.compile(model, backend="tessiq", options={"device": "tessiq:1", "rewrites": ["rms_norm"]})
tessiq.explain(model, x, device=tessiq.gpu, rewrites=["rms_norm"])
```

[docs/passes.md](../../docs/passes.md) shows where rewrites and overrides run in the pipeline.

## Behaviour

- **Argument types.** Schema `Tensor`, `float`, `int`, `bool` map to `const at::Tensor&`,
  `double`, `std::int64_t`, `bool` in the reference and launch function.
- **Dtypes.** `.dtypes()` lists the types the kernel handles, float32 by default. Other types run
  the reference, and a compiled graph keeps them on the host.
- **Output shape and type** come from running the reference on meta tensors.
- **Declining.** A launch function can decline with `TORCH_CHECK`, and host code with
  `call.decline()`; the reference runs instead. `rms_norm`, `layer_norm` and the GEMMs decline a
  weight or bias that isn't one value per column.
- **Limitations.** Schemas can't have defaults (a compiled graph drops arguments equal to their
  default). Operations have no backward.

| Called on | Runs |
|---|---|
| tessiq GPU tensor, declared dtype | The kernel, expression or host code |
| tessiq GPU tensor, other dtype | The reference, on the host |
| `tessiq.cpu` or a plain CPU tensor | The reference |
| `torch.compile` for a GPU | The kernel, alone or fused with its neighbours |

`tessiq.library.fallbacks("tessiq::scale_rows")` counts calls on tessiq tensors that ran the
reference instead of the kernel.

## Errors

Definitions are checked after the extension loads, and the first bad one fails `import tessiq`:

```
ImportError: invalid_argument: custom operation tessiq::rms_norm in csrc/custom_ops/rms_norm: its reference takes (Tensor, Tensor, float) but the schema declares (Tensor, Tensor, int)
ImportError: invalid_argument: custom operation tessiq::softmax in csrc/custom_ops/softmax: it replaces aten::softmax.int, which torch breaks into other operations before any backend sees it; replace one of those instead
ImportError: invalid_argument: custom operation tessiq::rms_norm in csrc/custom_ops/rms_norm: its pattern csrc/custom_ops/rms_norm/rms_norm.pattern.tsr has the inputs (x, w), and its tensor arguments are (x, weight); they have to be the same names
ImportError: invalid_argument: a .nvcc.cu file registers host code for tessiq::layer_norm, which no .cpp defines with .precompiled()
```

The same applies to a missing kernel file, a launch function whose parameters don't match the
schema, a schema that doesn't parse, more or fewer than one of `.launch()`, `.expression()` and
`.precompiled()`, and a replacement of an operation PyTorch doesn't have.
