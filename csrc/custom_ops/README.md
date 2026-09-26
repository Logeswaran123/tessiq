# Custom operations

Each directory here is one operation that ships with tessiq. It is compiled into the extension,
registered when `tessiq` is imported, and called as `torch.ops.tessiq.<name>`. It runs on the GPU
both called directly and inside `torch.compile`.

```
custom_ops/
    CMakeLists.txt              picks up every directory below; never needs editing
    rms_norm/
        rms_norm.cu             the kernel
        rms_norm.cpp            the operation: its schema, its reference, how to launch the kernel
        rms_norm.pattern.tsr    RMSNorm written out, which its rewrite replaces with one call
    softmax/
        softmax.cu              the kernel
        softmax.cpp             the same, and it replaces torch's aten::_softmax
    layer_norm/
        layer_norm.nvcc.cu      host code and two kernels, which nvcc compiles when tessiq is built
        layer_norm.cpp          the operation, declared .precompiled()
    gemm_bias_gelu/
        gemm_bias_gelu.nvcc.cu        a CUTLASS GEMM whose epilogue adds the bias and applies GELU
        gemm_bias_gelu.cpp            the operation, declared .precompiled()
        gemm_bias_gelu.pattern.tsr    a linear layer followed by GELU, which its rewrite replaces
    fast_gelu/
        fast_gelu.cpp           an elementwise expression the compiler fuses, so no kernel file
        fast_gelu.pattern.tsr   the sigmoid GELU written out
    swiglu/
        swiglu.cpp              an expression with two operands
```

Adding an operation is a new directory with those files, and a rebuild. Nothing else in tessiq is
edited.

## Adding a kernel

Take a hypothetical `scale_rows`, which multiplies each row of a matrix by its own factor.

**1. Make a directory named after the operation.** Create `custom_ops/scale_rows/`.

**2. Write the kernel in `scale_rows.cu`.** Declare it `extern "C" __global__` and give it the
operation's name. Its parameters come in a fixed order: a pointer to each tensor argument in schema
order, then a pointer to the result, then the launch arguments.

```cuda
extern "C" __global__ void scale_rows(const float* x, const float* factors, float* out,
                                      long long columns) {
  const long long row = blockIdx.x;
  for (long long c = threadIdx.x; c < columns; c += blockDim.x) {
    out[row * columns + c] = x[row * columns + c] * factors[row];
  }
}
```

Every tensor the kernel receives is packed, so it can index each one from its first element. The
file is compiled on its own with NVRTC, for whichever GPU it runs on, the first time it runs there.
It cannot `#include` other files of yours.

**3. Define the operation in `scale_rows.cpp`.** The reference says what the operation computes, in
ATen. The launch function takes the same arguments followed by the result, and says how to launch
the kernel over them:

```cpp
#include <ATen/ATen.h>
#include <cstdint>

#include "tessiq/torch/custom_op.h"

namespace {

using tessiq::custom_ops::Launch;

// What the kernel computes, in ATen.
at::Tensor scale_rows(const at::Tensor& x, const at::Tensor& factors) {
  return x * factors.unsqueeze(-1);
}

// One block of 256 threads per row.
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

That is the whole definition. The kernel is found by name: `scale_rows.cu` in the same directory,
entry point `scale_rows`. If yours is named differently, add `.kernel("other.cu", "other_name")`.

**4. Rebuild and check.** `pip install -e . --no-build-isolation` picks up the new directory. Then
`torch.ops.tessiq.scale_rows` exists, and `tessiq.library.check("tessiq::scale_rows")` compiles the
kernel and raises with the compiler's log if it does not compile. The test suite compiles every
operation here, so a broken kernel fails there rather than in someone's model.

## Adding an elementwise operation

An operation that computes each element of its result from the elements at the same position, such
as an activation function, needs no kernel. Give an expression instead, and the compiler writes it
into the same loop as the operations around it:

```cpp
at::Tensor fast_gelu(const at::Tensor& x) { return x * (1.702 * x).sigmoid(); }

TESSIQ_CUSTOM_OP(fast_gelu, "(Tensor x) -> Tensor")
    .reference(fast_gelu)
    .expression("$0 * (scalar_t(1) / (scalar_t(1) + exp(scalar_t(-1.702) * $0)))")
    .dtypes({at::kFloat, at::kDouble});
```

`$0` is the first operand's element and `$1` the second's. Write constants as `scalar_t(1.702)`,
never as a bare `1.702`, which C++ reads as a double and which turns a float32 kernel into double
arithmetic. The registration refuses the bare form. Every argument of such an operation is a
tensor, and they broadcast against each other like the operands of torch's own `add`.

## Adding host code

A `.cu` kernel is compiled at run time with NVRTC, which compiles device code only. An operation that
needs host code puts it in `<name>.nvcc.cu` instead, which nvcc compiles when tessiq is built. Host
code can launch several kernels, keep workspace between them, call a library, or use a header NVRTC
does not have. `layer_norm` launches one kernel for each row's statistics and another to apply them,
with the statistics in workspace between the two.

**1. Declare it** in `layer_norm.cpp`, with `.precompiled()` where a kernel would have `.launch()`:

```cpp
TESSIQ_CUSTOM_OP(layer_norm, "(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor")
    .reference(layer_norm)
    .precompiled()
    .dtypes({at::kFloat});
```

**2. Write the host code** in `layer_norm.nvcc.cu`. It includes `tessiq/codegen/precompiled.h`, which
has no torch in it, and registers one function with `TESSIQ_PRECOMPILED`:

```cuda
#include "tessiq/codegen/precompiled.h"

namespace {

void run(tessiq::codegen::Call& call) {
  const tessiq::codegen::DeviceTensor& x = call.input(0);
  const tessiq::codegen::DeviceTensor& weight = call.input(1);
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

The `Call` holds:

- **Tensors:** `input(i)`, counting only the tensor arguments, and the result, `output()`. Each is
  packed and on the call's device, as a data pointer, its sizes and an element type.
- **Scalars:** `real(i)`, `integer(i)` and `flag(i)` for float, int and bool, counting only the
  scalar arguments.
- **The stream:** `stream()`, a `cudaStream_t`. Every launch and library call goes on it; work
  queued anywhere else can run before its inputs exist.
- **Workspace:** `workspace(bytes)`, from tessiq's allocator on the call's device, so the memory
  statistics count it. It stays valid until the work the call queued has finished. It is null,
  having failed the call, when the device has none to give.
- **Declining:** `decline(reason)` hands the call to the reference, as `TORCH_CHECK` does in a
  launch function.

The call's device is current when the function runs, and a launch that fails fails the call with
CUDA's message.

**3. Rebuild.** nvcc compiles the file for `CMAKE_CUDA_ARCHITECTURES`, which is the GPUs in the
machine unless the build is told otherwise:

```
pip install -e . --no-build-isolation
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="89;120"
```

On a card it was not built for, the operation is refused before it runs, with both architectures
named. That happens called directly, from `tessiq.library.check()`, and in `torch.compile`:

```
RuntimeError: external: tessiq::layer_norm was built for sm_89, and CUDA device 0 is sm_120; rebuild tessiq with CMAKE_CUDA_ARCHITECTURES including 120
```

The check asks the CUDA runtime whether it can load the file on the card. An architecture listed
without `-real` also carries PTX, which the driver can compile for a newer card, and that passes.

`tessiq.library.precompiled()` lists the operations whose host code the build compiled. A build
without nvcc leaves the `.nvcc.cu` files out, and their operations run the reference everywhere.

### Using CUTLASS

A `.nvcc.cu` that includes a header from `cutlass/` gets CUTLASS 4.8.0, which is header-only,
downloaded when tessiq is configured. A build with no such file never downloads it. On a machine
that cannot download, point the build at a checkout:

```
pip install -e . --no-build-isolation -Ccmake.define.FETCHCONTENT_SOURCE_DIR_CUTLASS=/path/to/cutlass
```

`gemm_bias_gelu` is one. It computes `gelu(x @ weight.T + bias)` with one CUTLASS GEMM whose
epilogue, the stage that turns each block's accumulators into output elements, adds the bias and
applies GELU before writing. Without it a linear layer followed by GELU writes the product and a
second kernel reads it back. Its pattern is that pair as `nn.Linear` and `F.gelu` emit it, so
`rewrites=["gemm_bias_gelu"]` puts it in a model that never names it.

It uses CUTLASS's float32 SIMT kernel rather than TF32 tensor cores, so it rounds like cuBLAS's
SGEMM, which tessiq's matmuls call. When the output has too few tiles to fill the GPU, K is split
into slices that take turns adding into the output, through a semaphore per tile in workspace.

## Replacing a torch operation

An operation can stand in for one of torch's own. Then a model does not have to call it by name:
it calls the torch operation, and the custom one runs wherever it declares the types. `softmax`
replaces `aten::_softmax`, which `torch.softmax`, `F.softmax` and `nn.Softmax` all reach:

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

The schema has to take the same arguments as the torch operation's, with the same names and types,
and the reference is usually the torch operation itself. The replacement happens in three places:

- **Called on a tessiq tensor**, the custom operation is the torch operation's implementation on
  tessiq's devices. For a type it does not declare, it runs the torch operation on the host, and
  counts that in `fallbacks()`.
- **In a compiled graph**, a pass named `override-softmax` runs first and makes each
  `aten._softmax.default` node a `tessiq.softmax.default` node, where every tensor has a declared
  type. `tessiq.explain` shows it in the pass-by-pass section:

  ```
  override-softmax: 1 line(s) rewritten
    -  %_softmax = aten._softmax.default(%mul) {arg1 = -1, arg2 = false} : f32[8,128]
    +  %_softmax = tessiq.softmax.default(%mul) {arg1 = -1, arg2 = false} : f32[8,128]
  ```

- **Before the graph reaches tessiq**, torch would break `_softmax` into five smaller operations.
  tessiq keeps it whole where the custom operation declares the types, so the pass has something
  to find, and breaks it up as before everywhere else.

Training still works through it. The torch operation keeps its own backward, which reads the
result the custom kernel computed.

An operation torch breaks up as soon as it is called, such as `aten::softmax.int`, cannot be
replaced, because no backend ever sees it. Replace the one it becomes, as `softmax` does.
`tessiq.library.overrides()` lists what replaces what.

## Rewriting a computation written out in PyTorch

Most models do not call `torch.ops.tessiq.rms_norm`. They write RMSNorm out:

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

A pattern lets the operation find that and take its place. `rms_norm.pattern.tsr` is the graph
AOTAutograd hands tessiq for that line, in tessiq's textual IR:

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

and `.pattern()` in `rms_norm.cpp` declares it. The operation then has a pass, `rewrite-rms-norm`,
that replaces each match with one call. The rules for writing one:

- **Inputs are the operation's tensor arguments,** by name. `%x` and `%weight` above are
  `rms_norm`'s `x` and `weight`.
- **`"$name"` captures a scalar argument** from wherever the pattern has it, here `eps` from the
  `add`. Every other attribute has to match exactly, except that `2` and `2.0` are the same number.
- **Types are not compared,** so write them as anything, such as `f32[?]`. A match whose tensors
  are of a type the operation does not declare is left alone.
- **Write what AOTAutograd emits, not what the model says.** `tessiq.import_graph(tessiq.capture(
  model, x))` prints it. `1.702 * x`, for instance, arrives as `mul(x) {arg1 = 1.702}`. The two
  operands of `add` and `mul` match either way round.

A match is refused when a value inside it is read by anything outside it or returned by the graph,
because the call replaces those values. A training step's forward graph returns the values its
backward graph reads, so there the rewrite finds nothing, and training runs the operations as they
were written.

A rewrite changes which operations run, so it runs only when a compile asks for it:

```python
torch.compile(model, backend="tessiq", options={"device": "tessiq:1", "rewrites": ["rms_norm"]})
tessiq.explain(model, x, device=tessiq.gpu, rewrites=["rms_norm"])
```

`tessiq.library.rewrites()` lists what has one. [docs/passes.md](../../docs/passes.md) shows where
rewrites and overrides sit among the compiler's other passes.

## What the definition decides

- **Types.** A schema's `Tensor`, `float`, `int` and `bool` are `const at::Tensor&`, `double`,
  `std::int64_t` and `bool` in the reference and the launch function, which is torch's own mapping.
- **Launch arguments are typed.** Each entry of `Launch::arguments` has the C++ type of the kernel
  parameter it fills: `std::int64_t` for `long long`, `std::int32_t` for `int`, `float` and
  `double`. The kernel reads the bytes as its signature says, so `static_cast<float>(eps)` for a
  `float eps` is not optional.
- **Element types.** `.dtypes()` lists the types the kernel was written for, float32 when not
  given. On any other type the operation runs its reference instead, and a compiled graph keeps it
  on the host.
- **The result's shape and type** are the reference's. They are found by running the reference on
  tensors that have a shape and no data, so there is nothing else to write for them.
- **Declining a call.** A launch function can decline arguments its kernel cannot handle with
  `TORCH_CHECK`, and host code with `call.decline()`. The call runs the reference instead, counted
  in `fallbacks()`. `rms_norm` and `layer_norm` do this for a weight that is not one value per
  column, which the reference would broadcast and the kernel would read past the end of, and
  `gemm_bias_gelu` for a bias that is not one value per column.
- **No defaults, no backward.** A schema argument cannot have a default yet, because a compiled
  graph leaves out an argument equal to its default. The operation has no backward, so it is for
  inference until one is registered.

## Where it runs

| called on | runs |
|---|---|
| a tessiq GPU tensor, in a declared type | the kernel, expression or host code |
| a tessiq GPU tensor, in another type | the reference, on the host |
| the CPU reference device, `tessiq.cpu` | the reference |
| an ordinary CPU tensor | the reference |
| inside `torch.compile` for a GPU | a kernel, its own launch or fused with its neighbours |

`tessiq.library.fallbacks("tessiq::scale_rows")` counts the calls on a tessiq tensor that ran the
reference instead of the kernel. The answer is the same either way, so this count is the only sign
that a model is taking the slow path.

## When a definition is wrong

Loading the extension only records each definition. They are checked and registered once it has
loaded, and the first one that does not hold together fails the import of `tessiq` with its name
and what is wrong:

```
ImportError: invalid_argument: custom operation tessiq::rms_norm in csrc/custom_ops/rms_norm: its reference takes (Tensor, Tensor, float) but the schema declares (Tensor, Tensor, int)
ImportError: invalid_argument: custom operation tessiq::rms_norm in csrc/custom_ops/rms_norm: there is no kernel file csrc/custom_ops/rms_norm/rms.cu
```

A replacement is checked the same way:

```
ImportError: invalid_argument: custom operation tessiq::softmax in csrc/custom_ops/softmax: it replaces aten::softmax.int, which torch breaks into other operations before any backend sees it; replace one of those instead
ImportError: invalid_argument: custom operation tessiq::softmax in csrc/custom_ops/softmax: it replaces aten::_softmax, whose schema is aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor; the two have to take the same arguments, with the same names and types, and no defaults
```

A pattern is checked against the schema:

```
ImportError: invalid_argument: custom operation tessiq::rms_norm in csrc/custom_ops/rms_norm: its pattern csrc/custom_ops/rms_norm/rms_norm.pattern.tsr has the inputs (x, w), and its tensor arguments are (x, weight); they have to be the same names
```

So is host code that no definition declares:

```
ImportError: invalid_argument: a .nvcc.cu file registers host code for tessiq::layer_norm, which no .cpp defines with .precompiled()
```

The same goes for a launch function whose parameters differ from the schema, a schema that does not
parse, an operation given more than one of a launch function, an expression and `.precompiled()`,
or none of them, and a replacement of an operation torch does not have.
