# Custom operations

Each directory here is one operation that ships with tessiq. It is compiled into the extension,
registered when `tessiq` is imported, and called as `torch.ops.tessiq.<name>`. It runs on the GPU
both called directly and inside `torch.compile`.

```
custom_ops/
    CMakeLists.txt      picks up every directory below; never needs editing
    rms_norm/
        rms_norm.cu     the kernel
        rms_norm.cpp    the operation: its schema, its reference, and how to launch the kernel
    softmax/
        softmax.cu      the kernel
        softmax.cpp     the same, and it replaces torch's aten::_softmax
    fast_gelu/
        fast_gelu.cpp   an elementwise expression the compiler fuses, so no kernel file
    swiglu/
        swiglu.cpp      the same, with two operands
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
  `TORCH_CHECK`, and the call runs the reference instead, counted in `fallbacks()`. `rms_norm` does
  this for a weight that is not one value per column, which the reference would broadcast and the
  kernel would read past the end of.
- **No defaults, no backward.** A schema argument cannot have a default yet, because a compiled
  graph leaves out an argument equal to its default. The operation has no backward, so it is for
  inference until one is registered.

## Where it runs

| called on | runs |
|---|---|
| a tessiq GPU tensor, in a declared type | the kernel or expression |
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

The same goes for a launch function whose parameters differ from the schema, a schema that does not
parse, an operation given both a kernel and an expression, or neither, and a replacement of an
operation torch does not have.
