# Roadmap

| Stage | Scope | Status |
|---|---|---|
| 0 | Build, packaging, CI gates | Done |
| 1 | Device runtime, eager execution | Done |
| 2 | CUDA device | Done |
| 3 | IR and pass infrastructure | Done |
| 4 | Graph capture, partitioning | Done |
| 5 | Shape inference, memory planning, recipe cache | Done |
| 6 | Elementwise fusion, CUDA codegen | Done |
| 7 | Op coverage, model blocks | Done |
| 8 | Training, dynamic shapes, tooling | Done |
| 9 | Device-side copies | Done |
| 10 | Performance analysis | Done |
| 11 | Convolution | Done |
| 12 | Custom operations, rewrite passes | Done |
| 13 | Precompiled custom operations | Done |
| 14 | Eager elementwise operations on the device | Done |
| 15 | Eager reductions, matmuls and convolution on the device | Done |
| 16 | MLIR export | Planned, optional |

## Done

**0. Build, packaging, CI gates.** CMake, scikit-build-core, pybind11, GoogleTest and pytest. CI
enforces `-Werror`, clang-format, clang-tidy, ruff, `mypy --strict`, ASan/UBSan and conventional
commits, with tooling pinned in `requirements-dev.txt`.

**1. Device runtime, eager execution.** A `Device` interface, a CPU reference backend and a caching
allocator. The backend registers with PyTorch as `tessiq` through PrivateUse1. Allocation, copies
and views are native; the rest of ATen goes through a CPU fallback.

**2. CUDA device.** `CudaDevice` with async copies, streams and events. Index 0 is the CPU
reference and 1..N are GPUs, each with its own allocator. Cross-backend copies stage through the
host.

**3. IR and pass infrastructure.** SSA IR with a verifier and a round-tripping text format,
`PassManager` and pass registry, `dead-node-elimination`, `cse`, `const-fold`, `tessiq-opt` and
golden tests. No PyTorch dependency.

**4. Graph capture, partitioning.** FX importer, `mark-supported` and `cluster`, and the
`torch.compile` backend as a `torch_dynamo_backends` entry point. Unsupported ops run on the host
inside the compiled call.

**5. Shape inference, memory planning, recipe cache.** `shape-infer`, `algebraic-simplify`,
liveness-based arena planning, and `Recipe` with an output spec
([design note 0002](docs/design_notes/0002-output-structure-is-calling-convention.md)). Recipes
are cached by graph structure, value types and device.

**6. Fusion, CUDA codegen.** `fuse-elementwise` and an `Emitter` that writes each fusion group as
one CUDA kernel, compiled with NVRTC and cached as a cubin on disk.

**7. Op coverage, model blocks.** `ops.toml` and `gen_ops.py`; reductions, movement ops and cuBLAS
matmuls; `layout-assign`. A ResNet-18 residual block and a transformer encoder block match eager on
both backends.

**8. Training, dynamic shapes, tooling.** Backward graphs compile through the same path. Unknown
extents are `?` in the IR and kernels take extents as arguments: 20 sequence lengths through the
encoder block cause 2 compiles, and a symbolic graph gets the same 25 kernels as a static one.
Profiler with Chrome traces, and `tessiq.explain`. Bucketing was dropped since symbolic shapes made
it unnecessary.

**9. Device-side copies.** A pack kernel for strided tensors, used by `clone` and `layout-assign`.
The encoder block now runs 76 of 76 ops on the device.

**10. Performance analysis.** Device timing from events instead of synchronising around every
kernel, `tessiq.kernel_costs`, and `bench/where_the_gap_is.py`. Result: the device is idle for most
of a call, waiting on the Python executor and on steps interpreted through the dispatcher.

**11. Convolution.** cuDNN convolution, and eval-mode batch norm decomposed into fusable math. The
residual block runs on the device except four float32-to-float32 casts; it went from 0.90 ms to
0.57 ms per call (batch 8, 32 × 32).

**12. Custom operations, rewrite passes.** A single `Kernel` interface with an explicit stream per
launch. Custom operations live in `csrc/custom_ops/`, each with a schema, an ATen reference, and
an NVRTC kernel or a fusable expression. `.replaces()` makes one the implementation of a PyTorch op
(`override-<name>`); `.pattern()` adds a `rewrite-<name>` pass that runs when a compile asks for
it. With both rewrites, an RMSNorm + GELU block went from 6 kernels to 3 and from 60 µs to 34 µs
of device time. No backward and no schema defaults yet; rewrites skip training graphs.

**13. Precompiled custom operations.** `.precompiled()` host code in `<name>.nvcc.cu`, built by
nvcc for `CMAKE_CUDA_ARCHITECTURES`, gets the call's tensors, stream and allocator workspace, and is
refused on an architecture it wasn't built for. CUTLASS is fetched and cuBLAS linked only for files
that include them. Adds `layer_norm`, `gemm_bias_gelu` (CUTLASS) and `gemm_bias_gelu_tanh`
(cuBLASLt). cuBLASLt's fused epilogue beat cuBLAS + a GELU kernel at all four benchmark shapes;
the CUTLASS SIMT kernel only at one. Exact GELU moved onto the device, and nodes with keyword
arguments the kernel ignores (`alpha`, `beta`) now stay on the host. The rewrites only match 2-D
inputs.

**14. Eager elementwise operations on the device.** Every ATen spelling of an elementwise operation
in `ops.toml` gets a tessiq kernel, registered from the table at import. On a GPU a call runs one
kernel from the compiler's emitter, compiled once per operation, dtype and the way each operand is
read. Numbers are kernel arguments, so `x * 0.5` and `x * 0.25` share a kernel. Broadcast rows and
columns are read in place and other layouts are packed on the device. A non-default `alpha` or
`approximate`, type promotion, mixed dtypes or devices, and integer operations other than add,
sub, mul, neg and relu still go through the host. From DRAM a call matches PyTorch's CUDA kernel,
1.24 ms against 1.25 ms of device time at 1 GiB, where the fallback took about 170 ms. In L2
PyTorch's kernels are faster, 8 µs against 12 µs, and both calls take about 33 µs.

**15. Eager reductions, matmuls and convolution on the device.** The rest of `ops.toml` runs
eagerly on a GPU. `mm`, `bmm` and `addmm` call cuBLAS, reading packed columns through the transpose
flag, so a `Linear` weight and the transposes autograd makes are not copied. Convolution calls
cuDNN with a plan built once per shape. Reductions over one contiguous range of axes use the
compiler's reduction kernel, and a long axis with few outputs is reduced in two passes: a full sum
of 256 MiB went from 27 ms in one block to 0.32 ms, and drifts less than the host's. In-place and
`out=` forms of the elementwise operations write into their tensor with the same kernels.
Eval-mode batch norm and layer norm, which `ops.toml` does not list, are built from those kernels,
so the encoder and residual blocks run eagerly with no host calls: 0.68 and 0.54 ms a call, down
from 2.9 ms with the norms on the host, against 0.53 and 0.20 ms on eager CUDA, which runs cuDNN in
TF32 and has a batch norm kernel of its own.

## Planned

**16. MLIR export (optional).** Lower tessiq IR to `linalg` on tensors, generate ODS from
`ops.toml`, and lower through the standard bufferization pipeline. Behind `TESSIQ_ENABLE_MLIR`,
which is reserved and off. Custom operations export through their reference implementation. There
is no lowering-target interface yet; the exporter would consume the IR after the pass pipeline, as
`build_recipe` does.
