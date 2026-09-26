# tessiq

A PyTorch accelerator backend built from scratch: a custom device, a graph compiler, and a kernel
library. It exists to answer, in working code, what a hardware vendor actually has to build to put
a chip behind `model.to(...)` and `torch.compile`.

Named after *tessera*, the Latin word for a mosaic tile, which is what the compiler makes of a
graph: pieces cut to fit, then set into one surface. Said `/ˈtɛs.ɪk/`.

Phases 0 to 12 of [15](PHASES.md) are done. Tensors allocate, transfer and compute on two
backends, each with its own caching allocator. `torch.compile(model, backend="tessiq")` captures a
graph, imports it into the IR, decides what the backend takes, and compiles a kernel for each part:
fused loops and reductions it generates and compiles with NVRTC, matmuls through cuBLAS,
convolutions through cuDNN, and custom operations defined in
[`csrc/custom_ops/`](csrc/custom_ops/). Forward and backward both go through it, and a graph
compiled once is not compiled again.

```python
import torch

model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
compiled = torch.compile(model, backend="tessiq")  # the name resolves without importing tessiq
batch = torch.randn(8, 16)

torch.allclose(compiled(batch), model(batch))
```

The device is usable on its own, without the compiler:

```python
import torch, tessiq

reference = torch.randn(1024, device=tessiq.cpu)  # host memory, the correctness oracle
fast = reference.to(tessiq.gpu)  # real GPU memory

torch.allclose(fast.to(tessiq.cpu), reference)
tessiq.memory_stats(tessiq.gpu).allocated_bytes
```

Passes are exercised as text in, text out:

```console
$ tessiq-opt --passes=cse,dead-node-elimination graph.tsr
graph @demo(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {
  %s0 = add(%a, %b) : f32[4,8]
  %out = mul(%s0, %s0) : f32[4,8]
  return %out
}
```

[`docs/passes.md`](docs/passes.md) lists the passes a compile runs and the order it runs them in,
and how to add one.

**What runs on the device.** All 52 operations the compiler knows have kernels. A chain of
elementwise operations becomes one kernel, its intermediates never leave a register, and the
generated code is cached as a cubin between runs. Reductions get a kernel of their own, since they
read a whole axis to write one element and cannot share that loop. Matmuls call cuBLAS and
convolutions call cuDNN, both float32 only for now. Anything else, an operation of a type its kernel
does not take included, reaches PyTorch's CPU implementation through the fallback, which costs two
transfers. [`docs/op_coverage.md`](docs/op_coverage.md) is generated from [`ops.toml`](ops.toml)
and lists all of them. A custom operation brings a kernel of its own, compiled at run time or, as
host code, when tessiq is built, or calls a library such as cuBLASLt. It can stand in for a torch
operation or for a computation a model writes out; see [`csrc/custom_ops/`](csrc/custom_ops/).

**A shape that changes costs one compile, not one per shape.** Dynamo captures concrete shapes the
first time it sees a function and recompiles with symbolic ones the second time a dimension
changes, which is its default rather than something a caller asks for. The IR records an extent it
does not know as `?`, and every kernel takes the numbers it needs as arguments: the element count
and a broadcast's divisor for a fused loop, three extents for a reduction, and m, n and k for a
matmul. On the encoder block in [`tests/python/test_models.py`](tests/python/test_models.py) that
is the same 28 kernels a fixed shape gets, and the only data to cross between host and device is
the 128 KB result coming back. The one form left out is stepping through a permuted operand axis by
axis, which needs an extent per axis written into the source.

`(a*b + c).relu().sigmoid()` on 4M floats on an RTX 5080. The generated kernel is good and the
path around it is not, so both are measured. From
[`bench/against_torch.py`](bench/against_torch.py), with the data already on the device, medians of
five runs:

| | | |
|---|---|---|
| tessiq generated kernel | 0.074 ms | 904 GB/s |
| `torch.compile` inductor | 0.089 ms | 756 GB/s |

Three loads, four operations in registers, one store. Measure the whole call from host tensors
instead and PyTorch wins by 18x, because tessiq moves 48 MB to the device and 16 MB back every call
and the kernel is about 7% of that call.

Within tessiq, a kernel beats running the same graph without one by 5.7x
([`bench/fusion.py`](bench/fusion.py)). Compilation costs 440 us and a cached recipe costs 110 us
([`bench/compile_overhead.py`](bench/compile_overhead.py)).

Reductions, from [`bench/reduction.py`](bench/reduction.py), summing a 4096 by 4096 tensor with
the data already on the device:

| | tessiq | torch |
|---|---|---|
| last axis | 1387 GB/s | 1388 GB/s |
| first axis | 601 GB/s | 1261 GB/s |

Reducing the last axis reads contiguous memory and lands where PyTorch does. Reducing the first
reads with a stride and is 2.1x behind, and a short axis with many outputs is further behind than
that, because a block of 256 threads combining 64 elements leaves most of them idle.

**How much of a real model this covers is a separate question, and a smaller number.** From
[`tools/model_coverage.py`](tools/model_coverage.py):

```
residual block: 37 of 41 operations on the device (90%), 37 claimable
  left on the host                                 count
  aten._to_copy.default                                4

encoder block: 76 of 76 operations on the device (100%), 76 claimable
```

The four left on the residual block are float32 to float32 casts that the batch norm decomposition
puts on its running statistics.

The two numbers differ on purpose. **Claimable** is what the backend has an implementation for.
**On the device** is what partitioning puts there, which is fewer: a movement operation costs
nothing on either side, so a group holding nothing but views is left where its input already was
rather than paying two transfers to rearrange indices somewhere else.

Both blocks give the numbers eager gives, on the reference backend and on a GPU.

**A whole block is a different measurement, and a less flattering one.** A 512-wide encoder block
over 128 steps at batch 8, from [`bench/against_torch.py`](bench/against_torch.py), medians of five
runs:

| | | |
|---|---|---|
| tessiq | 1.17 ms | |
| torch on the same GPU, moving the input too | 0.83 ms | 1.4x faster |
| torch on the same GPU, everything resident | 0.61 ms | 1.9x faster |
| `torch.compile` inductor, resident | 0.54 ms | 2.2x faster |
| torch eager on the CPU | 9.1 ms | 7.8x slower |

Well ahead of the CPU, and about twice as slow as the card it runs on can manage.

**The weights used to cross on every call**, because AOTAutograd lifts parameters to graph inputs
and they arrive as ordinary arguments. The executor keeps the device copy of an input that has not
changed, checked against its version counter so a parameter an optimizer writes to crosses again.
That was 2 ms.

**What is left is the host.** From [`bench/where_the_time_goes.py`](bench/where_the_time_goes.py)
on the same block:

```
              count        host   share      device       moved
interpreted      41      0.40 ms     40%
kernel           25      0.40 ms     40%
transfer          1      0.19 ms     19%                  2.0 MB
total            67      1.00 ms
```

One transfer is left, the 2 MB result coming back. The rest of the call is the host issuing 25
kernels and interpreting 41 steps that have no kernel: views, permutes and clones going one
operation at a time through PyTorch's dispatcher, at about ten microseconds a step whatever the
tensors are. That is the price of interpreting a graph rather than running it, and it is paid per
operation. Phase 10 measured it: the device is idle for most of the call, waiting for the host to
issue the next piece of work.

The generated code is not what is behind: on a single chain it is
[904 GB/s against Inductor's 756](bench/against_torch.py). Everything around it is.

## How it fits together

```
  torch.compile(model, backend="tessiq")      model.to("tessiq")
              │                                        │
              v                                        v
   Dynamo → FX → AOTAutograd                   aten op → dispatch
              │                                        │
              v                                        │
       Tessiq IR ──> passes ──> memory plan ──> recipe │
              │        rewrites, overrides             │
              │        mark-supported                  │
              │        layout-assign                   │
              │        cluster                         │
              │        fuse-elementwise                │
              v                                        v
        ┌──────────────────── kernels ──────────────────┐
        │  CPU reference     CUDA: NVRTC, cuBLAS,       │
        │                    cuDNN, custom operations   │
        └───────────────────────────────────────────────┘
```

Two ways in. `torch.compile` hands over a whole graph, which is where the compiler earns its keep.
`.to("tessiq")` dispatches one op at a time and catches whatever the compiler declines to take.

The CPU backend is there to be obviously right rather than fast; CUDA output is checked against it.
The passes don't know what hardware they're targeting, so CI runs the compiler on a GPU-less runner.

## Build

```bash
pip install -r requirements-dev.txt
pip install -e . --no-build-isolation
pytest

cmake -S . -B build -G Ninja
cmake --build build -j8
ctest --test-dir build
```

Python 3.10+, CMake 3.26+, and a compiler with C++20 for the torch-facing code (the core is
C++17). With the CUDA toolkit's nvcc, the precompiled custom operations are built for the GPUs in
the machine; set `CMAKE_CUDA_ARCHITECTURES` to build for others. The one that uses CUTLASS has its
headers downloaded when the build is configured. The extension links libtorch, so PyTorch is needed
to import tessiq. The C++ core builds and tests without it, which is what keeps the compiler
testable on a machine with neither torch nor a GPU:

```bash
cmake -S . -B build -G Ninja -DTESSIQ_BUILD_PYTHON=OFF
```

If you're going to commit:

```bash
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

## Layout

| Path | |
|---|---|
| [`csrc/`](csrc/) | C++/CUDA core: runtime, IR, passes, codegen, and the PyTorch integration |
| [`csrc/custom_ops/`](csrc/custom_ops/) | Custom operations, one directory each |
| [`python/tessiq/`](python/tessiq/) | Python API and the `torch.compile` backend |
| [`ops.toml`](ops.toml) | Every operation the compiler knows, and what generates the rest from it |
| [`tools/`](tools/) | `gen_ops.py`, `tessiq-opt`, `tessiq_explain.py`, `model_coverage.py` |
| [`tests/`](tests/) | gtest suites and golden pass tests; Python accuracy and end-to-end tests |
| [`bench/`](bench/) | Benchmarks, and the scripts behind any number published here |
| [`docs/`](docs/) | The pass pipeline, operation coverage, design notes |

## Standards

`-Werror` with a wide warning set, clang-tidy, ruff, `mypy --strict` and ASan/UBSan all gate merges,
and the pre-commit hooks run the same versions CI does. The rules that come up most: user input
never reaches an assert, every pass states why its transformation is legal, and no performance
claim ships without the script that produced it.

[CONTRIBUTING.md](CONTRIBUTING.md) has the rest, and the commands to run every gate locally.

## Benchmarks

Nothing worth showing before phase 6. Numbers land here when they exist, next to the scripts that
produced them.

## Prior art

Built on publicly documented work: PyTorch's out-of-tree backend mechanism and `torch.compile`
internals, `pytorch/xla`, MLIR's pass infrastructure, IREE, TVM. Non-obvious decisions cite their
precedent where they're made.

## License

Apache-2.0.
