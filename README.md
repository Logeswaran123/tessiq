# tessiq

A PyTorch accelerator backend built from scratch: a custom device, a graph compiler, and a kernel
library. It exists to answer, in working code, what a hardware vendor actually has to build to put
a chip behind `model.to(...)` and `torch.compile`.

Named after *tessera*, the Latin word for a mosaic tile, which is what the compiler makes of a
graph: pieces cut to fit, then set into one surface. Said `/ˈtɛs.ɪk/`.

Phases 1 to 6 of 10 are done. Tensors allocate, transfer and compute on two backends, each with
its own caching allocator. `torch.compile(model, backend="tessiq")` captures a graph, imports it
into the IR, partitions it into the parts the backend claims, groups the elementwise chains, and
generates a CUDA kernel for each group which it compiles with NVRTC and caches on disk. Forward
and backward both go through it, and a graph compiled once is not compiled again.

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
$ tessiq-opt --passes=cse,dce graph.tsr
graph @demo(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {
  %s0 = add(%a, %b) : f32[4,8]
  %out = mul(%s0, %s0) : f32[4,8]
  return %out
}
```

**Anything outside a fused chain still runs on the host.** An op the backend has no kernel for
reaches PyTorch's CPU implementation through the fallback, which costs two transfers. What is fast
is what fuses: a chain of elementwise operations becomes one kernel, its intermediates never leave
a register, and the generated code is cached as a cubin between runs. Reductions get a kernel of
their own, since they read a whole axis to write one element and cannot share that loop.
Thirty-seven of the 38 operations the compiler knows have kernels; only matmul does not.
[`docs/op_coverage.md`](docs/op_coverage.md) is generated from [`ops.toml`](ops.toml) and lists all
of them. [PHASES.md](PHASES.md) has the full breakdown.

`(a*b + c).relu().sigmoid()` on 4M floats on an RTX 5080. The generated kernel is good and the
path around it is not, so both are measured. From
[`bench/against_torch.py`](bench/against_torch.py), with the data already on the device:

| | | |
|---|---|---|
| tessiq generated kernel | 0.035 ms | 1766 GB/s |
| `torch.compile` inductor | 0.064 ms | 975 GB/s |

Three loads, four operations in registers, one store, and near the bandwidth the card can sustain.
Measure the whole call from host tensors instead and PyTorch wins by 60x, because tessiq moves 48
MB to the device and 16 MB back every call and the kernel is one percent of what that costs. Only
part of a graph is claimed today, so every unclaimed operation pulls its operands back to the
host. Matmul is still on that list.

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
residual block: 3 of 7 operations on the device (43%), 3 claimable
  aten.convolution.default                             2
  aten._native_batch_norm_legit_no_training.default    2

encoder block: 76 of 80 operations on the device (95%), 76 claimable
  aten.clone.default                                   4
```

The two numbers differ on purpose. **Claimable** is what the backend has an implementation for.
**On the device** is what partitioning puts there, which is fewer: a movement operation costs
nothing on either side, so a group holding nothing but views is left where its input already was
rather than paying two transfers to rearrange indices somewhere else.

Both blocks give the numbers eager gives, on the reference backend and on a GPU.

**Correct is not yet fast on these, and coverage is no longer what is holding it back.** A
512-wide encoder block over 128 steps runs in 21.9 ms against 8.6 ms for eager on the CPU, with 95%
of it on the device. The recipe says why: 80 steps produce 16 launches and 28 transfers, and only
two of those launches are a fused loop. Breaking a softmax into its five operations puts every one
of them on the device and gives the device five things to launch.

What would change that is fusing more per launch, which means a kernel that can read an operand
with fewer elements than the result. A layer norm subtracts one mean per row, and that one
broadcast is what keeps the rest of the chain from sharing a loop with it.

## How it fits together

```
  torch.compile(model, backend="tessiq")      model.to("tessiq")
              │                                        │
              v                                        v
   Dynamo → FX → AOTAutograd                   aten op → dispatch
              │                                        │
              v                                        │
       Tessiq IR ──> passes ──> lowering ──> recipe   │
              │        canonicalize                    │
              │        shape infer                     │
              │        layout                          │
              │        fuse                            │
              │        schedule                        │
              │        memory plan                     │
              v                                        v
        ┌─────────────── kernel registry ───────────────┐
        │  CPU reference              CUDA + NVRTC      │
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
C++17). The extension links libtorch, so PyTorch is needed to import
tessiq. The C++ core builds and tests without it, which is what keeps the compiler testable on a
machine with neither torch nor a GPU:

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
| [`csrc/`](csrc/) | C++/CUDA core: runtime, IR, passes, lowering, codegen, kernels |
| [`python/tessiq/`](python/tessiq/) | Python API and the `torch.compile` backend |
| [`ops/`](ops/) | Op definitions in YAML, and the source of truth for codegen (phase 7) |
| [`tools/`](tools/) | `gen_ops.py`, `tessiq-opt`, `tessiq-explain` (phases 3, 7, 8) |
| [`tests/`](tests/) | gtest suites and golden pass tests; Python accuracy and end-to-end tests |
| [`bench/`](bench/) | Benchmarks, and the scripts behind any number published here |
| [`docs/`](docs/) | Architecture, IR spec, pass reference, design notes |

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
