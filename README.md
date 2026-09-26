# tessiq

tessiq is an out-of-tree PyTorch backend for NVIDIA GPUs. It adds a `tessiq` device through
PrivateUse1 and a `torch.compile` backend of the same name. The compiler imports each captured
graph into its own IR, partitions and fuses it, and runs it with NVRTC-compiled kernels, cuBLAS,
cuDNN and custom operations.

- Device memory with a caching allocator per device, `torch.accelerator` memory stats, pinned
  memory, `torch.save`/`torch.load`, and a CPU fallback for every ATen op without a native kernel.
- An SSA IR with a textual form, a pass pipeline (partitioning, layout assignment, elementwise
  fusion), arena memory planning and a recipe cache.
- Generated fused elementwise, reduction and pack kernels, cached as cubins on disk.
- Symbolic shapes, and training through AOTAutograd.
- Custom operations in [`csrc/custom_ops/`](csrc/custom_ops/): NVRTC kernels, fusable expressions,
  or nvcc-compiled host code using CUTLASS or cuBLASLt. An operation can replace a PyTorch op or a
  pattern of ops.

## Requirements

| | |
|---|---|
| OS | Linux x86-64 |
| Python | 3.10+ |
| PyTorch | 2.11, CUDA 12.8 build |
| CUDA Toolkit | 12.x (tested 12.8). nvcc is optional; it builds the precompiled custom ops |
| cuDNN | 9.x from PyTorch's `nvidia-cudnn-cu12` wheel. Optional; convolution falls back to the host |
| CMake | 3.26+ |
| Compiler | C++20 (GCC 11 and 13 tested) |
| GPU | Tested on RTX 5080 (sm_120) |

Without the CUDA Toolkit, only the CPU reference device is built.

## Build

```bash
pip install -r requirements-dev.txt
pip install -e . --no-build-isolation
```

Precompiled custom ops are built for the GPUs in the machine. To target others:

```bash
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="89;120"
```

The C++ core (IR, passes, runtime) builds and tests without PyTorch:

```bash
cmake -S . -B build -G Ninja -DTESSIQ_BUILD_PYTHON=OFF
cmake --build build && ctest --test-dir build
```

| CMake option | Default | |
|---|---|---|
| `TESSIQ_ENABLE_CUDA` | ON if the toolkit is found | CUDA backend |
| `TESSIQ_BUILD_PYTHON` | ON | Python extension |
| `TESSIQ_BUILD_TESTS` | ON, OFF for wheels | C++ tests |
| `TESSIQ_WARNINGS_AS_ERRORS` | ON | `-Werror` |
| `TESSIQ_SANITIZER` | empty | e.g. `address,undefined` |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Targets for precompiled custom ops |
| `FETCHCONTENT_SOURCE_DIR_CUTLASS` | unset | Local CUTLASS checkout instead of downloading 4.8.0 |

## Usage

```python
import torch
import tessiq

model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
compiled = torch.compile(model, backend="tessiq", options={"device": "tessiq:1"})

batch = torch.randn(8, 16)
torch.testing.assert_close(compiled(batch), model(batch))
```

`backend="tessiq"` is a `torch_dynamo_backends` entry point, so it resolves without importing
tessiq. Options:

- `device`: where compiled clusters run. Defaults to `tessiq:0`, the CPU reference.
- `rewrites`: custom operations whose rewrite passes to run, e.g. `["rms_norm"]`.

Inputs may be host or tessiq tensors; an input unchanged since the last call stays on the device.
Results are returned on the host. Operations the backend cannot run stay on the host inside the
compiled call.

Devices:

| Device | Python | |
|---|---|---|
| `tessiq:0` | `tessiq.cpu` | CPU reference, used to check GPU results |
| `tessiq:n`, n ≥ 1 | `tessiq.gpu`, `tessiq.gpu_at(n)` | CUDA device `n - 1` |

```python
x = torch.randn(1024, device=tessiq.gpu)
y = (x * 2).to(tessiq.cpu)
tessiq.memory_stats(tessiq.gpu).allocated_bytes
```

Outside `torch.compile`, most ops on tessiq tensors run through the CPU fallback (copy to host,
compute, copy back). Allocation, copies, views, `clone` and custom operations run on the device.

To see how a model compiles, what each pass changed, which kernels were built and where every op
runs:

```python
print(tessiq.explain(model, batch, device=tessiq.gpu))
```

Custom operations are registered as `torch.ops.tessiq.*` on import:

| Op | Implementation |
|---|---|
| `rms_norm` | NVRTC kernel; rewrite for RMSNorm written out |
| `softmax` | NVRTC kernel; replaces `aten::_softmax` |
| `fast_gelu`, `swiglu` | Fused elementwise expressions; `fast_gelu` has a rewrite |
| `layer_norm` | nvcc host code, two launches |
| `gemm_bias_gelu` | CUTLASS GEMM, bias + exact GELU epilogue; rewrite for `Linear` + GELU |
| `gemm_bias_gelu_tanh` | cuBLASLt with GELU_BIAS epilogue; rewrite for `Linear` + tanh GELU |

See [`csrc/custom_ops/README.md`](csrc/custom_ops/README.md) for adding one.

Environment variables:

| | |
|---|---|
| `TESSIQ_LOG_LEVEL` | `error`, `warn` (default), `info`, `debug`, `trace` |
| `TESSIQ_LOG_MODULES` | Comma-separated modules to log, e.g. `passes,custom_ops` |
| `TESSIQ_CACHE_DIR` | Cubin cache. Default `$XDG_CACHE_HOME/tessiq/cubin` or `~/.cache/tessiq/cubin` |

## How it works

```
  torch.compile(model, backend="tessiq")      tensor.to("tessiq")
              │                                        │
              v                                        v
   Dynamo → FX → AOTAutograd                    ATen dispatch
              │                                        │
              v                                        │
       tessiq IR ──> passes ──> memory plan ──> recipe │
              v                                        v
        ┌──────────────────── kernels ──────────────────┐
        │  CPU reference     CUDA: NVRTC, cuBLAS,       │
        │                    cuDNN, custom operations   │
        └───────────────────────────────────────────────┘
```

AOTAutograd lowers the captured graph to core ATen, and the importer converts it to tessiq IR. The
passes ([`docs/passes.md`](docs/passes.md)) mark what the device can run, cluster it and fuse
elementwise chains. The result is a recipe: kernels, a device per step, which steps move their
arguments, and an arena size. The executor runs it per call; steps without a kernel go through the
ATen dispatcher.

Operations are defined in [`ops.toml`](ops.toml), which generates the C++ op table and
[`docs/op_coverage.md`](docs/op_coverage.md). Generated kernels handle float32, float64, int32 and
int64; cuBLAS and cuDNN are used for float32.

## Performance

RTX 5080, PyTorch 2.11, medians. Scripts are in [`bench/`](bench/).

`(a * b + c).relu().sigmoid()`, 4M floats, data on the device
([`against_torch.py`](bench/against_torch.py)):

| | ms | GB/s |
|---|---|---|
| tessiq fused kernel | 0.057 | 1101 |
| Inductor | 0.072 | 874 |

Sum of a 4096 × 4096 tensor ([`reduction.py`](bench/reduction.py)):

| Axis | tessiq | PyTorch |
|---|---|---|
| last | 1494 GB/s | 1463 GB/s |
| first (strided) | 649 GB/s | 1323 GB/s |

Linear + GELU, device time ([`gemm_bias_gelu.py`](bench/gemm_bias_gelu.py)):

| m × k → n | cuBLAS + GELU kernel | CUTLASS `gemm_bias_gelu` | cuBLASLt `gemm_bias_gelu_tanh` |
|---|---|---|---|
| 512 × 768 → 3072 | 114 µs | 126 µs | 103 µs |
| 4096 × 1024 → 4096 | 1369 µs | 1204 µs | 1236 µs |
| 8192 × 768 → 3072 | 1772 µs | 1840 µs | 1580 µs |
| 64 × 4096 → 4096 | 104 µs | 113 µs | 97 µs |

The cuBLASLt column computes tanh GELU; the others compute exact GELU.

Transformer encoder block, 512 wide, 128 steps, batch 8, input on the host
([`against_torch.py`](bench/against_torch.py)):

| | ms/call |
|---|---|
| tessiq | 1.14 |
| PyTorch eager, GPU | 0.88 |
| PyTorch eager, GPU, input resident | 0.53 |
| Inductor, input resident | 0.48 |
| PyTorch eager, CPU | 8.83 |

The gap is host overhead, not kernels. Of a tessiq call, 42% is interpreting the 41 steps with no
kernel (views, permutes, clones) through the dispatcher, 34% is issuing the 25 kernels, and 24% is
copying the result back ([`where_the_time_goes.py`](bench/where_the_time_goes.py)).

## Limitations

- Importing tessiq makes it PyTorch's accelerator, which breaks CUDA autograd in the same process
  ([design note 0003](docs/design_notes/0003-one-accelerator-per-process.md)).
- Compiled calls return results on the host.
- Eager ops on tessiq tensors mostly run through the CPU fallback.
- cuBLAS, cuDNN and most custom ops are float32 only.
- Custom operations have no backward and no default arguments.
- Random ops run on the host.

## Repository

| | |
|---|---|
| [`csrc/tessiq/`](csrc/tessiq/) | C++: `support`, `runtime`, `ir`, `passes`, `codegen`, `torch` |
| [`csrc/bindings/`](csrc/bindings/) | pybind11 module |
| [`csrc/custom_ops/`](csrc/custom_ops/) | Custom operations |
| [`python/tessiq/`](python/tessiq/) | Python API and `torch.compile` backend |
| [`tools/`](tools/) | `tessiq-opt`, `tessiq_explain.py`, `model_coverage.py`, `gen_ops.py` |
| [`tests/`](tests/) | GoogleTest, golden pass tests, pytest |
| [`bench/`](bench/) | Benchmarks |
| [`docs/`](docs/) | Pass pipeline, op coverage, design notes |

[`ROADMAP.md`](ROADMAP.md) tracks what is done and planned, [`CHANGELOG.md`](CHANGELOG.md) lists
changes, and [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, CI checks and conventions.

## License

Apache 2.0. See [`LICENSE`](LICENSE).
