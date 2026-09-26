# tessiq

tessiq is a PyTorch backend for NVIDIA GPUs, built outside the PyTorch source tree. It adds two
things to PyTorch:

- **A `tessiq` device.** Tensors on it are allocated, copied and freed by tessiq's own runtime,
  the way tensors on `cuda` are by PyTorch's.
- **A `torch.compile` backend, also named `tessiq`.** It takes the graphs `torch.compile` captures
  and runs them on the tessiq device. Elementwise chains are fused into generated CUDA kernels,
  matrix multiplies go to cuBLAS, and convolutions go to cuDNN.

It is structured the way a hardware vendor would integrate a new accelerator. The device is
registered through PyTorch's PrivateUse1 extension point. The compiler has its own IR, pass
pipeline, code generator and kernel library. You can extend it with your own operations, written
as CUDA kernels, CUTLASS templates or cuBLAS calls, without touching the rest of the code.

```python
import torch
import tessiq

model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
compiled = torch.compile(model, backend="tessiq", options={"device": "tessiq:1"})

out = compiled(torch.randn(8, 16))
```

tessiq is in early development (version 0.0.1). It builds from source on Linux and is tested on
an RTX 5080.

## How it works

There are two ways to use tessiq.

**Eager tensors.** `torch.randn(1024, device="tessiq:1")` allocates on the GPU through tessiq's
caching allocator. Copies, views and custom operations run on the device. Most other operations on
tessiq tensors run through a CPU fallback: the inputs are copied to the host, PyTorch computes the
result, and it is copied back. The results are correct but slow; this path exists so that
everything works.

**Compiled models.** `torch.compile(model, backend="tessiq")` is where the performance is:

```
 model ──> Dynamo ──> FX graph ──> AOTAutograd ──> tessiq IR ──> passes ──> recipe ──> run
                                   (core ATen)                   │
                                                                 ├─ fused CUDA kernels (NVRTC)
                                                                 ├─ cuBLAS, cuDNN
                                                                 └─ custom operations
```

1. Dynamo captures the model as an FX graph, and AOTAutograd lowers it to core ATen operations,
   including the backward graph when training.
2. tessiq imports the graph into its own IR.
3. A pipeline of passes marks each operation the device can run, and groups neighbouring ones into
   clusters. Chains of elementwise operations inside a cluster are fused into one kernel.
4. The result is a recipe: the kernels to launch, where each step runs, and one memory arena for
   the intermediates. Recipes are cached, so a graph is compiled once.
5. Each call runs the recipe. Some steps have no kernel, such as views and permutes; they are
   interpreted on the device. Operations the backend cannot run go to the host, inside the same
   call.

Generated kernels are compiled for the GPU in the machine the first time they are needed, and the
binaries are cached on disk for the next run.

## Requirements

| | |
|---|---|
| OS | Linux, x86-64 |
| Python | 3.10 or later |
| PyTorch | 2.11, CUDA 12.8 build |
| CUDA Toolkit | 12.x, tested with 12.8. `nvcc` is only needed for the precompiled custom operations |
| cuDNN | 9.x. It comes with PyTorch's `nvidia-cudnn-cu12` package. Without it, convolution runs on the host |
| CMake | 3.26 or later |
| C++ compiler | GCC with C++20 support (11 and 13 are tested) |
| GPU | An NVIDIA GPU supported by CUDA 12. Tested on an RTX 5080 (compute capability 12.0) |

Without the CUDA Toolkit, tessiq still builds, with only the CPU reference device.

## Installation

tessiq is built from source.

```bash
git clone https://github.com/Logeswaran123/tessiq.git
cd tessiq
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt        # build, test and lint tools
pip install -e . --no-build-isolation      # builds the C++/CUDA extension
```

`--no-build-isolation` matters: the extension is compiled against the PyTorch you just installed.

Check that it works, and that it found your GPU:

```console
$ python -c "import tessiq; print(tessiq.__version__, tessiq.gpu_count())"
0.0.1 1
```

A GPU count of 0 means the CUDA backend was not built, usually because CMake did not find the
toolkit. `pip install -v` shows the configuration summary, where this appears as
`cuda backend ....... OFF`. Point CMake at the toolkit and rebuild:

```bash
pip install -e . --no-build-isolation -Ccmake.define.CUDAToolkit_ROOT=/usr/local/cuda
```

Custom operations compiled with nvcc are built for the GPUs in the build machine. To build them
for other GPUs as well, list their architectures:

```bash
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="89;120"
```

### Build options

Pass any of these with `-Ccmake.define.<OPTION>=<value>`:

| Option | Default | |
|---|---|---|
| `TESSIQ_ENABLE_CUDA` | ON if the toolkit is found | Build the CUDA backend |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | GPUs to build the precompiled custom operations for |
| `FETCHCONTENT_SOURCE_DIR_CUTLASS` | unset | Use a local CUTLASS checkout instead of downloading CUTLASS 4.8.0 |
| `TESSIQ_WARNINGS_AS_ERRORS` | ON | Treat compiler warnings as errors |
| `TESSIQ_SANITIZER` | empty | Sanitizers to build with, for example `address,undefined` |

## Quick start

### Compile a model

```python
import torch
import tessiq

model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
compiled = torch.compile(model, backend="tessiq", options={"device": "tessiq:1"})

batch = torch.randn(8, 16)
torch.testing.assert_close(compiled(batch), model(batch))
```

Always pass `options={"device": ...}`. Without it, the model compiles for `tessiq:0`, the CPU
reference device. That is useful for checking results but generates no GPU kernels.

- **Inputs** can be ordinary CPU tensors or tessiq tensors. Weights and other inputs that have not
  changed since the previous call stay on the device.
- **Results** are returned as CPU tensors.
- **Training** works the same way; the backward pass is compiled too.

`backend="tessiq"` also works without `import tessiq`, because the backend is registered as a
`torch_dynamo_backends` entry point.

### See how a model was compiled

`tessiq.explain` shows what the compiler did with a model:

```python
print(tessiq.explain(model, batch, device=tessiq.gpu))
```

```
== summary ==

model           Sequential on tessiq:1
shapes          known
operations      5
on the device   5
kernels         3, covering 3 operation(s)
...
== schedule ==

  permute       aten.permute.default     interpreted on the device (moves its inputs)
  addmm         aten.addmm.default       kernel %addmm (moves its inputs)
  gelu          aten.gelu.default        kernel %gelu
  permute_1     aten.permute.default     interpreted on the device (moves its inputs)
  addmm_1       aten.addmm.default       kernel %addmm_1 (moves its inputs)
```

The report also shows the imported graph, what each pass changed, and each kernel. Every operation
is placed one of three ways:

| Placement | Meaning |
|---|---|
| `kernel` | It runs in a kernel |
| `interpreted on the device` | It has no kernel, but runs on device tensors. A view moves no data |
| `host` | It runs on the CPU, with a copy each way |

### Work with tensors on the device

```python
x = torch.randn(1024, device=tessiq.gpu)  # same as device="tessiq:1"
y = x.to(tessiq.cpu)  # the CPU reference device, tessiq:0
z = x.cpu()  # an ordinary CPU tensor

tessiq.memory_stats(tessiq.gpu).allocated_bytes
torch.accelerator.memory_allocated(tessiq.gpu)
```

`device="tessiq"` without an index means the current tessiq device, which is `tessiq:0` until you
call `tessiq.set_device(tessiq.gpu)`.

### Use custom operations

tessiq ships custom operations under `torch.ops.tessiq`. You can call them directly:

```python
x = torch.randn(8, 64, device=tessiq.gpu)
weight = torch.randn(64, device=tessiq.gpu)
torch.ops.tessiq.rms_norm(x, weight, 1e-6)
```

Some also have a rewrite, which finds the same computation written in plain PyTorch and replaces
it. Rewrites change numerics slightly, so you opt in per compile:

```python
compiled = torch.compile(
    model, backend="tessiq", options={"device": "tessiq:1", "rewrites": ["gemm_bias_gelu"]}
)
```

For the model above, this turns the first `Linear` and the `GELU` into a single CUTLASS kernel.

| Operation | What it is | Rewrite matches |
|---|---|---|
| `rms_norm` | RMSNorm in one kernel | RMSNorm written out |
| `softmax` | Softmax kernel. Replaces PyTorch's float32 softmax on tessiq devices, no rewrite needed | |
| `fast_gelu` | Sigmoid-approximated GELU, fused with neighbouring operations | `x * sigmoid(1.702 * x)` |
| `swiglu` | `silu(gate) * up`, fused with neighbouring operations | |
| `layer_norm` | Layer norm in two kernels | |
| `gemm_bias_gelu` | CUTLASS GEMM with the bias and GELU fused into it | `Linear` then `GELU` |
| `gemm_bias_gelu_tanh` | cuBLASLt GEMM with the bias and tanh GELU fused into it | `Linear` then `GELU(approximate="tanh")` |

`tessiq.library.registered()` lists them, and `tessiq.library.rewrites()` lists the rewrites.
[`csrc/custom_ops/README.md`](csrc/custom_ops/README.md) explains how to add your own.

## Devices

PyTorch gives an out-of-tree backend one device type, so tessiq tells its backends apart by index:

| Device | Python name | Backed by |
|---|---|---|
| `tessiq:0` | `tessiq.cpu` | Host memory. The reference implementation that GPU results are tested against |
| `tessiq:1` | `tessiq.gpu`, `tessiq.gpu_at(0)` | The first CUDA GPU |
| `tessiq:2`, ... | `tessiq.gpu_at(1)`, ... | Further GPUs |

Each device has its own caching allocator. `tessiq.memory_stats()`, `tessiq.empty_cache()` and
PyTorch's `torch.accelerator` memory functions report on and manage it.

## Configuration

`torch.compile` options:

| Option | Default | |
|---|---|---|
| `device` | `"tessiq:0"` | The tessiq device to compile for |
| `rewrites` | none | Custom operations whose rewrites to run, such as `["rms_norm"]` |

Environment variables:

| Variable | Default | |
|---|---|---|
| `TESSIQ_LOG_LEVEL` | `warn` | `error`, `warn`, `info`, `debug` or `trace` |
| `TESSIQ_LOG_MODULES` | all | Comma-separated modules to log, such as `passes,custom_ops` |
| `TESSIQ_CACHE_DIR` | `~/.cache/tessiq/cubin` | Where compiled kernels are cached. Honours `XDG_CACHE_HOME` |

## Performance

These were measured on an RTX 5080 with PyTorch 2.11, and each is a median. The scripts are in
[`bench/`](bench/).

**One fused kernel.** `(a * b + c).relu().sigmoid()` on 4M floats, with the data already on the
GPU, runs as a single tessiq kernel. It is faster than Inductor's
([`against_torch.py`](bench/against_torch.py)):

| | Time | Bandwidth |
|---|---|---|
| tessiq | 0.057 ms | 1101 GB/s |
| `torch.compile` (Inductor) | 0.072 ms | 874 GB/s |

**Linear + GELU.** GPU time for the three ways tessiq can run it
([`gemm_bias_gelu.py`](bench/gemm_bias_gelu.py)). The cuBLASLt version computes the tanh
approximation of GELU; the others compute the exact one.

| m × k → n | cuBLAS, then a GELU kernel | CUTLASS `gemm_bias_gelu` | cuBLASLt `gemm_bias_gelu_tanh` |
|---|---|---|---|
| 512 × 768 → 3072 | 114 µs | 126 µs | 103 µs |
| 4096 × 1024 → 4096 | 1369 µs | 1204 µs | 1236 µs |
| 8192 × 768 → 3072 | 1772 µs | 1840 µs | 1580 µs |
| 64 × 4096 → 4096 | 104 µs | 113 µs | 97 µs |

**A whole model block.** A transformer encoder block (512 wide, 128 steps, batch 8), called with
its input on the host ([`against_torch.py`](bench/against_torch.py)):

| | Time per call |
|---|---|
| tessiq | 1.14 ms |
| PyTorch eager, GPU | 0.88 ms |
| PyTorch eager, GPU, input already on the GPU | 0.53 ms |
| `torch.compile` (Inductor), input already on the GPU | 0.48 ms |
| PyTorch eager, CPU | 8.83 ms |

End to end, tessiq is behind PyTorch on the GPU, and most of a call is spent on the host
([`where_the_time_goes.py`](bench/where_the_time_goes.py)):

| Share of a call | |
|---|---|
| 42% | Steps without a kernel (views, permutes, clones), run one at a time through PyTorch's dispatcher |
| 34% | Launching the 25 kernels |
| 24% | Copying the result back to the host |

## Limitations

- **CUDA autograd in the same process.** Importing tessiq makes it PyTorch's accelerator, and
  `backward()` on ordinary CUDA tensors then fails in the same process. See
  [design note 0003](docs/design_notes/0003-one-accelerator-per-process.md).
- **Results come back to the host.** Compiled calls always return CPU tensors, so a large output
  pays a device-to-host copy on every call.
- **Eager operations are slow.** Outside `torch.compile`, most operations on tessiq tensors go
  through the CPU fallback.
- **Data types.** Generated kernels support float32, float64, int32 and int64. cuBLAS, cuDNN and
  most custom operations are float32 only, and other types run on the host.
- **Custom operations.** They have no backward pass yet, and their schemas can't have default
  arguments.
- **Random numbers.** Random operations run on the host.

## Development

Run the tests:

```bash
pytest                    # Python tests; add -m "not cuda" on a machine without a GPU
```

The C++ core (IR, passes, runtime) builds and tests without PyTorch:

```bash
cmake -S . -B build -G Ninja -DTESSIQ_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build
```

Developer tools ([`tools/README.md`](tools/README.md)):

- **`tessiq-opt`** runs compiler passes on IR text files.
- **`tools/tessiq_explain.py`** runs `explain` on the test suite's model blocks.
- **`tools/model_coverage.py`** reports how much of a model runs on the device.

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers the development setup, the checks CI runs, and code
conventions.

## Documentation

| | |
|---|---|
| [`csrc/custom_ops/README.md`](csrc/custom_ops/README.md) | Writing custom operations |
| [`docs/passes.md`](docs/passes.md) | The compiler's pass pipeline, and adding a pass |
| [`docs/op_coverage.md`](docs/op_coverage.md) | The operations the compiler supports |
| [`docs/design_notes/`](docs/design_notes/) | Design decisions |
| [`bench/README.md`](bench/README.md) | Benchmarks |
| [`ROADMAP.md`](ROADMAP.md) | What is done and what is planned |
| [`CHANGELOG.md`](CHANGELOG.md) | Changes |

## Repository layout

| | |
|---|---|
| [`python/tessiq/`](python/tessiq/) | The Python package and the `torch.compile` backend |
| [`csrc/tessiq/`](csrc/tessiq/) | The C++ core: runtime, IR, passes, code generation, PyTorch integration |
| [`csrc/custom_ops/`](csrc/custom_ops/) | Custom operations, one directory each |
| [`ops.toml`](ops.toml) | The operations the compiler knows |
| [`tests/`](tests/) | C++ and Python tests |
| [`bench/`](bench/) | Benchmarks |
| [`tools/`](tools/) | Developer tools |
| [`docs/`](docs/) | Documentation |

## License

Apache 2.0. See [`LICENSE`](LICENSE).
