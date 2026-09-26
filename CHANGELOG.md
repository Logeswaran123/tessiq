# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CMake/scikit-build-core build with a pybind11 extension; CI with `-Werror`, clang-format,
  clang-tidy, ruff, mypy and ASan/UBSan. The C++ core builds without PyTorch.
- `tessiq` device via PrivateUse1: `tessiq:0` (`tessiq.cpu`) is a CPU reference, `tessiq:1+`
  (`tessiq.gpu`, `tessiq.gpu_at(n)`) are CUDA devices. Per-device caching allocators,
  `tessiq.memory_stats()`, `memory_allocated()`, `memory_reserved()`, `empty_cache()`.
- Native allocation, copies, views, `set_` and `clone`; CPU fallback for all other ATen ops.
- `tessiq.current_device()`, `set_device()`, `synchronize()`, and `torch.tessiq.device`.
- PrivateUse1 hooks: pinned memory, `torch.Generator(device="tessiq")` (a CPU generator), storage
  resize, `torch.accelerator.synchronize()`.
- `torch.accelerator` memory functions, including peak/accumulated resets and `get_memory_info`.
- `torch.save`/`torch.load` for tessiq tensors, including `map_location`.
- SSA IR with a textual form, verifier, pass registry and `PassManager`.
- Passes: `dead-node-elimination`, `cse`, `const-fold`, `shape-infer`, `algebraic-simplify`,
  `mark-supported`, `layout-assign`, `cluster`, `fuse-elementwise`.
- Liveness-based arena memory planning.
- `torch.compile(backend="tessiq")` with `device` and `rewrites` options, registered as a
  `torch_dynamo_backends` entry point. Training via AOTAutograd; symbolic shapes.
- `tessiq.capture()`, `import_graph()`, `partition()`, `run_passes()`, `pass_names()`,
  `default_pipeline()`, `build_recipe()`, `execute()`.
- Recipe cache keyed on graph structure, types and device; `cache_stats()`, `clear_cache()`.
- Unchanged inputs stay on the device between calls.
- NVRTC-compiled fused elementwise, reduction and pack kernels, with an on-disk cubin cache
  (`TESSIQ_CACHE_DIR`). `tessiq.emit_cuda()` prints the source.
- cuBLAS matmuls and cuDNN convolution; eval-mode batch norm decomposed.
- GELU as a fused elementwise op (exact only; tanh stays on the host).
- `ops.toml` and `tools/gen_ops.py`, generating the op table and `docs/op_coverage.md`.
- `Kernel` interface for all compiled work, with an explicit stream per launch.
- Custom operations in `csrc/custom_ops/`, as NVRTC kernels, fusable expressions or
  `.precompiled()` nvcc host code. `.replaces()` adds `override-<name>`; `.pattern()` adds
  `rewrite-<name>`. CUTLASS is fetched and cuBLAS/cuBLASLt linked only for files that include them.
- `tessiq.library`: `registered()`, `precompiled()`, `overrides()`, `rewrites()`, `check()`,
  `fallbacks()`.
- Custom ops: `rms_norm`, `softmax`, `fast_gelu`, `swiglu`, `layer_norm`, `gemm_bias_gelu`,
  `gemm_bias_gelu_tanh`.
- Tooling: `tessiq-opt`, `tessiq.explain()` and `tools/tessiq_explain.py`, `tessiq.profile()`
  with Chrome traces and device time, `tessiq.kernel_costs()`, `tools/model_coverage.py`, and
  benchmarks in `bench/`.

### Changed

- `dce` renamed to `dead-node-elimination`.
- A launch function that rejects its arguments with `TORCH_CHECK` now falls back to the reference
  instead of failing the call.

### Fixed

- Profiler counted reused inputs as full transfers.
- `tessiq-explain` showed host ops as running on the device.
- Convolution on the CPU reference device raised instead of running on the host.
- Kernels reading transposed or broadcast values failed to compile.
- String attributes containing commas did not round-trip through the text format.
- float64 matmuls and convolutions were claimed for the GPU, then rejected by cuBLAS/cuDNN.
- Dynamic-shape fusion groups sized their result from the first input, wrong when it broadcast.
- `torch.tessiq.synchronize()` didn't wait, `set_device()` was ignored, and GPU `synchronize()`
  waited only for copies.
- Ops whose output the fallback resizes (e.g. `torch.arange`) failed on a GPU.
- Cloning a lazily cloned GPU tensor without a device guard crashed.
- Compiled elementwise ops and matmuls ignored keyword arguments: `torch.add(a, b, alpha=2)`
  computed `a + b`. Such nodes now run on the host.
