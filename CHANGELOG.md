# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CMake build, scikit-build-core packaging, pybind11 extension module.
- Support layer: `TESSIQ_CHECK`/`TESSIQ_DCHECK`, `Error`/`ErrorCode`, `Result<T>` with
  `TESSIQ_RETURN_IF_ERROR` and `TESSIQ_ASSIGN_OR_RETURN`, and module-scoped logging that skips
  argument evaluation when the level is off.
- `ErrorCode` to Python exception mapping in the bindings, so callers can use `except` rather than
  matching on message text.
- Quality gates: `-Werror`, clang-format, clang-tidy, ruff, mypy, ASan/UBSan, and CI running all
  of them.
- `requirements-dev.txt` pinning the tooling. CI and the pre-commit hooks install from it, which
  stops a hook disagreeing with the gate it mirrors.
- Device runtime: a `Device` interface with a CPU reference backend, and a caching allocator with
  size classes, reuse, statistics and `empty_cache()`.
- Registration with PyTorch as the `tessiq` device. `torch.randn(n, device="tessiq")`,
  `.to("tessiq")` and `.cpu()` work, including for non-contiguous tensors and across dtypes.
- A CPU fallback for ops without a native kernel, so the whole ATen surface is usable now and
  kernels can replace it one at a time.
- `tessiq.memory_stats()`, `memory_allocated()`, `memory_reserved()` and `empty_cache()`.
- A CUDA backend, selected per tensor: `tessiq.cpu` is the reference, `tessiq.gpu` the first GPU,
  `tessiq.gpu_at(n)` the rest. Each device has its own allocator, so memory statistics are per
  backend rather than pooled.
- Copies between backends, staged through the host.
- An SSA IR with a textual form: `Graph`, `Node`, `Value`, `TensorType` with dynamic dimensions,
  use-def chains, a verifier, and a printer and parser that round-trip.
- Pass infrastructure: a `Pass` interface, a name-keyed registry, and a `PassManager` that
  verifies the graph after each pass in debug builds.
- `dce`, `cse` and `const-fold`.
- `tessiq-opt`, which runs a named pipeline over textual IR, and golden-file pass tests.
- `tessiq.import_graph(gm)`, which converts the FX graph a `torch.compile` backend receives into
  tessiq IR and returns its textual form.
- Partitioning, as two passes: `mark-supported` records whether the backend can run each node, and
  `cluster` groups neighbouring supported nodes into units, refusing any merge that would make the
  graph unexecutable.
- `tessiq.run_passes(ir, pipeline)` and `tessiq.pass_names()`.
- `shape-infer`, which derives result types from operand types and checks the ones already
  recorded. Resolves dimensions a graph left unknown, and reports a disagreement between two
  known sizes rather than overwriting one with the other.
- `tessiq.partition(gm)`, which imports an FX graph, partitions it, and returns the IR along
  with the cluster each node landed in.
- `tessiq.execute(gm, clusters)`, which runs a partitioned graph: clustered nodes on a tessiq
  device, everything else on the host, results back on the host.
- `torch.compile(model, backend="tessiq")`, declared as a `torch_dynamo_backends` entry point
  so the name resolves without importing tessiq first. Forward and backward both go through
  the compiler. `options={"device": "tessiq:1"}` picks which device clusters run on.
