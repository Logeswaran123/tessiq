# Testing

## Layout

- `tests/cpp/` holds the GoogleTest suites for the C++ core. `tests/cpp/golden/` holds pass tests as
  text: a `// RUN:` pipeline, the input IR and the expected output. The suite picks up every file,
  so adding a case is adding a file.
- `tests/python/` holds the pytest suites, including `tests/python/golden/` for the rewrite passes
  that need the Python extension. Tests that need a GPU are marked `cuda`.
- `bench/` holds benchmarks, which are not part of the test suite.

## What runs where

| | CI | A machine with a GPU |
|---|---|---|
| C++ tests, debug and release | yes | yes |
| C++ tests under ASan and UBSan | yes | yes, with `detect_leaks=0` |
| Python tests not marked `cuda` | yes | yes |
| Python tests marked `cuda` | no | `pytest` without `-m` |
| Coverage | yes | `ctest --preset coverage`, `pytest --cov` |

CI has no GPU. On it, the CPU reference device runs the whole compiled path: import, passes,
recipe, executor, placement and transfers. Generated kernels, the cuBLAS, cuDNN and cuBLASLt
bindings, the device copy kernel and precompiled operations run only on a GPU, so run the full
suite on one before sending a change to any of them.

## Conventions

- Compare against PyTorch on the CPU, at a tolerance of 1e-5 unless the test says why it needs more.
- When the result alone cannot tell, check the side effect: the allocator's counters for work on the
  device, a custom operation's fallback count for its kernel running, the PTX for double precision.
- See a new test fail. Break the code it covers and check that it fails, then restore it.
- Use the inputs that break backends: mixed element types, integer reductions, NaN, infinities and
  negative zero, a layout that changes between calls, and a view as the destination of a copy.

## Known gaps

- `unfold`, `view_as_real` and `view_as_complex` run through the CPU fallback and return copies, so
  writes through them do not reach the original tensor.
- `convolution_backward_overrideable` is not registered, so an eager convolution backward pass on a
  tessiq tensor raises.
- cuDNN's workspace is allocated outside the caching allocator and does not appear in its counters.
